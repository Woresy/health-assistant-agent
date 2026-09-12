"""有限轮次健康 Agent Loop 和多轮会话。"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.agent.context_pipeline import build_prompt_context
from src.agent.models import (
    AgentFinishReason,
    AgentMessage,
    AgentModel,
    AgentRunResult,
    AgentState,
    AgentToolStep,
    AgentTurnOutcome,
    PendingConfirmation,
    PendingTask,
    SessionState,
)
from src.agent.tool_router import (
    HealthToolRouter,
)


SYSTEM_PROMPT = """
你是个人健康管理助理 Agent。

你是一个管理健康事实、健康目标、明确偏好、可信知识与提醒行动的个人健康助理。

你可以协助：记录和查询健康事实；读取及修改用户档案；创建、调整、暂停或恢复
健康目标；生成今日与 7/14/30 天确定性汇总；检索带来源的一般健康知识；创建、
查看、修改、延后、暂停或取消飞书提醒。

规则：
1. 不做诊断，不替代医生，不夸大健康结论。
2. 记录健康事件必须调用 prepare_health_event。
3. 查询健康事件必须调用 get_health_events。
4. 每日汇总必须调用 get_daily_summary，周期趋势必须调用 get_period_summary。
5. 修改或删除健康事件必须调用 prepare_event_change。
6. 档案写入调用 prepare_profile_update，目标写入调用 prepare_goal_change。
7. 一般健康知识必须调用 retrieve_health_knowledge 并在回答中展示来源；医疗、用药、
   急症或证据不足时遵守工具的拒答结果。
8. 提醒先调用 create_reminder_draft；查看、修改或改变提醒调用
   list_or_cancel_reminders。所有提醒都通过飞书发送，不要询问用户选择渠道。
9. 不能只用文本声称已经生成草稿或已经调用工具。
10. 所有写操作必须先生成草稿，等待用户明确确认。
11. 不得调用工具白名单以外的函数。
12. 缺少必填参数时应提出工具调用，由工具校验生成追问。
13. occurred_at 是可选参数；用户未说明时间时省略它，由程序使用当前时间。
14. 汇总只读取已保存事件，不得补数据或推断变化原因。
15. 饮食营养值必须来自候选检索和 calculate_nutrition 的确定性计算。记录饮食时，
   依次调用 retrieve_nutrition_candidates、calculate_nutrition 和
   prepare_health_event；营养来源由工具结果提供，不得向用户索取 source_refs 等内部字段。
16. 只有工具真正返回草稿后，才能告诉用户等待确认。
17. 用户可以用时间、类型和内容指代记录。修改或删除缺少 event_id 时，先调用
   get_health_events 查找候选；多条相似记录时用自然语言请用户进一步说明。修改饮食
   的食物或份量时，也必须重新执行营养候选检索与计算，再把完整结果放入 patch.payload。
18. 面向用户的回答不得展示 UUID、内部 ID、原始 JSON、确认令牌或内部字段名。
19. 教练风格只改变表达，不得改变事实、数值、来源、安全规则或确认要求。
20. “我今天吃了什么”“今天喝了多少”“今天记录了什么”等问句属于查询，调用
   get_health_events 或 get_daily_summary，不得误建为新增草稿。
21. 用户明确要求飞书通知时，create_reminder_draft 使用 delivery_channel=feishu；
   Webhook 和签名密钥只由应用配置提供，不得向用户索取或在回答中展示。
22. 主动 check-in 复用 create_reminder_draft，使用 reminder_type=check_in、
   recurrence=daily 或 weekdays、delivery_channel=feishu；它只询问已确认记录是否需要
   补充，不得把缺少记录表述成用户没有完成，也不得自动写入健康事实。
23. 多轮对话中应解析“刚才那条”“改成”“不是”等上下文指代。用户修订待确认草稿时，
   旧草稿立即失效，必须基于修订内容重新调用准备工具并再次等待确认。
24. 系统会提供当前“连续对话场景”。场景策略负责组织多轮服务节奏，但不能覆盖安全、
   工具、事实来源和确认规则；省略式追问应延续最近场景，明确切换主题时跟随用户新意图。
25. 用户要求修改已有提醒时，必须先查看提醒，再使用
   list_or_cancel_reminders 的 update 操作原地修改；不得通过新建提醒替代修改。
""".strip()


_TOOL_ACTION_TERMS = (
    "记录",
    "记一下",
    "录入",
    "保存",
    "新增",
    "添加",
    "修改",
    "更新",
    "改成",
    "改为",
    "删除",
    "移除",
    "查询",
    "查一下",
    "查看",
    "时间线",
    "汇总",
    "统计",
    "多少",
    "目标",
    "提醒",
    "偏好",
    "档案",
    "周报",
    "趋势",
    "建议",
    "怎么",
    "如何",
    "应该",
)

_HEALTH_DOMAIN_TERMS = (
    "饮食",
    "吃饭",
    "早餐",
    "午餐",
    "晚餐",
    "食物",
    "喝水",
    "饮水",
    "毫升",
    "体重",
    "公斤",
    "千克",
    "运动",
    "跑步",
    "步行",
    "公里",
    "分钟",
    "健康记录",
    "健康事件",
    "事件",
    "目标",
    "提醒",
    "教练风格",
    "营养",
    "健康建议",
    "睡眠",
    "失眠",
    "作息",
    "疲劳",
    "胸痛",
    "胸闷",
    "呼吸困难",
    "过敏",
)

_IMPLICIT_RECORD_TERMS = (
    "喝了",
    "吃了",
    "跑步了",
    "运动了",
    "步行了",
    "走了",
    "称重",
    "胸痛",
    "胸闷",
    "呼吸困难",
    "晕厥",
)

_IMPLICIT_KNOWLEDGE_TERMS = (
    "睡不着",
    "失眠",
    "睡眠不好",
    "白天很困",
    "严重过敏",
)

TOOL_REQUIRED_RETRY_PROMPT = """
上一条响应没有调用工具，因此不能作为有效结果。
当前用户请求涉及健康事件的记录、查询、修改、删除或汇总，
也可能涉及档案、目标、可信知识、安全分流或提醒。你必须调用匹配的白名单工具。
不要通过普通文本声称已经生成草稿或已经保存。
如果缺少必填参数，也应提出工具调用，由工具校验生成追问。
""".strip()


SEQUENTIAL_TOOL_RETRY_PROMPT = """
上一条响应同时提出了多个工具调用，当前编排器会按顺序执行工具。
请一次只选择一个工具，先调用当前最需要的那个；获得结果后，
再在下一轮决定是否调用其他工具。不要向用户提及这项内部约束。
""".strip()


_INTERNAL_FOOD_REFERENCE = re.compile(
    r"\s*[（(]\s*(?:[a-z]+:)?food[_-]?\d+\s*[）)]",
    flags=re.IGNORECASE,
)
_INTERNAL_FOOD_CODE = re.compile(
    r"(?<![\w:])(?:[a-z]+:)?food[_-]?\d+(?!\w)",
    flags=re.IGNORECASE,
)


def _sanitize_user_answer(answer: str) -> str:
    """从最终用户文案中移除内部编号和实现术语。"""

    sanitized = _INTERNAL_FOOD_REFERENCE.sub("", answer)
    sanitized = _INTERNAL_FOOD_CODE.sub("标准食物条目", sanitized)
    sanitized = re.sub(
        r"ISO\s*8601(?:\s*格式)?",
        "清楚的日期和时间",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"[ \t]+([，。；：！？])", r"\1", sanitized)
    return sanitized.strip()


_TEXT_CONFIRMATIONS = {
    "确认", "确认保存", "确认执行", "可以", "没问题", "就这样", "保存吧",
}
_TEXT_CANCELLATIONS = {
    "取消", "取消操作", "不要了", "算了", "先不保存", "不保存",
}
_DRAFT_REVISION_TERMS = (
    "改成", "改为", "修改为", "更正", "纠正", "调整为", "不是", "应该是",
    "换成", "少了", "多了", "补充",
)


def _pending_confirmation_intent(user_text: str) -> str | None:
    """识别待确认阶段的文本确认、取消或草稿修订。"""

    normalized = user_text.strip().casefold().strip("，。！？!?；; ")
    if normalized in _TEXT_CONFIRMATIONS:
        return "confirm"
    if normalized in _TEXT_CANCELLATIONS:
        return "cancel"
    if any(term in normalized for term in _DRAFT_REVISION_TERMS):
        return "revise"
    return None


_RELATIVE_REMINDER_PATTERN = re.compile(
    r"(?P<amount>\d{1,4}|[一二两三四五六七八九十]{1,3})\s*"
    r"(?P<unit>分钟|小时)\s*后\s*"
    r"(?:(?:在|通过|用)\s*)?(?P<channel>飞书|本地)?(?:上)?\s*"
    r"提醒我\s*(?P<content>.+)"
)

_PREFIX_RELATIVE_REMINDER_PATTERN = re.compile(
    r"(?:(?:通过|用)\s*)?(?P<channel>飞书|本地)?\s*"
    r"提醒我\s*(?:在\s*)?"
    r"(?P<amount>\d{1,4}|[一二两三四五六七八九十]{1,3})\s*"
    r"(?P<unit>分钟|小时)\s*后\s*"
    r"(?P<content>.+)"
)

_RECURRING_REMINDER_PATTERN = re.compile(
    r"(?P<recurrence>每天|工作日)\s*"
    r"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*"
    r"(?P<hour>\d{1,2})\s*点\s*"
    r"(?:(?P<half>半)|(?P<minute>\d{1,2})\s*分?)?\s*"
    r"(?:(?:在|通过|用)\s*飞书(?:上)?\s*)?"
    r"提醒我\s*(?P<content>.+)"
)


def _duration_number(raw_value: str) -> int | None:
    """解析提醒快路径需要的小型中文整数。"""

    if raw_value.isdigit():
        value = int(raw_value)
        return value if value > 0 else None

    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if raw_value in digits:
        return digits[raw_value]
    if raw_value == "十":
        return 10
    if "十" in raw_value:
        tens_text, ones_text = raw_value.split("十", 1)
        tens = digits.get(tens_text, 1) if tens_text else 1
        ones = digits.get(ones_text, 0) if ones_text else 0
        return tens * 10 + ones
    return None


def _try_direct_relative_reminder(
    *,
    session_state: SessionState,
    user_text: str,
    router: HealthToolRouter,
) -> AgentTurnOutcome | None:
    """明确的相对、周期提醒和主动 check-in 走确定性草稿路径。"""

    normalized_text = user_text.strip()
    is_check_in = (
        "飞书" in normalized_text
        and ("主动" in normalized_text or "check-in" in normalized_text.casefold())
        and ("每天" in normalized_text or "工作日" in normalized_text)
    )
    if is_check_in:
        time_match = re.search(
            r"(?P<period>上午|下午|晚上)?\s*(?P<hour>\d{1,2})\s*点"
            r"(?:\s*(?P<minute>\d{1,2})\s*分)?",
            normalized_text,
        )
        if time_match is None:
            return None
        hour = int(time_match.group("hour"))
        minute = int(time_match.group("minute") or 0)
        period = time_match.group("period") or ""
        if period in {"下午", "晚上"} and hour < 12:
            hour += 12
        if period == "上午" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            return None
        active_timezone = ZoneInfo(session_state.timezone_name)
        now = datetime.now(active_timezone)
        scheduled_for = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        recurrence = "weekdays" if "工作日" in normalized_text else "daily"
        while scheduled_for <= now or (
            recurrence == "weekdays" and scheduled_for.weekday() >= 5
        ):
            scheduled_for += timedelta(days=1)
        focus_labels = {
            "饮食": "meal",
            "吃": "meal",
            "饮水": "water",
            "喝水": "water",
            "运动": "exercise",
            "体重": "weight",
        }
        check_in_focus = list(
            dict.fromkeys(
                value for label, value in focus_labels.items() if label in normalized_text
            )
        ) or ["meal", "water", "exercise"]
        content = "主动健康 check-in"
        arguments = {
            "content": content,
            "scheduled_for": scheduled_for.isoformat(),
            "timezone_name": session_state.timezone_name,
            "delivery_channel": "feishu",
            "reminder_type": "check_in",
            "recurrence": recurrence,
            "check_in_focus": check_in_focus,
        }
    else:
        recurring_match = _RECURRING_REMINDER_PATTERN.search(normalized_text)
        if recurring_match is not None:
            hour = int(recurring_match.group("hour"))
            minute = (
                30
                if recurring_match.group("half")
                else int(recurring_match.group("minute") or 0)
            )
            period = recurring_match.group("period") or ""
            if period in {"中午", "下午", "傍晚", "晚上"} and hour < 12:
                hour += 12
            if period in {"凌晨", "早上", "上午"} and hour == 12:
                hour = 0
            if hour > 23 or minute > 59:
                return None
            active_timezone = ZoneInfo(session_state.timezone_name)
            now = datetime.now(active_timezone)
            scheduled_for = now.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            recurrence = (
                "weekdays"
                if recurring_match.group("recurrence") == "工作日"
                else "daily"
            )
            while scheduled_for <= now or (
                recurrence == "weekdays" and scheduled_for.weekday() >= 5
            ):
                scheduled_for += timedelta(days=1)
            content = recurring_match.group("content").strip(
                " ，。！？!?；;"
            )
            if not content:
                return None
            arguments = {
                "content": content,
                "scheduled_for": scheduled_for.isoformat(),
                "timezone_name": session_state.timezone_name,
                "delivery_channel": "feishu",
                "recurrence": recurrence,
            }
        else:
            matched = _RELATIVE_REMINDER_PATTERN.search(
                normalized_text
            ) or _PREFIX_RELATIVE_REMINDER_PATTERN.search(
                normalized_text
            )
            if matched is None:
                return None
            amount = _duration_number(matched.group("amount"))
            if amount is None:
                return None
            delta = (
                timedelta(minutes=amount)
                if matched.group("unit") == "分钟"
                else timedelta(hours=amount)
            )
            scheduled_for = (
                datetime.now(ZoneInfo(session_state.timezone_name)) + delta
            ).replace(microsecond=0)
            content = matched.group("content").strip(" ，。！？!?；;")
            if not content:
                return None
            arguments = {
                "content": content,
                "scheduled_for": scheduled_for.isoformat(),
                "timezone_name": session_state.timezone_name,
                "delivery_channel": "feishu",
            }

    call_id = f"direct-reminder-{session_state.turn_count + 1}"
    dispatch = router.dispatch(
        tool_name="create_reminder_draft",
        arguments=arguments,
        user_id=session_state.user_id,
        timezone_name=session_state.timezone_name,
        session_id=session_state.session_id,
        call_id=call_id,
    )
    if dispatch.status != "executed" or dispatch.result is None:
        return None

    result_data = dispatch.result.get("data")
    if dispatch.result.get("ok") and isinstance(result_data, dict):
        pending = PendingConfirmation(
            action="reminder_create",
            tool_name="create_reminder_draft",
            draft_data=result_data,
        )
        answer = _preview_answer(result_data)
        state = AgentState.AWAITING_CONFIRMATION
        finish_reason = AgentFinishReason.AWAITING_CONFIRMATION
    else:
        pending = None
        error = dispatch.result.get("error")
        answer = (
            str(error.get("message", "提醒草稿生成失败"))
            if isinstance(error, dict)
            else "提醒草稿生成失败"
        )
        state = AgentState.FAILED
        finish_reason = AgentFinishReason.TOOL_ERROR

    messages = (
        *session_state.messages,
        AgentMessage(role="user", content=user_text.strip()),
        AgentMessage(role="assistant", content=answer),
    )
    step = AgentToolStep(
        call_id=call_id,
        tool_name="create_reminder_draft",
        arguments=arguments,
        result=_redact_result(dispatch.result),
    )
    new_state = SessionState(
        session_id=session_state.session_id,
        user_id=session_state.user_id,
        timezone_name=session_state.timezone_name,
        state=state,
        messages=messages,
        turn_count=session_state.turn_count + 1,
        pending_confirmation=pending,
    )
    result = AgentRunResult(
        answer=answer,
        finish_reason=finish_reason,
        state=state,
        model_rounds=0,
        tool_steps=(step,),
        pending_confirmation=pending,
    )
    return AgentTurnOutcome(result=result, session_state=new_state)


def _requires_health_tool(
    user_text: str,
) -> bool:
    """判断当前用户输入是否必须经过健康工具。"""

    normalized_text = (
        user_text.strip()
        .casefold()
    )

    if not normalized_text:
        return False

    has_action_term = any(
        term in normalized_text
        for term in _TOOL_ACTION_TERMS
    )
    has_health_term = any(
        term in normalized_text
        for term in _HEALTH_DOMAIN_TERMS
    )

    if has_action_term and has_health_term:
        return True

    if any(
        term in normalized_text
        for term in _IMPLICIT_RECORD_TERMS
    ):
        return True

    if any(
        term in normalized_text
        for term in _IMPLICIT_KNOWLEDGE_TERMS
    ):
        return True

    has_number = any(
        character.isdigit()
        for character in normalized_text
    )
    has_health_unit = any(
        unit in normalized_text
        for unit in (
            "毫升",
            "ml",
            "公斤",
            "kg",
            "千克",
            "分钟",
            "公里",
            "km",
        )
    )

    return has_number and has_health_unit


def _redact_result(
    result: dict[str, object],
) -> dict[str, object]:
    """从 ToolStep 展示结果中移除确认令牌。"""

    redacted = deepcopy(
        result
    )

    data = redacted.get("data")

    if isinstance(data, dict):
        if (
            "confirmation_token"
            in data
        ):
            data[
                "confirmation_token"
            ] = "***redacted***"

    return redacted


def _tool_request_message(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> AgentMessage:
    """保存模型提出的工具调用。"""

    return AgentMessage(
        role="assistant",
        content=json.dumps(
            {
                "tool_call": {
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": (
                        arguments
                    ),
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_call_id=call_id,
        tool_name=tool_name,
    )


def _tool_result_message(
    *,
    call_id: str,
    tool_name: str,
    result: dict[str, object],
) -> AgentMessage:
    """把工具结果交回模型。"""

    return AgentMessage(
        role="tool",
        content=json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_call_id=call_id,
        tool_name=tool_name,
    )


def _display_number(value: object) -> str:
    """把健康数值转成简洁、稳定的展示文本。"""

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return f"{value:g}"

    return str(value or "").strip()


def health_event_type(
    event_data: object,
) -> str:
    """从草稿预览或完整 HealthEvent 中提取事件类型。"""

    if not isinstance(event_data, dict):
        return ""

    value = event_data.get("event_type")
    return (
        str(value).strip().lower()
        if value is not None
        else ""
    )


def health_event_label(
    event_data: object,
) -> str:
    """返回适合普通用户阅读的事件类型名称。"""

    return {
        "meal": "饮食",
        "water": "饮水",
        "weight": "体重",
        "exercise": "运动",
    }.get(
        health_event_type(event_data),
        "健康",
    )


def format_health_event_summary(
    event_data: object,
) -> str:
    """将草稿预览或完整 HealthEvent 转为自然语言摘要。"""

    if not isinstance(event_data, dict):
        return "健康记录"

    event_type = health_event_type(
        event_data
    )
    raw_payload = event_data.get(
        "payload"
    )
    payload = (
        raw_payload
        if isinstance(raw_payload, dict)
        else event_data
    )

    if event_type == "meal":
        raw_food = payload.get("food")
        food = (
            raw_food.get("name", "饮食")
            if isinstance(raw_food, dict)
            else raw_food or "饮食"
        )
        raw_portion = payload.get("portion")
        grams = (
            raw_portion.get("grams")
            if isinstance(raw_portion, dict)
            else payload.get("grams")
        )
        raw_nutrition = payload.get(
            "nutrition"
        )
        calories = (
            raw_nutrition.get(
                "calories_kcal"
            )
            if isinstance(
                raw_nutrition,
                dict,
            )
            else payload.get(
                "calories_kcal"
            )
        )
        details = [str(food)]
        if grams is not None:
            details.append(
                f"{_display_number(grams)} g"
            )
        if calories is not None:
            details.append(
                f"约 {_display_number(calories)} kcal"
            )
        return "，".join(details)

    if event_type == "water":
        beverage = str(
            payload.get("beverage")
            or "饮用水"
        )
        if beverage.strip().casefold() in {"water", "plain water"}:
            beverage = "饮用水"
        amount = payload.get("amount_ml")
        summary = beverage
        if amount is not None:
            summary += (
                f" {_display_number(amount)} ml"
            )
        note = str(
            payload.get("note") or ""
        ).strip()
        return (
            f"{summary}，{note}"
            if note
            else summary
        )

    if event_type == "weight":
        weight = payload.get("weight_kg")
        summary = (
            "体重"
            if weight is None
            else (
                "体重 "
                f"{_display_number(weight)} kg"
            )
        )
        note = str(
            payload.get("note") or ""
        ).strip()
        return (
            f"{summary}，{note}"
            if note
            else summary
        )

    if event_type == "exercise":
        activity = str(
            payload.get("activity_type")
            or "运动"
        )
        duration = payload.get(
            "duration_minutes"
        )
        distance = payload.get(
            "distance_km"
        )
        intensity = payload.get("intensity")
        note = str(
            payload.get("note") or ""
        ).strip()
        details = [activity]
        if duration is not None:
            details.append(
                f"{_display_number(duration)} 分钟"
            )
        if distance is not None:
            details.append(
                f"{_display_number(distance)} km"
            )
        intensity_label = {
            "low": "低强度",
            "medium": "中等强度",
            "high": "高强度",
        }.get(str(intensity), "")
        if intensity_label:
            details.append(intensity_label)
        if note:
            details.append(note)
        return "，".join(details)

    return "健康记录"


def _display_reminder_time(value: object, timezone_name: str) -> str:
    """把提醒时间转成用户熟悉的本地表达。"""

    try:
        parsed = datetime.fromisoformat(str(value))
        return parsed.astimezone(ZoneInfo(timezone_name)).strftime(
            "%m月%d日 %H:%M"
        )
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        return str(value or "时间待确认")


def _preview_answer(
    draft_data: dict[str, object],
) -> str:
    """将副作用草稿转成普通用户可读的待确认文本。"""

    action = draft_data.get("action")

    if action == "save":
        preview = draft_data.get(
            "preview",
            {},
        )
        summary = format_health_event_summary(
            preview
        )
        return (
            "我已经整理好这条记录，"
            "目前还没有保存。\n\n"
            f"**{summary}**\n\n"
            "请核对内容，然后确认保存。"
        )

    if action == "update":
        current_event = draft_data.get(
            "current_event",
            {},
        )
        proposed_event = draft_data.get(
            "proposed_event",
            {},
        )
        return (
            "我已经整理好修改内容，"
            "目前还没有更改原记录。\n\n"
            "**修改前**  "
            f"{format_health_event_summary(current_event)}\n\n"
            "**修改后**  "
            f"{format_health_event_summary(proposed_event)}\n\n"
            "请核对内容，然后确认修改。"
        )

    if action == "delete":
        target_event = draft_data.get(
            "target_event",
            {},
        )
        return (
            "我找到了要删除的记录，"
            "目前还没有执行删除。\n\n"
            "**"
            f"{format_health_event_summary(target_event)}"
            "**\n\n"
            "请核对内容，然后确认删除。"
        )

    preview = draft_data.get("preview", {})
    if not isinstance(preview, dict):
        preview = {}

    if action == "profile_update":
        after = preview.get("after", {})
        if not isinstance(after, dict):
            after = {}
        style = {
            "gentle": "温和陪伴",
            "rational": "理性复盘",
            "concise": "简洁提醒",
            "goal_focused": "目标督促",
        }.get(str(after.get("coach_style", "")), "保持当前风格")
        return (
            "我已经整理好档案变更，目前还没有写入。\n\n"
            f"**教练风格：{style}**\n\n"
            "偏好只会在你确认后保存，并且可以随时再次修改。"
        )

    if action == "goal_change":
        after = preview.get("after", {})
        if not isinstance(after, dict):
            after = {}
        return (
            "我已经整理好目标草稿，目前还没有生效。\n\n"
            f"**{after.get('title', '健康目标')}**  "
            f"{_display_number(after.get('target_value'))} "
            f"{after.get('unit', '')} · {after.get('period', '')}\n\n"
            f"调整原因：{after.get('reason', '用户主动设置')}。"
            "确认后会新增一个版本，旧版本仍然保留。"
        )

    if action == "reminder_create":
        destination = preview.get("destination_label", "飞书")
        scheduled_text = _display_reminder_time(
            preview.get("scheduled_for"),
            str(preview.get("timezone_name", "Asia/Shanghai")),
        )
        if preview.get("reminder_type") == "check_in":
            recurrence = "工作日" if preview.get("recurrence") == "weekdays" else "每天"
            focus_labels = {
                "meal": "饮食",
                "water": "饮水",
                "exercise": "运动",
                "weight": "体重",
            }
            focus = "、".join(
                focus_labels.get(str(item), str(item))
                for item in preview.get("check_in_focus", [])
            )
            return (
                "主动问候草稿已经准备好，目前还没有启用。\n\n"
                f"**{recurrence}主动询问：{focus or '饮食、饮水和运动'}**  "
                f"首次发送 {scheduled_text}\n\n"
                f"发送到：{destination}。"
                "届时只根据已确认记录询问是否需要补记，不会自动写入；确认后才启用。"
            )
        return (
            "提醒草稿已经准备好，目前还没有安排。\n\n"
            f"**{preview.get('content', '健康提醒')}**  "
            f"{scheduled_text}\n\n"
            f"发送到：{destination}。"
            "确认后只会创建一次。"
        )

    if action == "reminder_change":
        operation = {
            "update": "修改",
            "cancel": "取消",
            "snooze": "延后",
            "pause": "暂停",
            "resume": "恢复",
        }.get(str(preview.get("operation", "")), "修改")
        after = preview.get("after", {})
        updated_summary = ""
        if operation == "修改" and isinstance(after, dict):
            updated_summary = (
                f"\n\n**{after.get('content', '健康提醒')}**  "
                f"{_display_reminder_time(after.get('scheduled_for'), str(after.get('timezone_name', 'Asia/Shanghai')))}"
            )
        return (
            f"我已经准备好{operation}这条提醒，目前还没有执行。\n\n"
            f"{updated_summary}"
            "确认后提醒状态才会改变；取消草稿则保持原状态。"
        )

    return "操作草稿已经准备好，请核对后确认。"


class AgentRunner:
    """协调模型、pending_task 和工具 Router。"""

    def __init__(
        self,
        *,
        model: AgentModel,
        router: HealthToolRouter,
        max_model_rounds: int = 4,
    ) -> None:
        if max_model_rounds <= 0:
            raise ValueError(
                "max_model_rounds "
                "必须大于 0"
            )

        self._model = model
        self._router = router
        self._max_model_rounds = (
            max_model_rounds
        )

    @property
    def router(
        self,
    ) -> HealthToolRouter:
        """供确认操作使用同一个 Router。"""

        return self._router

    def create_session_state(
        self,
        *,
        session_id: str,
        user_id: str,
        timezone_name: str = (
            "Asia/Shanghai"
        ),
    ) -> SessionState:
        """创建新会话状态。"""

        return SessionState(
            session_id=session_id,
            user_id=user_id,
            timezone_name=(
                timezone_name
            ),
            messages=(
                AgentMessage(
                    role="system",
                    content=SYSTEM_PROMPT,
                ),
            ),
        )

    def run_turn(
        self,
        *,
        session_state: SessionState,
        user_text: str,
    ) -> AgentTurnOutcome:
        """执行一个用户对话回合。"""

        normalized_text = (
            user_text.strip()
        )

        if not normalized_text:
            result = AgentRunResult(
                answer="请输入内容。",
                finish_reason=(
                    AgentFinishReason
                    .INVALID_ARGUMENTS
                ),
                state=(
                    session_state.state
                ),
                model_rounds=0,
                pending_task=(
                    session_state
                    .pending_task
                ),
                pending_confirmation=(
                    session_state
                    .pending_confirmation
                ),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        if (
            session_state
            .pending_confirmation
            is not None
        ):
            pending_intent = _pending_confirmation_intent(normalized_text)
            if pending_intent == "confirm":
                return self.confirm_pending(session_state)
            if pending_intent == "cancel":
                return self.cancel_pending(session_state)
            if pending_intent == "revise":
                revision_state = session_state.model_copy(
                    update={
                        "state": AgentState.RUNNING,
                        "pending_confirmation": None,
                    }
                )
                return self.run_turn(
                    session_state=revision_state,
                    user_text=normalized_text,
                )
            result = AgentRunResult(
                answer=(
                    "当前已有待确认操作。"
                    "请先确认或取消，也可以直接说明要改成什么；"
                    "不会继续执行新的写操作。"
                ),
                finish_reason=(
                    AgentFinishReason
                    .AWAITING_CONFIRMATION
                ),
                state=(
                    AgentState
                    .AWAITING_CONFIRMATION
                ),
                model_rounds=0,
                pending_confirmation=(
                    session_state
                    .pending_confirmation
                ),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        direct_reminder = _try_direct_relative_reminder(
            session_state=session_state,
            user_text=normalized_text,
            router=self._router,
        )
        if direct_reminder is not None:
            return direct_reminder

        messages = list(session_state.messages)
        messages.append(
            AgentMessage(
                role="user",
                content=normalized_text,
            )
        )

        tool_steps: list[
            AgentToolStep
        ] = []

        pending_task = (
            session_state.pending_task
        )

        for model_round in range(
            1,
            self._max_model_rounds
            + 1,
        ):
            current_context = self._router.minimal_user_context_data(
                user_id=session_state.user_id,
                timezone_name=session_state.timezone_name,
            )
            if messages and messages[0].role == "system":
                prompt_context = build_prompt_context(
                    system_rules=SYSTEM_PROMPT,
                    user_input=normalized_text,
                    profile_context=current_context,
                    pending_task=pending_task,
                    messages=messages,
                )
                messages[0] = AgentMessage(
                    role="system",
                    content=prompt_context.render_system_message(),
                )
            reply = self._model.complete(
                messages,
                self._router.tool_definitions_for(
                    normalized_text,
                    pending_tool_name=(
                        pending_task.tool_name
                        if pending_task is not None
                        else None
                    ),
                ),
            )

            if not reply.tool_calls:
                assert (
                    reply.content
                    is not None
                )

                answer = _sanitize_user_answer(reply.content.strip())

                requires_tool = (
                    pending_task
                    is not None
                    or (
                        not tool_steps
                        and _requires_health_tool(
                            normalized_text
                        )
                    )
                )

                if requires_tool:
                    if (
                        model_round
                        < self._max_model_rounds
                    ):
                        messages.append(
                            AgentMessage(
                                role="system",
                                content=(
                                    TOOL_REQUIRED_RETRY_PROMPT
                                ),
                            )
                        )

                        continue

                    return self._failed_outcome(
                        session_state=(
                            session_state
                        ),
                        messages=messages,
                        answer=(
                            "模型没有按照协议调用"
                            "健康工具，本轮操作已终止。"
                            "没有写入或修改健康记录。"
                        ),
                        finish_reason=(
                            AgentFinishReason
                            .TOOL_ERROR
                        ),
                        model_rounds=(
                            model_round
                        ),
                        tool_steps=tool_steps,
                    )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=answer,
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .COMPLETED
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                    )
                )

                result = AgentRunResult(
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .COMPLETED
                    ),
                    state=(
                        AgentState
                        .COMPLETED
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            if len(
                reply.tool_calls
            ) != 1:
                if model_round < self._max_model_rounds:
                    messages.append(
                        AgentMessage(
                            role="system",
                            content=SEQUENTIAL_TOOL_RETRY_PROMPT,
                        )
                    )
                    continue

                return self._failed_outcome(
                    session_state=session_state,
                    messages=messages,
                    answer=(
                        "这次没能完成请求，也没有写入或修改"
                        "任何健康数据。请稍后再试。"
                    ),
                    finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
                    model_rounds=model_round,
                    tool_steps=tool_steps,
                )

            tool_call = (
                reply.tool_calls[0]
            )

            merged_arguments = dict(
                tool_call.arguments
            )

            if (
                pending_task
                is not None
                and pending_task
                .tool_name
                == tool_call.name
            ):
                merged_arguments = {
                    **pending_task.arguments,
                    **tool_call.arguments,
                }

            dispatch = (
                self._router.dispatch(
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                    user_id=(
                        session_state
                        .user_id
                    ),
                    timezone_name=(
                        session_state
                        .timezone_name
                    ),
                    session_id=(
                        session_state
                        .session_id
                    ),
                    call_id=(
                        tool_call.call_id
                    ),
                )
            )

            if (
                dispatch.status
                == "needs_clarification"
            ):
                assert (
                    dispatch.question
                    is not None
                )

                pending = PendingTask(
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                    missing_parameters=list(
                        dispatch
                        .missing_parameters
                    ),
                    question=(
                        dispatch.question
                    ),
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=(
                            dispatch.question
                        ),
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .AWAITING_CLARIFICATION
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                        pending_task=pending,
                    )
                )

                result = AgentRunResult(
                    answer=(
                        dispatch.question
                    ),
                    finish_reason=(
                        AgentFinishReason
                        .NEEDS_CLARIFICATION
                    ),
                    state=(
                        AgentState
                        .AWAITING_CLARIFICATION
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                    pending_task=pending,
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            if (
                dispatch.status
                == "invalid"
            ):
                assert (
                    dispatch.result
                    is not None
                )

                error = (
                    dispatch.result.get(
                        "error"
                    )
                )

                if isinstance(
                    error,
                    dict,
                ):
                    answer = _sanitize_user_answer(str(
                        error.get(
                            "message",
                            "工具参数无效",
                        )
                    ))
                else:
                    answer = (
                        "工具参数无效"
                    )

                return self._failed_outcome(
                    session_state=(
                        session_state
                    ),
                    messages=messages,
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .INVALID_ARGUMENTS
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tool_steps,
                )

            assert (
                dispatch.result
                is not None
            )

            tool_step = AgentToolStep(
                call_id=(
                    tool_call.call_id
                ),
                tool_name=(
                    tool_call.name
                ),
                arguments=(
                    merged_arguments
                ),
                result=_redact_result(
                    dispatch.result
                ),
            )

            tool_steps.append(
                tool_step
            )

            if not dispatch.result.get(
                "ok"
            ):
                error = (
                    dispatch.result.get(
                        "error"
                    )
                )

                if isinstance(
                    error,
                    dict,
                ):
                    answer = _sanitize_user_answer(str(
                        error.get(
                            "message",
                            "工具执行失败",
                        )
                    ))
                else:
                    answer = (
                        "工具执行失败"
                    )

                return self._failed_outcome(
                    session_state=(
                        session_state
                    ),
                    messages=messages,
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .TOOL_ERROR
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tool_steps,
                )

            result_data = (
                dispatch.result.get(
                    "data"
                )
            )

            draft_tools = {
                "prepare_health_event",
                "prepare_event_change",
                "prepare_update_health_event",
                "prepare_delete_health_event",
                "prepare_profile_update",
                "prepare_goal_change",
                "create_reminder_draft",
                "list_or_cancel_reminders",
            }
            if (
                tool_call.name in draft_tools
                and isinstance(
                    result_data,
                    dict,
                )
                and result_data.get("action") in {
                    "save",
                    "update",
                    "delete",
                    "profile_update",
                    "goal_change",
                    "reminder_create",
                    "reminder_change",
                }
            ):
                action = str(
                    result_data[
                        "action"
                    ]
                )

                pending_confirmation = (
                    PendingConfirmation(
                        action=action,
                        tool_name=(
                            tool_call.name
                        ),
                        draft_data=(
                            result_data
                        ),
                    )
                )

                answer = (
                    _preview_answer(
                        result_data
                    )
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=answer,
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .AWAITING_CONFIRMATION
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                        pending_confirmation=(
                            pending_confirmation
                        ),
                    )
                )

                result = AgentRunResult(
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .AWAITING_CONFIRMATION
                    ),
                    state=(
                        AgentState
                        .AWAITING_CONFIRMATION
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                    pending_confirmation=(
                        pending_confirmation
                    ),
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            messages.append(
                _tool_request_message(
                    call_id=(
                        tool_call.call_id
                    ),
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                )
            )

            messages.append(
                _tool_result_message(
                    call_id=(
                        tool_call.call_id
                    ),
                    tool_name=(
                        tool_call.name
                    ),
                    result=(
                        dispatch.result
                    ),
                )
            )

            pending_task = None

        return self._failed_outcome(
            session_state=session_state,
            messages=messages,
            answer=(
                "Agent 达到最大模型轮数，"
                "仍未产生最终回答。"
            ),
            finish_reason=(
                AgentFinishReason
                .LOOP_LIMIT
            ),
            model_rounds=(
                self._max_model_rounds
            ),
            tool_steps=tool_steps,
        )

    def _failed_outcome(
        self,
        *,
        session_state: SessionState,
        messages: list[
            AgentMessage
        ],
        answer: str,
        finish_reason: (
            AgentFinishReason
        ),
        model_rounds: int,
        tool_steps: list[
            AgentToolStep
        ],
    ) -> AgentTurnOutcome:
        """统一构造失败结果。"""

        messages.append(
            AgentMessage(
                role="assistant",
                content=answer,
            )
        )

        new_state = SessionState(
            session_id=(
                session_state.session_id
            ),
            user_id=(
                session_state.user_id
            ),
            timezone_name=(
                session_state
                .timezone_name
            ),
            state=AgentState.FAILED,
            messages=tuple(messages),
            turn_count=(
                session_state.turn_count
                + 1
            ),
        )

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                finish_reason
            ),
            state=AgentState.FAILED,
            model_rounds=(
                model_rounds
            ),
            tool_steps=tuple(
                tool_steps
            ),
        )

        return AgentTurnOutcome(
            result=result,
            session_state=new_state,
        )

    def confirm_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        """明确确认当前待执行草稿。"""

        pending = (
            session_state
            .pending_confirmation
        )

        if pending is None:
            result = AgentRunResult(
                answer="当前没有待确认操作。",
                finish_reason=(
                    AgentFinishReason
                    .INVALID_ARGUMENTS
                ),
                state=(
                    session_state.state
                ),
                model_rounds=0,
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        tool_result = (
            self._router.confirm(
                pending
            )
        )

        execution_tool_names = {
            "save": "save_health_event",
            "update": (
                "update_health_event"
            ),
            "delete": (
                "delete_health_event"
            ),
            "profile_update": "prepare_profile_update",
            "goal_change": "prepare_goal_change",
            "reminder_create": "execute_reminder",
            "reminder_change": "list_or_cancel_reminders",
        }

        step = AgentToolStep(
            call_id=(
                "user-confirmation"
            ),
            tool_name=(
                execution_tool_names[
                    pending.action
                ]
            ),
            arguments={
                "confirmed": True,
                "action": (
                    pending.action
                ),
            },
            result=tool_result,
        )

        if tool_result.get("ok"):
            answer = {
                "save": (
                    "健康事件已确认保存。"
                ),
                "update": (
                    "健康事件已确认修改。"
                ),
                "delete": (
                    "健康事件已确认删除。"
                ),
                "profile_update": "个人档案已确认更新。",
                "goal_change": "健康目标已确认更新，历史版本已保留。",
                "reminder_create": "提醒已确认安排。",
                "reminder_change": "提醒状态已确认更新。",
            }[pending.action]

            messages = (
                *session_state.messages,
                AgentMessage(
                    role="assistant",
                    content=answer,
                ),
            )

            new_state = SessionState(
                session_id=(
                    session_state
                    .session_id
                ),
                user_id=(
                    session_state.user_id
                ),
                timezone_name=(
                    session_state
                    .timezone_name
                ),
                state=(
                    AgentState.COMPLETED
                ),
                messages=messages,
                turn_count=(
                    session_state
                    .turn_count
                ),
            )

            result = AgentRunResult(
                answer=answer,
                finish_reason=(
                    AgentFinishReason
                    .COMPLETED
                ),
                state=(
                    AgentState.COMPLETED
                ),
                model_rounds=0,
                tool_steps=(step,),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    new_state
                ),
            )

        error = tool_result.get(
            "error"
        )

        if isinstance(error, dict):
            answer = str(
                error.get(
                    "message",
                    "确认执行失败",
                )
            )
        else:
            answer = "确认执行失败"

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                AgentFinishReason
                .TOOL_ERROR
            ),
            state=(
                AgentState
                .AWAITING_CONFIRMATION
            ),
            model_rounds=0,
            tool_steps=(step,),
            pending_confirmation=(
                pending
            ),
        )

        return AgentTurnOutcome(
            result=result,
            session_state=(
                session_state
            ),
        )

    def cancel_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        """取消 pending_task 或待确认草稿。"""

        answer = (
            "已取消当前任务，"
            "没有写入或修改任何健康数据。"
        )

        messages = (
            *session_state.messages,
            AgentMessage(
                role="assistant",
                content=answer,
            ),
        )

        new_state = SessionState(
            session_id=(
                session_state.session_id
            ),
            user_id=(
                session_state.user_id
            ),
            timezone_name=(
                session_state
                .timezone_name
            ),
            state=AgentState.CANCELLED,
            messages=messages,
            turn_count=(
                session_state.turn_count
            ),
        )

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                AgentFinishReason
                .CANCELLED
            ),
            state=(
                AgentState.CANCELLED
            ),
            model_rounds=0,
        )

        return AgentTurnOutcome(
            result=result,
            session_state=new_state,
        )


class ConversationSession:
    """保存多轮 Agent 会话状态。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        session_id: str,
        user_id: str,
        timezone_name: str = (
            "Asia/Shanghai"
        ),
        session_state: (
            SessionState
            | None
        ) = None,
    ) -> None:
        self._runner = runner

        if session_state is not None:
            if (
                session_state.session_id
                != session_id
                or session_state.user_id
                != user_id
            ):
                raise ValueError(
                    "恢复的会话状态与当前会话不匹配"
                )

            self._state = session_state
        else:
            self._state = (
                runner.create_session_state(
                    session_id=session_id,
                    user_id=user_id,
                    timezone_name=(
                        timezone_name
                    ),
                )
            )

    @property
    def state(
        self,
    ) -> SessionState:
        """返回当前会话状态。"""

        return self._state

    def send(
        self,
        user_text: str,
    ) -> AgentRunResult:
        """提交一条用户消息。"""

        outcome = (
            self._runner.run_turn(
                session_state=(
                    self._state
                ),
                user_text=user_text,
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result

    def confirm(
        self,
    ) -> AgentRunResult:
        """执行用户明确确认的草稿。"""

        outcome = (
            self._runner
            .confirm_pending(
                self._state
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result

    def cancel(
        self,
    ) -> AgentRunResult:
        """取消当前短期任务或草稿。"""

        outcome = (
            self._runner
            .cancel_pending(
                self._state
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result
