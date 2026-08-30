"""个人健康管理助理 Gradio 应用。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import gradio as gr
from dotenv import load_dotenv

from src.agent.models import AgentRunResult
from src.agent.langgraph_runner import (
    LangGraphAgentRunner,
)
from src.agent.openai_model import (
    AgentConfigurationError,
    AgentProviderError,
    create_agent_model_from_environment,
)
from src.agent.runner import (
    AgentRunner,
    ConversationSession,
)
from src.agent.tool_router import HealthToolRouter
from src.agent.trace import (
    AgentTraceReadError,
    AgentTraceStore,
    DEFAULT_AGENT_TRACE_PATH,
    TracedConversationSession,
)
from src.health.models import (
    ExercisePayload,
    HealthEvent,
    MealPayload,
    WaterPayload,
    WeightPayload,
)
from src.nutrition.calculator import (
    NutritionCalculationError,
    calculate_nutrition,
    parse_grams,
)
from src.nutrition.repository import (
    FoodRepository,
    NutritionDataError,
)
from src.storage.jsonl_store import HealthEventStore
from src.tools.get_daily_health_summary import (
    get_daily_health_summary,
)
from src.tools.prepare_health_event import (
    prepare_health_event,
)
from src.tools.query_health_events import (
    query_health_events,
)
from src.tools.retrieve_nutrition_candidates import (
    retrieve_nutrition_candidates,
)
from src.tools.save_health_event import (
    save_health_event,
)
from src.ui.image_input import validate_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "health_events.jsonl"
)

LOCAL_USER_ID = "local-demo-user"

APP_TIMEZONE = os.getenv(
    "APP_TIMEZONE",
    "Asia/Shanghai",
).strip()

AGENT_ORCHESTRATOR = os.getenv(
    "AGENT_ORCHESTRATOR",
    "langgraph",
).strip().lower()


repository = FoodRepository()

event_store = HealthEventStore(
    EVENTS_PATH
)

tool_router = HealthToolRouter(
    event_store
)

agent_trace_store = AgentTraceStore(
    DEFAULT_AGENT_TRACE_PATH
)


try:
    if AGENT_ORCHESTRATOR not in {
        "legacy",
        "langgraph",
    }:
        raise AgentConfigurationError(
            "AGENT_ORCHESTRATOR 只能是 "
            "legacy 或 langgraph"
        )

    (
        agent_model,
        AGENT_PROVIDER_STATUS,
    ) = create_agent_model_from_environment()

    AGENT_PROVIDER_STATUS = (
        f"{AGENT_PROVIDER_STATUS.rstrip('。；;')}；"
        f"编排器：{AGENT_ORCHESTRATOR}"
    )
except AgentConfigurationError as exc:
    agent_model = None
    AGENT_PROVIDER_STATUS = (
        "Agent 配置错误："
        f"{exc}"
    )


_AGENT_SESSIONS: dict[
    str,
    TracedConversationSession,
] = {}

_AGENT_SESSIONS_LOCK = RLock()


def _error_text(
    error_code: str,
    message: str,
) -> str:
    """生成统一的 UI 错误文本。"""

    return (
        f"错误 [{error_code}]："
        f"{message}"
    )


def _timezone() -> ZoneInfo:
    """取得应用时区。"""

    try:
        return ZoneInfo(
            APP_TIMEZONE
        )
    except (
        ZoneInfoNotFoundError,
        ValueError,
    ) as exc:
        raise ValueError(
            "无法加载 APP_TIMEZONE："
            f"{APP_TIMEZONE}"
        ) from exc


def _today_string() -> str:
    """返回应用时区中的今天。"""

    return (
        datetime.now(
            _timezone()
        )
        .date()
        .isoformat()
    )


def _session_key(
    request: gr.Request,
) -> str:
    """获取 Gradio 浏览器会话标识。"""

    session_hash = getattr(
        request,
        "session_hash",
        None,
    )

    if (
        isinstance(session_hash, str)
        and session_hash.strip()
    ):
        return session_hash.strip()

    return "local-fallback-session"


def _get_agent_session(
    request: gr.Request,
) -> TracedConversationSession | None:
    """按浏览器会话获取带 Trace 的 Agent Session。"""

    if agent_model is None:
        return None

    key = _session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        existing = _AGENT_SESSIONS.get(
            key
        )

        if existing is not None:
            return existing

        runner_class = (
            LangGraphAgentRunner
            if AGENT_ORCHESTRATOR
            == "langgraph"
            else AgentRunner
        )

        runner = runner_class(
            model=agent_model,
            router=tool_router,
            max_model_rounds=4,
        )

        session = TracedConversationSession(
            runner=runner,
            session_id=key,
            user_id=LOCAL_USER_ID,
            timezone_name=APP_TIMEZONE,
            trace_store=agent_trace_store,
        )

        _AGENT_SESSIONS[key] = session

        return session


def cleanup_agent_session(
    request: gr.Request,
) -> None:
    """浏览器会话结束时清理短期状态。"""

    key = _session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        _AGENT_SESSIONS.pop(
            key,
            None,
        )


def _result_state_json(
    *,
    session: ConversationSession,
    result: AgentRunResult,
) -> dict[str, Any]:
    """生成不包含健康参数和确认令牌的状态证据。"""

    pending_task = (
        session.state.pending_task
    )

    pending_confirmation = (
        session.state.pending_confirmation
    )

    trace_warning = getattr(
        session,
        "last_trace_warning",
        None,
    )

    return {
        "session_id": (
            session.state.session_id
        ),
        "state": (
            session.state.state.value
        ),
        "turn_count": (
            session.state.turn_count
        ),
        "finish_reason": (
            result.finish_reason.value
        ),
        "model_rounds": (
            result.model_rounds
        ),
        "message_count": len(
            session.state.messages
        ),
        "pending_task": (
            {
                "tool_name": (
                    pending_task.tool_name
                ),
                "known_argument_names": sorted(
                    pending_task.arguments.keys()
                ),
                "missing_parameters": (
                    pending_task.missing_parameters
                ),
                "question": (
                    pending_task.question
                ),
            }
            if pending_task is not None
            else None
        ),
        "pending_confirmation": (
            {
                "action": (
                    pending_confirmation.action
                ),
                "tool_name": (
                    pending_confirmation.tool_name
                ),
                "confirmation_token": (
                    "***redacted***"
                ),
            }
            if pending_confirmation is not None
            else None
        ),
        "trace": {
            "enabled": True,
            "path": (
                "data/agent_traces.jsonl"
            ),
            "write_warning": (
                trace_warning
            ),
        },
    }


def _tool_steps_json(
    result: AgentRunResult,
) -> list[dict[str, Any]]:
    """将工具步骤转换为脱敏的开发者证据。"""

    safe_steps: list[
        dict[str, Any]
    ] = []

    for step in result.tool_steps:
        raw_ok = step.result.get(
            "ok"
        )

        ok = (
            raw_ok
            if isinstance(raw_ok, bool)
            else None
        )

        error_code: str | None = None

        raw_error = step.result.get(
            "error"
        )

        if isinstance(raw_error, dict):
            raw_code = raw_error.get(
                "code"
            )

            if isinstance(raw_code, str):
                error_code = raw_code

        safe_steps.append(
            {
                "call_id": (
                    step.call_id
                ),
                "tool_name": (
                    step.tool_name
                ),
                "argument_names": sorted(
                    step.arguments.keys()
                ),
                "ok": ok,
                "error_code": error_code,
            }
        )

    return safe_steps


def refresh_agent_traces(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """读取最近的脱敏 Agent Trace。"""

    try:
        traces = (
            agent_trace_store
            .read_recent(
                limit=limit
            )
        )
    except AgentTraceReadError as exc:
        return [
            {
                "ok": False,
                "error": {
                    "code": (
                        exc.error_code
                    ),
                    "message": str(
                        exc
                    ),
                },
            }
        ]
    except ValueError as exc:
        return [
            {
                "ok": False,
                "error": {
                    "code": (
                        "INVALID_TRACE_LIMIT"
                    ),
                    "message": str(
                        exc
                    ),
                },
            }
        ]

    return [
        trace.model_dump(
            mode="json"
        )
        for trace in traces
    ]


def _event_detail(
    event: HealthEvent,
) -> str:
    """将四类 payload 转为时间线摘要。"""

    payload = event.payload

    if isinstance(
        payload,
        MealPayload,
    ):
        return (
            f"{payload.food.name}，"
            f"{payload.portion.grams:g}g，"
            f"{payload.nutrition.calories_kcal:g}kcal"
        )

    if isinstance(
        payload,
        WaterPayload,
    ):
        return (
            f"{payload.beverage}，"
            f"{payload.amount_ml:g}ml"
            + (
                f"，{payload.note}"
                if payload.note
                else ""
            )
        )

    if isinstance(
        payload,
        WeightPayload,
    ):
        return (
            f"{payload.weight_kg:g}kg"
            + (
                f"，{payload.note}"
                if payload.note
                else ""
            )
        )

    if isinstance(
        payload,
        ExercisePayload,
    ):
        details = (
            f"{payload.activity_type}，"
            f"{payload.duration_minutes:g}分钟"
        )

        if payload.distance_km is not None:
            details += (
                f"，{payload.distance_km:g}km"
            )

        if payload.intensity is not None:
            details += (
                "，强度："
                f"{payload.intensity.value}"
            )

        if payload.note:
            details += (
                f"，{payload.note}"
            )

        return details

    return "未知事件"


def _timeline_rows(
    events: list[
        dict[str, Any]
    ],
) -> list[list[Any]]:
    """将工具返回事件转换为表格行。"""

    local_timezone = _timezone()

    rows: list[
        list[Any]
    ] = []

    for raw_event in events:
        event = (
            HealthEvent
            .model_validate(
                raw_event
            )
        )

        rows.append(
            [
                str(event.event_id),
                (
                    event.occurred_at
                    .astimezone(
                        local_timezone
                    )
                    .strftime(
                        "%Y-%m-%d "
                        "%H:%M:%S"
                    )
                ),
                event.event_type.value,
                _event_detail(
                    event
                ),
                event.input_source.value,
                (
                    event.updated_at
                    .astimezone(
                        local_timezone
                    )
                    .strftime(
                        "%Y-%m-%d "
                        "%H:%M:%S"
                    )
                ),
            ]
        )

    return rows


def _summary_markdown(
    summary: dict[str, Any],
) -> str:
    """将每日汇总转换为 Markdown。"""

    meal = summary["meal"]
    water = summary["water"]
    weight = summary["weight"]
    exercise = summary["exercise"]

    latest_weight = (
        f"{weight['latest_weight_kg']:g} kg"
        if weight["latest_weight_kg"]
        is not None
        else "无记录"
    )

    return (
        "### 每日确定性汇总\n\n"
        f"- 日期：`{summary['summary_date']}`\n"
        f"- 时区：`{summary['timezone']}`\n"
        f"- 健康事件总数：{summary['event_count']}\n"
        f"- 饮食：{meal['count']} 条，"
        f"{meal['calories_kcal']:.2f} kcal，"
        f"蛋白质 {meal['protein_g']:.2f} g，"
        f"脂肪 {meal['fat_g']:.2f} g，"
        f"碳水 {meal['carbs_g']:.2f} g\n"
        f"- 饮水：{water['count']} 条，"
        f"共 {water['total_ml']:.2f} ml\n"
        f"- 体重：{weight['count']} 条，"
        f"最近一次 {latest_weight}\n"
        f"- 运动：{exercise['count']} 条，"
        f"共 {exercise['total_duration_minutes']:.2f} 分钟，"
        f"{exercise['total_distance_km']:.2f} km\n\n"
        "**所有汇总均来自已保存事件；"
        "仅供学习，不构成医疗建议。**"
    )


def refresh_today() -> tuple[
    list[list[Any]],
    str,
]:
    """刷新今天的四类事件和汇总。"""

    try:
        today = _today_string()
    except ValueError as exc:
        return (
            [],
            _error_text(
                "TIMEZONE_INVALID",
                str(exc),
            ),
        )

    result = get_daily_health_summary(
        user_id=LOCAL_USER_ID,
        date=today,
        timezone_name=APP_TIMEZONE,
        store=event_store,
    )

    if not result["ok"]:
        error = result["error"]

        return (
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
        )

    data = result["data"]

    return (
        _timeline_rows(
            data["events"]
        ),
        _summary_markdown(
            data["summary"]
        ),
    )


def refresh_timeline(
    date_value: str,
    event_type_value: str,
) -> tuple[
    list[list[Any]],
    str,
]:
    """按日期和事件类型刷新时间线。"""

    selected_date = (
        date_value.strip()
        if isinstance(
            date_value,
            str,
        )
        else ""
    )

    if not selected_date:
        try:
            selected_date = (
                _today_string()
            )
        except ValueError as exc:
            return (
                [],
                _error_text(
                    "TIMEZONE_INVALID",
                    str(exc),
                ),
            )

    selected_event_type = (
        event_type_value.strip()
        if isinstance(
            event_type_value,
            str,
        )
        else ""
    )

    result = query_health_events(
        user_id=LOCAL_USER_ID,
        event_type=(
            selected_event_type
            or None
        ),
        date=selected_date,
        timezone_name=APP_TIMEZONE,
        newest_first=False,
        limit=500,
        store=event_store,
    )

    if not result["ok"]:
        error = result["error"]

        return (
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
        )

    data = result["data"]

    return (
        _timeline_rows(
            data["events"]
        ),
        (
            f"找到 {data['matched_count']} "
            "条已保存事件。"
        ),
    )


def search_candidates(
    image_path: str | None,
    food_query: str,
) -> tuple[
    gr.Dropdown,
    list[list[Any]],
    str,
    dict[str, Any],
]:
    """校验图片并检索人工填写的食物名称。"""

    image_result = validate_image(
        image_path
    )

    if not image_result.ok:
        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                (
                    image_result.error_code
                    or "IMAGE_INVALID"
                ),
                image_result.message,
            ),
            {},
        )

    tool_result = (
        retrieve_nutrition_candidates(
            query=food_query,
            top_k=5,
            repository=repository,
        )
    )

    if not tool_result["ok"]:
        error = tool_result["error"]

        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
            {},
        )

    data = tool_result["data"]
    trace = data["trace"]

    if data["status"] == "not_found":
        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                "NOT_FOUND",
                "没有找到可靠食物候选，"
                "不会猜测营养值。",
            ),
            trace,
        )

    choices: list[
        tuple[str, str]
    ] = []

    rows: list[
        list[Any]
    ] = []

    for candidate in data["candidates"]:
        choices.append(
            (
                f"{candidate['name']}｜"
                f"{candidate['category']}｜"
                f"stage {candidate['stage']} "
                f"{candidate['match_type']}",
                candidate["food_id"],
            )
        )

        rows.append(
            [
                candidate["food_id"],
                candidate["name"],
                candidate["category"],
                candidate["stage"],
                candidate["match_type"],
                candidate["matched_term"],
                candidate["score"],
                candidate["source"],
                candidate["source_version"],
                candidate["candidate_source"],
            ]
        )

    selected_value = (
        data["candidates"][0]["food_id"]
        if data["auto_select_allowed"]
        else None
    )

    warning_text = ""

    if data["trace_warning"] is not None:
        warning = data["trace_warning"]

        warning_text = (
            "\n\nTrace 警告 "
            f"[{warning['error_code']}]："
            f"{warning['message']}"
        )

    status = (
        f"已加载 {len(rows)} 个候选。"
        "归一化检索词："
        f"`{data['normalized_query']}`；"
        f"模式：`{data['selection_mode']}`；"
        f"数据集：`{data['dataset_id']}`；"
        f"耗时：{data['elapsed_ms']:.3f} ms。"
        + (
            "已按规则预选。"
            if selected_value
            else "候选有歧义，请手动选择。"
        )
        + warning_text
    )

    return (
        gr.Dropdown(
            choices=choices,
            value=selected_value,
        ),
        rows,
        status,
        trace,
    )


def calculate_meal_preview(
    image_path: str | None,
    food_query: str,
    selected_food_id: str | None,
    raw_grams: Any,
) -> tuple[
    str,
    dict[str, Any] | None,
    str,
    str,
]:
    """计算饮食营养并生成保存草稿。"""

    image_result = validate_image(
        image_path
    )

    if not image_result.ok:
        return (
            _error_text(
                (
                    image_result.error_code
                    or "IMAGE_INVALID"
                ),
                image_result.message,
            ),
            None,
            "",
            "",
        )

    try:
        search_result = repository.search(
            food_query,
            top_k=5,
        )

        if search_result.status == "not_found":
            return (
                _error_text(
                    "NOT_FOUND",
                    "没有可靠食物数据，"
                    "不能计算或保存。",
                ),
                None,
                "",
                "",
            )

        candidate_ids = {
            candidate.food_id
            for candidate
            in search_result.candidates
        }

        if (
            not selected_food_id
            or selected_food_id
            not in candidate_ids
        ):
            return (
                _error_text(
                    "CANDIDATE_REQUIRED",
                    "请选择当前检索结果"
                    "中的一个候选。",
                ),
                None,
                "",
                "",
            )

        food = repository.get_by_food_id(
            selected_food_id
        )

        grams = parse_grams(
            raw_grams
        )

        estimate = calculate_nutrition(
            food=food,
            raw_grams=grams,
            retrieval_query=food_query,
        )

    except NutritionDataError as exc:
        return (
            _error_text(
                exc.error_code,
                exc.message,
            ),
            None,
            "",
            "",
        )

    except NutritionCalculationError as exc:
        return (
            _error_text(
                exc.error_code,
                exc.message,
            ),
            None,
            "",
            "",
        )

    now = datetime.now(
        timezone.utc
    )

    draft_result = prepare_health_event(
        event_input={
            "event_type": "meal",
            "payload": {
                "food": {
                    "food_id": (
                        food.food_id
                    ),
                    "name": (
                        food.name
                    ),
                    "category": (
                        food.category
                    ),
                },
                "portion": {
                    "grams": (
                        float(grams)
                    ),
                    "unit": "g",
                },
                "nutrition": (
                    estimate.model_dump(
                        mode="json"
                    )
                ),
                "retrieval_query": (
                    food_query.strip()
                ),
                "candidate_source": (
                    "manual"
                ),
                "estimated": True,
            },
            "source_refs": [
                estimate.source_ref
            ],
            "input_source": "image",
            "occurred_at": (
                now.isoformat()
            ),
        },
        user_id=LOCAL_USER_ID,
        idempotency_key=str(
            uuid4()
        ),
        now=now,
    )

    if not draft_result["ok"]:
        error = draft_result["error"]

        return (
            _error_text(
                error["error_code"],
                error["message"],
            ),
            None,
            "",
            "",
        )

    preview_state = draft_result["data"]

    summary = (
        "### 待确认饮食记录\n\n"
        f"- 食物：{food.name}（{food.food_id}）\n"
        f"- 份量：{float(grams):g} g\n"
        f"- 热量估算：{estimate.calories_kcal:.2f} kcal\n"
        f"- 蛋白质估算：{estimate.protein_g:.2f} g\n"
        f"- 脂肪估算：{estimate.fat_g:.2f} g\n"
        f"- 碳水估算：{estimate.carbs_g:.2f} g\n"
        f"- 数据来源：{food.source}\n"
        f"- 来源版本：{food.source_version}\n"
        f"- 检索词：{estimate.retrieval_query}\n"
        f"- 份量假设：{estimate.portion_assumption}\n\n"
        "**尚未保存。以上均为估算值，"
        "仅供学习，不构成医疗建议。**"
    )

    evidence = (
        "### 可重算证据\n\n"
        f"- food_id：`{food.food_id}`\n"
        f"- 克重：`{float(grams):g} g`\n"
        f"- 热量：`{food.calories_per_100g:g} "
        "kcal/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.calories_kcal:.2f} kcal`\n"
        f"- 蛋白质：`{food.protein_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.protein_g:.2f} g`\n"
        f"- 脂肪：`{food.fat_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.fat_g:.2f} g`\n"
        f"- 碳水：`{food.carbs_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.carbs_g:.2f} g`"
    )

    return (
        summary,
        preview_state,
        (
            "计算完成，请核对后"
            "点击确认保存。"
        ),
        evidence,
    )


def confirm_meal_save(
    preview_state: (
        dict[str, Any]
        | None
    ),
) -> tuple[
    str,
    list[list[Any]],
    str,
]:
    """明确点击后保存饮食草稿。"""

    if not preview_state:
        rows, summary = refresh_today()

        return (
            _error_text(
                "PREVIEW_REQUIRED",
                "请先完成检索和"
                "营养计算。",
            ),
            rows,
            summary,
        )

    result = save_health_event(
        event_input=(
            preview_state["event"]
        ),
        confirmation_token=(
            preview_state[
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            preview_state[
                "idempotency_key"
            ]
        ),
        store=event_store,
    )

    rows, summary = refresh_today()

    if not result["ok"]:
        error = result["error"]

        return (
            _error_text(
                error["error_code"],
                error["message"],
            ),
            rows,
            summary,
        )

    if result["data"]["idempotent"]:
        status = (
            "该饮食草稿已保存过，"
            "没有新增重复记录。"
        )
    else:
        status = (
            "饮食记录成功："
            f"{result['data']['event']['event_id']}"
        )

    return (
        status,
        rows,
        summary,
    )


def cancel_meal_preview() -> tuple[
    None,
    str,
    str,
    str,
]:
    """取消饮食草稿。"""

    return (
        None,
        (
            "已取消，未写入"
            "任何饮食记录。"
        ),
        "尚未计算待确认记录。",
        (
            "### 可重算证据\n\n"
            "尚未选择数据行并计算。"
        ),
    )


def send_chat_message(
    user_text: str,
    history: (
        list[dict[str, Any]]
        | None
    ),
    request: gr.Request,
) -> tuple[
    list[dict[str, Any]],
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """向当前浏览器的 Agent Session 发送消息。"""

    chat_history = list(
        history or []
    )

    normalized_text = (
        user_text.strip()
        if isinstance(
            user_text,
            str,
        )
        else ""
    )

    if not normalized_text:
        return (
            chat_history,
            "",
            "请输入内容。",
            [],
            {},
        )

    chat_history.append(
        {
            "role": "user",
            "content": normalized_text,
        }
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型当前未启用。"
            "请在 `.env` 中配置 Provider，"
            "或继续使用饮食手动录入、"
            "时间线和每日汇总。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            AGENT_PROVIDER_STATUS,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
        )

    try:
        result = session.send(
            normalized_text
        )

    except AgentProviderError as exc:
        answer = (
            "模型调用失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            answer,
            [],
            {
                "state": (
                    "provider_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
        )

    except Exception as exc:
        answer = (
            "Agent 运行失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            answer,
            [],
            {
                "state": (
                    "agent_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
        )

    chat_history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    return (
        chat_history,
        "",
        (
            f"状态：{result.state.value}；"
            "结束原因："
            f"{result.finish_reason.value}；"
            f"模型轮次：{result.model_rounds}"
        ),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
    )


def confirm_agent_action(
    history: (
        list[dict[str, Any]]
        | None
    ),
    request: gr.Request,
) -> tuple[
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    dict[str, Any],
    list[list[Any]],
    str,
]:
    """用户点击按钮后确认 Agent 草稿。"""

    chat_history = list(
        history or []
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型未启用，"
            "当前没有可确认草稿。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        rows, summary = refresh_today()

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
            rows,
            summary,
        )

    try:
        result = session.confirm()

    except Exception as exc:
        answer = (
            "确认执行失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        rows, summary = refresh_today()

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "confirmation_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
            rows,
            summary,
        )

    chat_history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    rows, summary = refresh_today()

    return (
        chat_history,
        (
            f"状态：{result.state.value}；"
            "结束原因："
            f"{result.finish_reason.value}"
        ),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
        rows,
        summary,
    )


def cancel_agent_action(
    history: (
        list[dict[str, Any]]
        | None
    ),
    request: gr.Request,
) -> tuple[
    list[dict[str, Any]],
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """取消 Agent pending_task 或待确认草稿。"""

    chat_history = list(
        history or []
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型未启用，"
            "当前没有待取消任务。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
        )

    try:
        result = session.cancel()

    except Exception as exc:
        answer = (
            "取消操作失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "cancellation_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
        )

    chat_history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    return (
        chat_history,
        (
            f"状态：{result.state.value}；"
            "结束原因："
            f"{result.finish_reason.value}"
        ),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
    )


def reset_agent_conversation(
    request: gr.Request,
) -> tuple[
    list[dict[str, str]],
    str,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """清空当前浏览器的短期对话状态。"""

    cleanup_agent_session(
        request
    )

    return (
        [
            {
                "role": "assistant",
                "content": (
                    "会话已重置。"
                    "你可以记录饮水、体重、"
                    "运动，或者查询时间线。"
                ),
            }
        ],
        "会话已重置。",
        [],
        {
            "state": "idle",
            "trace": {
                "enabled": True,
                "path": (
                    "data/"
                    "agent_traces.jsonl"
                ),
            },
        },
    )


def build_demo() -> gr.Blocks:
    """构建完整的单页多 Tab 应用。"""

    initial_date = (
        _today_string()
        if APP_TIMEZONE
        else ""
    )

    with gr.Blocks(
        title=(
            "个人健康管理助理 Agent"
        ),
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# 个人健康管理助理 Agent\n\n"
            "支持饮食、饮水、体重和运动记录；"
            "所有写操作都必须先预览并确认。"
            "本项目仅用于学习，不构成医疗建议。"
        )

        with gr.Tab("今天"):
            refresh_today_button = gr.Button(
                "刷新今天",
                variant="secondary",
            )

            today_table = gr.Dataframe(
                headers=[
                    "event_id",
                    "occurred_at",
                    "event_type",
                    "detail",
                    "input_source",
                    "updated_at",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                value=[],
                interactive=False,
                label="今天的已保存事件",
            )

            today_summary = gr.Markdown(
                "正在读取今天的汇总……"
            )

        with gr.Tab("对话"):
            gr.Markdown(
                "## Agent 对话\n\n"
                f"{AGENT_PROVIDER_STATUS}\n\n"
                "饮食营养仍建议使用“饮食确认”"
                "Tab 完成候选选择与确定性计算。"
            )

            chatbot = gr.Chatbot(
                value=[
                    {
                        "role": "assistant",
                        "content": (
                            "你好。你可以说："
                            "“记录喝水500毫升”、"
                            "“我跑步了30分钟”、"
                            "“今天喝了多少水”。"
                        ),
                    }
                ],
                height=520,
                label="对话",
            )

            chat_input = gr.Textbox(
                label="输入消息",
                placeholder=(
                    "例如：记录跑步，"
                    "或查询今天的饮水记录"
                ),
                lines=2,
            )

            with gr.Row():
                send_button = gr.Button(
                    "发送",
                    variant="primary",
                )

                confirm_agent_button = gr.Button(
                    "确认当前操作",
                    variant="primary",
                )

                cancel_agent_button = gr.Button(
                    "取消当前操作",
                    variant="stop",
                )

                reset_agent_button = gr.Button(
                    "重置会话",
                    variant="secondary",
                )

            agent_status = gr.Markdown(
                AGENT_PROVIDER_STATUS
            )

        with gr.Tab("健康时间线"):
            with gr.Row():
                timeline_date = gr.Textbox(
                    label=(
                        "日期 "
                        "(YYYY-MM-DD)"
                    ),
                    value=initial_date,
                )

                timeline_event_type = (
                    gr.Dropdown(
                        label="事件类型",
                        choices=[
                            ("全部", ""),
                            ("饮食", "meal"),
                            ("饮水", "water"),
                            ("体重", "weight"),
                            ("运动", "exercise"),
                        ],
                        value="",
                    )
                )

                refresh_timeline_button = (
                    gr.Button(
                        "查询时间线",
                        variant="primary",
                    )
                )

            timeline_status = gr.Markdown()

            timeline_table = gr.Dataframe(
                headers=[
                    "event_id",
                    "occurred_at",
                    "event_type",
                    "detail",
                    "input_source",
                    "updated_at",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                value=[],
                interactive=False,
                label="健康时间线",
            )

            gr.Markdown(
                "修改或删除时，可以把表格中的 "
                "`event_id` 提供给 Agent。"
            )

        with gr.Tab("饮食确认"):
            gr.Markdown(
                "## 人工饮食主链\n\n"
                "图片只作为输入入口。"
                "食物名称和份量由用户填写，"
                "营养值来自 RAG 候选和确定性计算。"
            )

            meal_preview_state = gr.State(
                value=None
            )

            image_input = gr.File(
                label="上传一张食物图片",
                file_count="single",
                file_types=[
                    ".jpg",
                    ".jpeg",
                    ".png",
                ],
                type="filepath",
            )

            food_query = gr.Textbox(
                label="手动填写食物名称",
                placeholder="例如：西红柿",
            )

            grams_input = gr.Number(
                label="食物克重（g）",
                minimum=0.01,
                maximum=10000,
            )

            search_button = gr.Button(
                "查找候选",
                variant="secondary",
            )

            candidate_status = gr.Markdown()

            candidate_table = gr.Dataframe(
                headers=[
                    "food_id",
                    "name",
                    "category",
                    "stage",
                    "match_type",
                    "matched_term",
                    "score",
                    "source",
                    "source_version",
                    "candidate_source",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "str",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                ],
                value=[],
                interactive=False,
                label="RAG 候选",
            )

            selected_food = gr.Dropdown(
                label="选择候选食物",
                choices=[],
                value=None,
                interactive=True,
            )

            calculate_button = gr.Button(
                "计算营养估算",
                variant="primary",
            )

            calculation_status = gr.Markdown()

            meal_preview = gr.Markdown(
                "尚未计算待确认记录。"
            )

            recompute_evidence = gr.Markdown(
                "### 可重算证据\n\n"
                "尚未选择数据行并计算。"
            )

            with gr.Row():
                meal_save_button = (
                    gr.Button(
                        "确认保存饮食",
                        variant="primary",
                    )
                )

                meal_cancel_button = (
                    gr.Button(
                        "取消饮食草稿",
                        variant="stop",
                    )
                )

            meal_save_status = gr.Markdown()

        with gr.Tab("开发者证据"):
            gr.Markdown(
                "## Agent 执行证据\n\n"
                "工具参数值和确认令牌均已脱敏。"
            )

            latest_agent_steps = gr.JSON(
                value=[],
                label="脱敏 tool_steps",
            )

            latest_agent_state = gr.JSON(
                value={
                    "state": "idle",
                    "trace": {
                        "enabled": True,
                        "path": (
                            "data/"
                            "agent_traces.jsonl"
                        ),
                    },
                },
                label="Agent state",
            )

            gr.Markdown(
                "## 持久化 Agent Trace\n\n"
                "每次发送、确认和取消都会写入 "
                "`data/agent_traces.jsonl`。\n\n"
                "文件中只保存状态、工具名称、"
                "参数名称和错误码，不保存健康参数值、"
                "原始对话或确认令牌。"
            )

            with gr.Row():
                agent_trace_limit = gr.Number(
                    label="读取条数",
                    value=20,
                    minimum=1,
                    maximum=200,
                    precision=0,
                )

                refresh_agent_trace_button = (
                    gr.Button(
                        "刷新 Agent Trace",
                        variant="secondary",
                    )
                )

            recent_agent_traces = gr.JSON(
                value=refresh_agent_traces(),
                label="最近 Agent Trace",
            )

            gr.Markdown(
                "## 最近一次饮食检索 Trace"
            )

            latest_retrieval_trace = gr.JSON(
                value={},
                label="RetrievalTrace",
            )

        with gr.Tab("隐私与数据"):
            gr.Markdown(
                "## 隐私与数据\n\n"
                "- 健康事件保存在本机 "
                "`data/health_events.jsonl`。\n"
                "- Agent Trace 保存在本机 "
                "`data/agent_traces.jsonl`。\n"
                "- Agent Trace 不保存原始对话和健康参数值。\n"
                "- 原始图片不会复制到健康记录。\n"
                "- `.env` 和 API Key 不进入 Git。\n"
                "- 确认令牌不会展示在开发者证据中。\n"
                "- Agent Session 只保存在当前进程内存中。\n"
                "- 浏览器会话结束时清理短期 Session。\n"
                "- 每日汇总只读取 committed events。\n"
                "- 页面结果不构成医疗建议。\n"
                "- 当前 JSONL 方案只适合本地单用户、"
                "单进程学习项目。"
            )

        refresh_today_button.click(
            fn=refresh_today,
            outputs=[
                today_table,
                today_summary,
            ],
        )

        refresh_timeline_button.click(
            fn=refresh_timeline,
            inputs=[
                timeline_date,
                timeline_event_type,
            ],
            outputs=[
                timeline_table,
                timeline_status,
            ],
        )

        search_button.click(
            fn=search_candidates,
            inputs=[
                image_input,
                food_query,
            ],
            outputs=[
                selected_food,
                candidate_table,
                candidate_status,
                latest_retrieval_trace,
            ],
        )

        calculate_button.click(
            fn=calculate_meal_preview,
            inputs=[
                image_input,
                food_query,
                selected_food,
                grams_input,
            ],
            outputs=[
                meal_preview,
                meal_preview_state,
                calculation_status,
                recompute_evidence,
            ],
        )

        meal_save_button.click(
            fn=confirm_meal_save,
            inputs=[
                meal_preview_state
            ],
            outputs=[
                meal_save_status,
                today_table,
                today_summary,
            ],
        )

        meal_cancel_button.click(
            fn=cancel_meal_preview,
            outputs=[
                meal_preview_state,
                meal_save_status,
                meal_preview,
                recompute_evidence,
            ],
        )

        send_button.click(
            fn=send_chat_message,
            inputs=[
                chat_input,
                chatbot,
            ],
            outputs=[
                chatbot,
                chat_input,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
            ],
        )

        chat_input.submit(
            fn=send_chat_message,
            inputs=[
                chat_input,
                chatbot,
            ],
            outputs=[
                chatbot,
                chat_input,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
            ],
        )

        confirm_agent_button.click(
            fn=confirm_agent_action,
            inputs=[
                chatbot
            ],
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
                today_table,
                today_summary,
            ],
        )

        cancel_agent_button.click(
            fn=cancel_agent_action,
            inputs=[
                chatbot
            ],
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
            ],
        )

        reset_agent_button.click(
            fn=reset_agent_conversation,
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
            ],
        )

        refresh_agent_trace_button.click(
            fn=refresh_agent_traces,
            inputs=[
                agent_trace_limit
            ],
            outputs=[
                recent_agent_traces
            ],
        )

        demo.load(
            fn=refresh_today,
            outputs=[
                today_table,
                today_summary,
            ],
        )

        demo.load(
            fn=refresh_agent_traces,
            outputs=[
                recent_agent_traces
            ],
        )

        demo.unload(
            cleanup_agent_session
        )

    return demo


demo = build_demo()
