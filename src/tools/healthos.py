"""HealthOS P1 的 15 个受控工具及统一结果协议。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from src.healthos.models import (
    GoalStatus,
    GoalVersion,
    HealthGoal,
    Reminder,
    ReminderStatus,
    ReminderTransition,
    UserProfile,
)
from src.nutrition.calculator import NutritionCalculationError, calculate_nutrition as calculate_food
from src.nutrition.repository import FoodRepository, NutritionDataError
from src.storage.healthos_store import HealthOSStore, HealthOSStoreError
from src.storage.jsonl_store import HealthEventStore, JsonlReadError
from src.tools.confirmation import (
    issue_action_confirmation_token,
    verify_action_confirmation_token,
)
from src.tools.get_daily_health_summary import get_daily_health_summary
from src.tools.prepare_health_event import prepare_health_event as prepare_event
from src.tools.prepare_health_event_mutation import (
    prepare_delete_health_event,
    prepare_update_health_event,
)
from src.tools.query_health_events import query_health_events
from src.tools.retrieve_nutrition_candidates import (
    retrieve_nutrition_candidates as retrieve_food_candidates,
)


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def _failure(error_code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"error_code": error_code, "message": message},
    }


def _safe_call(function: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return function(*args, **kwargs)
    except (ValueError, ValidationError) as exc:
        return _failure("VALIDATION_ERROR", str(exc))
    except HealthOSStoreError as exc:
        return _failure("STORE_ERROR", str(exc))


def _parse_datetime(value: str, timezone_name: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("时间必须使用 ISO 8601 格式") from exc
    try:
        active_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"无法加载时区：{timezone_name}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=active_timezone)
    return parsed


def _draft(
    *,
    action: str,
    user_id: str,
    payload: dict[str, Any],
    idempotency_key: str,
    preview: dict[str, Any],
) -> dict[str, Any]:
    token = issue_action_confirmation_token(
        action=action,
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _ok(
        {
            "action": action,
            "user_id": user_id,
            "payload": payload,
            "preview": preview,
            "confirmation_token": token,
            "idempotency_key": idempotency_key,
        }
    )


def get_user_profile(
    *, user_id: str, timezone_name: str, store: HealthOSStore
) -> dict[str, Any]:
    """读取最小必要档案，不返回聊天或健康事实。"""

    return _safe_call(
        lambda: _ok(
            {
                "profile": store.get_profile(user_id, timezone_name).model_dump(
                    mode="json"
                ),
                "field_scope": [
                    "timezone_name",
                    "unit_system",
                    "coach_style",
                    "dietary_preferences",
                    "exclusions",
                    "reminders_enabled",
                    "quiet_hours_start",
                    "quiet_hours_end",
                    "version",
                    "updated_at",
                ],
            }
        )
    )


def prepare_profile_update(
    *,
    user_id: str,
    timezone_name: str,
    patch: dict[str, Any],
    idempotency_key: str,
    store: HealthOSStore,
) -> dict[str, Any]:
    """校验档案变更并生成无副作用草稿。"""

    allowed = {
        "timezone_name",
        "coach_style",
        "dietary_preferences",
        "exclusions",
        "reminders_enabled",
        "quiet_hours_start",
        "quiet_hours_end",
    }
    forbidden = set(patch) - allowed
    if forbidden:
        return _failure(
            "PROFILE_FIELD_FORBIDDEN",
            "档案不允许写入这些字段：" + "、".join(sorted(forbidden)),
        )
    try:
        current = store.get_profile(user_id, timezone_name)
        proposed = UserProfile.model_validate(
            {
                **current.model_dump(mode="python"),
                **patch,
                "version": current.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    except ValidationError:
        return _failure(
            "PROFILE_VALIDATION_ERROR",
            "这项个人设置无法识别。教练风格可以选择：温和陪伴、理性复盘、简洁提醒或目标督促；免打扰时间请使用 HH:MM。",
        )
    except (ValueError, HealthOSStoreError) as exc:
        return _failure("PROFILE_UPDATE_ERROR", str(exc))
    payload = {
        "before_version": current.version,
        "profile": proposed.model_dump(mode="json"),
    }
    return _draft(
        action="profile_update",
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
        preview={"before": current.model_dump(mode="json"), "after": payload["profile"]},
    )


def get_health_goals(*, user_id: str, store: HealthOSStore) -> dict[str, Any]:
    """返回当前目标及不可丢失的版本历史。"""

    try:
        goals = [goal for goal in store.read().goals if goal.user_id == user_id]
    except HealthOSStoreError as exc:
        return _failure("STORE_ERROR", str(exc))
    return _ok(
        {
            "goals": [goal.model_dump(mode="json") for goal in goals],
            "count": len(goals),
        }
    )


def prepare_goal_change(
    *,
    user_id: str,
    operation: str,
    idempotency_key: str,
    store: HealthOSStore,
    goal_id: str | None = None,
    title: str | None = None,
    goal_type: str | None = None,
    target_value: float | None = None,
    unit: str | None = None,
    period: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """创建、调整、暂停或恢复目标，且永不覆盖旧版本。"""

    normalized_operation = operation.strip().lower()
    if normalized_operation not in {"create", "update", "pause", "resume"}:
        return _failure("GOAL_OPERATION_INVALID", "目标操作必须是 create、update、pause 或 resume")
    now = datetime.now(timezone.utc)
    try:
        if normalized_operation == "create":
            if None in {title, goal_type, target_value, unit, period}:
                return _failure("GOAL_FIELDS_REQUIRED", "创建目标还需要标题、类型、目标值、单位和周期")
            new_goal_id = uuid4()
            version = GoalVersion(
                version=1,
                title=title,
                goal_type=goal_type,
                target_value=target_value,
                unit=unit,
                period=period,
                status=GoalStatus.ACTIVE,
                reason=reason or "用户创建目标",
                created_at=now,
            )
            before = None
        else:
            if not goal_id:
                return _failure("GOAL_ID_REQUIRED", "请先选择要调整的健康目标")
            state = store.read()
            index = store.find_goal_index(state, user_id, UUID(goal_id))
            goal = state.goals[index]
            current = goal.current
            new_goal_id = goal.goal_id
            target_status = current.status
            if normalized_operation == "pause":
                target_status = GoalStatus.PAUSED
            elif normalized_operation == "resume":
                target_status = GoalStatus.ACTIVE
            version = GoalVersion(
                version=current.version + 1,
                title=title or current.title,
                goal_type=goal_type or current.goal_type,
                target_value=target_value if target_value is not None else current.target_value,
                unit=unit or current.unit,
                period=period or current.period,
                status=target_status,
                reason=reason or f"用户{normalized_operation}目标",
                created_at=now,
            )
            before = current.model_dump(mode="json")
    except (ValueError, ValidationError, HealthOSStoreError) as exc:
        return _failure("VALIDATION_ERROR", str(exc))
    payload = {
        "operation": normalized_operation,
        "goal_id": str(new_goal_id),
        "version": version.model_dump(mode="json"),
    }
    return _draft(
        action="goal_change",
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
        preview={"before": before, "after": payload["version"]},
    )


def get_health_events(*, store: HealthEventStore, **kwargs: Any) -> dict[str, Any]:
    """只读取已经提交的 HealthEvent。"""

    return query_health_events(store=store, **kwargs)


def prepare_health_event(**kwargs: Any) -> dict[str, Any]:
    """生成健康事实草稿，不保存半成品。"""

    return prepare_event(**kwargs)


def prepare_event_change(
    *, operation: str, event_id: str, user_id: str, idempotency_key: str,
    store: HealthEventStore, patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用一个受控入口生成修改或删除草稿。"""

    if operation == "update":
        if not patch:
            return _failure("PATCH_REQUIRED", "请说明要修改哪些内容")
        return prepare_update_health_event(
            event_id=event_id,
            user_id=user_id,
            patch=patch,
            idempotency_key=idempotency_key,
            store=store,
        )
    if operation == "delete":
        return prepare_delete_health_event(
            event_id=event_id,
            user_id=user_id,
            idempotency_key=idempotency_key,
            store=store,
        )
    return _failure("EVENT_OPERATION_INVALID", "记录变更只支持 update 或 delete")


def retrieve_nutrition_candidates(**kwargs: Any) -> dict[str, Any]:
    return retrieve_food_candidates(**kwargs)


def calculate_nutrition(
    *, food_code: str, grams: float, retrieval_query: str, repository: FoodRepository
) -> dict[str, Any]:
    """只使用选中食物数据行和克重进行确定性计算。"""

    try:
        food = repository.get_by_food_id(food_code)
        if food is None:
            return _failure("FOOD_NOT_FOUND", "找不到所选食物数据行，请重新检索")
        estimate = calculate_food(food, grams, retrieval_query)
    except (NutritionCalculationError, NutritionDataError, ValueError) as exc:
        error_code = getattr(exc, "error_code", "NUTRITION_CALCULATION_ERROR")
        message = getattr(exc, "message", str(exc))
        return _failure(str(error_code), str(message))
    return _ok(
        {
            "food": food.model_dump(mode="json"),
            "portion_grams": float(grams),
            "nutrition": estimate.model_dump(mode="json"),
            "formula": "每 100g 数据 × 克重 ÷ 100",
        }
    )


_KNOWLEDGE = (
    {
        "document_id": "who-physical-activity-2024",
        "title": "WHO Physical activity fact sheet",
        "topics": ("运动", "活动", "久坐", "锻炼", "exercise", "physical activity"),
        "content": "成年人可把每周至少 150 分钟中等强度活动作为一般参考，并根据自身情况循序渐进。",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        "updated_at": "2024-06-26",
    },
    {
        "document_id": "who-healthy-diet-2026",
        "title": "WHO Healthy diet fact sheet",
        "topics": ("饮食", "蔬菜", "水果", "全谷物", "盐", "糖", "healthy diet"),
        "content": "一般健康饮食强调充足、平衡、适度和多样，优先选择多样化且少加工的食物。",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "updated_at": "2026-01-26",
    },
)

_URGENT_TERMS = ("胸痛", "呼吸困难", "晕厥", "昏倒", "自杀", "自伤", "急救", "emergency")
_MEDICAL_TERMS = ("诊断", "药", "用药", "剂量", "处方", "怀孕", "孕妇", "未成年人", "进食障碍")
_INJECTION_TERMS = ("忽略之前", "忽略系统", "system prompt", "developer message", "越过规则")


def retrieve_health_knowledge(*, question: str, top_k: int = 3) -> dict[str, Any]:
    """检索可信一般健康知识，并在高风险场景拒答。"""

    normalized = question.strip().casefold()
    if not normalized:
        return _failure("QUESTION_REQUIRED", "健康问题不能为空")
    if len(normalized) > 500:
        return _failure("QUESTION_TOO_LONG", "健康问题不得超过 500 个字符")
    if any(term in normalized for term in _INJECTION_TERMS):
        return _failure("PROMPT_INJECTION_DETECTED", "问题包含试图改变安全规则的内容，未执行检索")
    if any(term in normalized for term in _URGENT_TERMS):
        return _failure("URGENT_HELP_REQUIRED", "这可能涉及紧急健康风险，请立即联系当地急救服务或合格专业人员")
    if any(term in normalized for term in _MEDICAL_TERMS):
        return _failure("MEDICAL_BOUNDARY", "我不能提供诊断或用药建议，请咨询合格医疗专业人员")
    matches = []
    for document in _KNOWLEDGE:
        score = sum(1 for term in document["topics"] if term.casefold() in normalized)
        if score:
            matches.append({**document, "score": score})
    matches.sort(key=lambda item: (-item["score"], item["document_id"]))
    if not matches:
        return _failure("KNOWLEDGE_NOT_FOUND", "当前可信知识库没有足够证据回答这个问题")
    return _ok(
        {
            "answer_scope": "一般健康生活信息，不是医疗建议",
            "citations": matches[:top_k],
            "count": min(len(matches), top_k),
        }
    )


def _active_goal_versions(user_id: str, store: HealthOSStore | None) -> list[dict[str, Any]]:
    if store is None:
        return []
    return [
        goal.current.model_dump(mode="json")
        for goal in store.read().goals
        if goal.user_id == user_id and goal.current.status == GoalStatus.ACTIVE
    ]


def get_daily_summary(
    *, user_id: str, date: str, timezone_name: str, store: HealthEventStore,
    healthos_store: HealthOSStore | None = None,
) -> dict[str, Any]:
    """汇总当天事实、数据完整度和可确定计算的目标差距。"""

    result = get_daily_health_summary(
        user_id=user_id,
        date=date,
        timezone_name=timezone_name,
        store=store,
    )
    if not result.get("ok"):
        return result
    try:
        goals = _active_goal_versions(user_id, healthos_store)
    except HealthOSStoreError as exc:
        return _failure("STORE_ERROR", str(exc))
    summary = result["data"]["summary"]
    values = {
        "water": float(summary["water"]["total_ml"]),
        "exercise": float(summary["exercise"]["total_duration_minutes"]),
        "nutrition": float(summary["meal"]["calories_kcal"]),
        "weight": summary["weight"]["latest_weight_kg"],
    }
    gaps = []
    for goal in goals:
        if goal["period"] != "daily" or goal["goal_type"] == "custom":
            continue
        current_value = values.get(goal["goal_type"])
        gap = None if current_value is None else round(float(goal["target_value"]) - float(current_value), 2)
        gaps.append(
            {
                "title": goal["title"],
                "target_value": goal["target_value"],
                "current_value": current_value,
                "remaining": gap,
                "unit": goal["unit"],
            }
        )
    present_categories = sum(
        1
        for section in ("meal", "water", "weight", "exercise")
        if int(summary[section]["count"]) > 0
    )
    result["data"]["goal_gaps"] = gaps
    result["data"]["data_completeness"] = round(present_categories / 4, 2)
    return result


def get_period_summary(
    *, user_id: str, days: int, end_date: str | None, timezone_name: str,
    store: HealthEventStore, healthos_store: HealthOSStore | None = None,
) -> dict[str, Any]:
    """确定性汇总 7/14/30 天事实，不推断变化原因。"""

    if days not in {7, 14, 30}:
        return _failure("PERIOD_INVALID", "趋势周期只支持 7、14 或 30 天")
    try:
        active_timezone = ZoneInfo(timezone_name)
        final_date = date.fromisoformat(end_date) if end_date else datetime.now(active_timezone).date()
        start_date = final_date - timedelta(days=days - 1)
        occurred_from = datetime.combine(start_date, time.min, tzinfo=active_timezone)
        occurred_to = datetime.combine(final_date + timedelta(days=1), time.min, tzinfo=active_timezone)
        events = store.query(
            user_id=user_id,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
        )
    except (ValueError, ZoneInfoNotFoundError, JsonlReadError) as exc:
        return _failure("PERIOD_QUERY_ERROR", str(exc))
    event_days = {event.occurred_at.astimezone(active_timezone).date() for event in events}
    meals = [event for event in events if event.event_type.value == "meal"]
    waters = [event for event in events if event.event_type.value == "water"]
    exercises = [event for event in events if event.event_type.value == "exercise"]
    weights = [event for event in events if event.event_type.value == "weight"]
    calories = sum(float(event.payload.nutrition.calories_kcal) for event in meals)
    water_ml = sum(float(event.payload.amount_ml) for event in waters)
    exercise_minutes = sum(float(event.payload.duration_minutes) for event in exercises)
    weight_change = None
    if len(weights) >= 2:
        weight_change = round(float(weights[-1].payload.weight_kg - weights[0].payload.weight_kg), 2)
    try:
        goals = _active_goal_versions(user_id, healthos_store)
    except HealthOSStoreError as exc:
        return _failure("STORE_ERROR", str(exc))
    goal_progress = []
    period_values = {
        "exercise": exercise_minutes,
        "water": water_ml,
        "nutrition": calories,
        "weight": float(weights[-1].payload.weight_kg) if weights else None,
    }
    for goal in goals:
        if goal["goal_type"] == "custom":
            continue
        current_value = period_values.get(goal["goal_type"])
        target_value = float(goal["target_value"])
        if goal["goal_type"] != "weight":
            factor = {
                "daily": days,
                "weekly": days / 7,
                "monthly": days / 30,
                "8_weeks": days / 56,
            }[goal["period"]]
            target_value = round(target_value * factor, 2)
        goal_progress.append(
            {
                "title": goal["title"],
                "target_value": target_value,
                "current_value": current_value,
                "remaining": None if current_value is None else round(target_value - float(current_value), 2),
                "unit": goal["unit"],
                "period": goal["period"],
            }
        )
    return _ok(
        {
            "period": {"from": start_date.isoformat(), "to": final_date.isoformat(), "days": days},
            "event_count": len(events),
            "days_with_data": len(event_days),
            "data_completeness": round(len(event_days) / days, 3),
            "meal": {"count": len(meals), "calories_kcal": round(calories, 2)},
            "water": {"count": len(waters), "total_ml": round(water_ml, 2)},
            "exercise": {"count": len(exercises), "total_minutes": round(exercise_minutes, 2)},
            "weight": {"count": len(weights), "change_kg": weight_change},
            "goal_progress": goal_progress,
            "interpretation_boundary": "仅陈述已记录事实；数据不足时不推断原因",
        }
    )


def create_reminder_draft(
    *, user_id: str, content: str, scheduled_for: str, timezone_name: str,
    idempotency_key: str, store: HealthOSStore,
) -> dict[str, Any]:
    """生成本地提醒草稿，不立即安排。"""

    try:
        profile = store.get_profile(user_id, timezone_name)
        if not profile.reminders_enabled:
            return _failure("REMINDERS_DISABLED", "提醒已关闭，请先在档案中开启")
        scheduled = _parse_datetime(scheduled_for, profile.timezone_name)
        if scheduled <= datetime.now(timezone.utc).astimezone(scheduled.tzinfo):
            return _failure("REMINDER_TIME_PAST", "提醒时间必须晚于当前时间")
        reminder_id = uuid4()
    except (ValueError, HealthOSStoreError) as exc:
        return _failure("VALIDATION_ERROR", str(exc))
    payload = {
        "reminder_id": str(reminder_id),
        "content": content.strip(),
        "scheduled_for": scheduled.isoformat(),
        "timezone_name": profile.timezone_name,
    }
    if not payload["content"] or len(payload["content"]) > 300:
        return _failure("REMINDER_CONTENT_INVALID", "提醒内容必须为 1—300 个字符")
    return _draft(
        action="reminder_create",
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
        preview=payload,
    )


def _execute_healthos_draft(
    *, action: str, user_id: str, payload: dict[str, Any], confirmation_token: str,
    idempotency_key: str, store: HealthOSStore,
) -> dict[str, Any]:
    valid, message = verify_action_confirmation_token(
        token=confirmation_token,
        action=action,
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    if not valid:
        return _failure("CONFIRMATION_INVALID", message)
    try:
        def mutation(state: Any) -> dict[str, Any]:
            cached = state.idempotency_results.get(idempotency_key)
            if cached is not None:
                return {**cached, "idempotent_replay": True}
            now = datetime.now(timezone.utc)
            if action == "profile_update":
                proposed = UserProfile.model_validate(payload["profile"])
                current = state.profiles.get(user_id)
                current_version = current.version if current is not None else 1
                if current_version != int(payload["before_version"]):
                    raise ValueError("档案已被更新，请重新生成草稿")
                state.profiles[user_id] = proposed
                result = {"action": action, "profile": proposed.model_dump(mode="json")}
            elif action == "goal_change":
                goal_id = UUID(payload["goal_id"])
                version = GoalVersion.model_validate(payload["version"])
                if payload["operation"] == "create":
                    state.goals.append(HealthGoal(goal_id=goal_id, user_id=user_id, versions=[version]))
                else:
                    index = store.find_goal_index(state, user_id, goal_id)
                    goal = state.goals[index]
                    if version.version != goal.current.version + 1:
                        raise ValueError("目标已被更新，请重新生成草稿")
                    state.goals[index] = goal.model_copy(update={"versions": [*goal.versions, version]})
                result = {"action": action, "goal_id": str(goal_id), "version": version.model_dump(mode="json")}
            elif action == "reminder_create":
                scheduled = _parse_datetime(payload["scheduled_for"], payload["timezone_name"])
                transition = ReminderTransition(status=ReminderStatus.SCHEDULED, occurred_at=now, reason="用户确认创建")
                reminder = Reminder(
                    reminder_id=UUID(payload["reminder_id"]),
                    user_id=user_id,
                    content=payload["content"],
                    scheduled_for=scheduled,
                    timezone_name=payload["timezone_name"],
                    status=ReminderStatus.SCHEDULED,
                    created_at=now,
                    updated_at=now,
                    transitions=[transition],
                )
                state.reminders.append(reminder)
                result = {"action": action, "reminder": reminder.model_dump(mode="json")}
            elif action == "reminder_change":
                reminder_id = UUID(payload["reminder_id"])
                index = store.find_reminder_index(state, user_id, reminder_id)
                current = state.reminders[index]
                operation = payload["operation"]
                status_map = {
                    "cancel": ReminderStatus.CANCELLED,
                    "pause": ReminderStatus.PAUSED,
                    "resume": ReminderStatus.SCHEDULED,
                    "snooze": ReminderStatus.SNOOZED,
                }
                next_status = status_map[operation]
                scheduled_for = current.scheduled_for
                if operation == "snooze":
                    scheduled_for = _parse_datetime(payload["scheduled_for"], current.timezone_name)
                transition = ReminderTransition(status=next_status, occurred_at=now, reason=payload.get("reason") or f"用户{operation}")
                updated = current.model_copy(
                    update={
                        "status": next_status,
                        "scheduled_for": scheduled_for,
                        "updated_at": now,
                        "transitions": [*current.transitions, transition],
                    }
                )
                state.reminders[index] = updated
                result = {"action": action, "reminder": updated.model_dump(mode="json")}
            else:
                raise ValueError("不支持的确认操作")
            state.idempotency_results[idempotency_key] = result
            return {**result, "idempotent_replay": False}

        result = store.update(mutation)
    except (ValueError, ValidationError, HealthOSStoreError, KeyError) as exc:
        return _failure("EXECUTION_ERROR", str(exc))
    return _ok(result)


def execute_reminder(
    *, user_id: str, draft: dict[str, Any], confirmation_token: str,
    idempotency_key: str, store: HealthOSStore,
) -> dict[str, Any]:
    """带确认令牌与幂等键执行提醒草稿。"""

    action = str(draft.get("action", "reminder_create"))
    payload = draft.get("payload", draft)
    if not isinstance(payload, dict):
        return _failure("REMINDER_DRAFT_INVALID", "提醒草稿格式无效")
    return _execute_healthos_draft(
        action=action,
        user_id=user_id,
        payload=payload,
        confirmation_token=confirmation_token,
        idempotency_key=idempotency_key,
        store=store,
    )


def list_or_cancel_reminders(
    *, user_id: str, action: str, store: HealthOSStore, idempotency_key: str,
    reminder_id: str | None = None, scheduled_for: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """查看提醒；取消、延后、暂停或恢复时只生成草稿。"""

    normalized_action = action.strip().lower()
    if normalized_action == "list":
        try:
            reminders = [item for item in store.read().reminders if item.user_id == user_id]
        except HealthOSStoreError as exc:
            return _failure("STORE_ERROR", str(exc))
        reminders.sort(key=lambda item: (item.scheduled_for, str(item.reminder_id)))
        return _ok({"reminders": [item.model_dump(mode="json") for item in reminders], "count": len(reminders)})
    if normalized_action not in {"cancel", "snooze", "pause", "resume"}:
        return _failure("REMINDER_OPERATION_INVALID", "提醒操作必须是 list、cancel、snooze、pause 或 resume")
    if not reminder_id:
        return _failure("REMINDER_ID_REQUIRED", "请先选择要操作的提醒")
    if normalized_action == "snooze" and not scheduled_for:
        return _failure("REMINDER_TIME_REQUIRED", "延后提醒需要新的时间")
    try:
        state = store.read()
        index = store.find_reminder_index(state, user_id, UUID(reminder_id))
        current = state.reminders[index]
        if current.status in {ReminderStatus.CANCELLED, ReminderStatus.COMPLETED}:
            return _failure("REMINDER_FINAL_STATE", "这条提醒已经结束，不能再次修改")
        payload = {
            "operation": normalized_action,
            "reminder_id": reminder_id,
            "scheduled_for": scheduled_for,
            "reason": reason or f"用户{normalized_action}",
        }
    except (ValueError, HealthOSStoreError) as exc:
        return _failure("VALIDATION_ERROR", str(exc))
    return _draft(
        action="reminder_change",
        user_id=user_id,
        payload=payload,
        idempotency_key=idempotency_key,
        preview={"before": current.model_dump(mode="json"), "operation": normalized_action, "scheduled_for": scheduled_for},
    )


def execute_healthos_confirmation(*, draft_data: dict[str, Any], store: HealthOSStore) -> dict[str, Any]:
    """供统一确认中间件执行档案、目标和提醒写操作。"""

    return _execute_healthos_draft(
        action=draft_data["action"],
        user_id=draft_data["user_id"],
        payload=draft_data["payload"],
        confirmation_token=draft_data["confirmation_token"],
        idempotency_key=draft_data["idempotency_key"],
        store=store,
    )
