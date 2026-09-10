"""到期提醒调度器：确认后的任务才可能进入自动发送。"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from threading import Event, Thread
from typing import Any, Protocol
from uuid import UUID
from zoneinfo import ZoneInfo

from src.automation.feishu import FeishuDeliveryUnknown
from src.healthos.models import Reminder, ReminderStatus, ReminderTransition, UserProfile
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore


class ReminderNotifier(Protocol):
    channel: str
    destination_label: str
    destination_key: str

    def send(self, text: str) -> dict[str, Any]: ...


def _in_quiet_hours(profile: UserProfile, now: datetime) -> bool:
    if not profile.quiet_hours_start or not profile.quiet_hours_end:
        return False
    local_time = now.astimezone(ZoneInfo(profile.timezone_name)).time().replace(tzinfo=None)
    start = time.fromisoformat(profile.quiet_hours_start)
    end = time.fromisoformat(profile.quiet_hours_end)
    if start < end:
        return start <= local_time < end
    return local_time >= start or local_time < end


def _claim_due_reminder(
    store: HealthOSStore,
    *,
    reminder_id: UUID,
    user_id: str,
    now: datetime,
) -> Reminder | None:
    def mutation(state: Any) -> Reminder | None:
        try:
            index = store.find_reminder_index(state, user_id, reminder_id)
        except ValueError:
            return None
        current = state.reminders[index]
        if current.status not in {ReminderStatus.SCHEDULED, ReminderStatus.SNOOZED}:
            return None
        if current.scheduled_for.astimezone(timezone.utc) > now.astimezone(timezone.utc):
            return None
        transition = ReminderTransition(
            status=ReminderStatus.FIRED,
            occurred_at=now,
            reason="自动化调度器已领取发送任务",
        )
        claimed = current.model_copy(
            update={
                "status": ReminderStatus.FIRED,
                "delivery_attempts": current.delivery_attempts + 1,
                "updated_at": now,
                "last_delivery_error": None,
                "transitions": [*current.transitions, transition],
            }
        )
        state.reminders[index] = claimed
        return claimed

    return store.update(mutation)


def _finish_delivery(
    store: HealthOSStore,
    *,
    reminder: Reminder,
    now: datetime,
    error: str | None,
    outcome_unknown: bool = False,
) -> None:
    def mutation(state: Any) -> None:
        index = store.find_reminder_index(state, reminder.user_id, reminder.reminder_id)
        current = state.reminders[index]
        if current.status != ReminderStatus.FIRED:
            return
        if outcome_unknown:
            status = ReminderStatus.UNKNOWN
        elif error:
            status = ReminderStatus.FAILED
        elif current.reminder_type == "check_in" and current.recurrence != "once":
            status = ReminderStatus.SCHEDULED
        else:
            status = ReminderStatus.COMPLETED
        scheduled_for = current.scheduled_for
        if status == ReminderStatus.SCHEDULED:
            scheduled_for = _next_occurrence(current, now)
        transition = ReminderTransition(
            status=status,
            occurred_at=now,
            reason=(
                error
                or (
                    f"已发送到 {reminder.destination_label}；"
                    f"下次 {scheduled_for.isoformat()}"
                    if status == ReminderStatus.SCHEDULED
                    else f"已发送到 {reminder.destination_label}"
                )
            )[:500],
        )
        state.reminders[index] = current.model_copy(
            update={
                "status": status,
                "scheduled_for": scheduled_for,
                "updated_at": now,
                "last_delivery_at": now,
                "last_delivery_error": error[:500] if error else None,
                "transitions": [*current.transitions, transition],
            }
        )

    store.update(mutation)


def _next_occurrence(reminder: Reminder, now: datetime) -> datetime:
    """保持用户本地钟点，计算下一次每天或工作日 check-in。"""

    active_timezone = ZoneInfo(reminder.timezone_name)
    candidate = reminder.scheduled_for.astimezone(active_timezone)
    active_now = now.astimezone(active_timezone)
    while candidate <= active_now:
        candidate += timedelta(days=1)
    if reminder.recurrence == "weekdays":
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
    return candidate


def build_check_in_message(
    reminder: Reminder,
    *,
    event_store: HealthEventStore,
    now: datetime,
) -> str:
    """只根据当日已确认事件生成可回复的 check-in，不推断用户行为。"""

    active_timezone = ZoneInfo(reminder.timezone_name)
    local_now = now.astimezone(active_timezone)
    day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    events = event_store.query(
        user_id=reminder.user_id,
        occurred_from=day_start,
        occurred_to=day_end,
    )
    by_type: dict[str, list[Any]] = {
        "meal": [],
        "water": [],
        "exercise": [],
        "weight": [],
    }
    for event in events:
        by_type[event.event_type.value].append(event)

    focus = reminder.check_in_focus or ["meal", "water", "exercise"]
    facts: list[str] = []
    missing: list[str] = []
    labels = {"meal": "饮食", "water": "饮水", "exercise": "运动", "weight": "体重"}
    for event_type in focus:
        matches = by_type[event_type]
        if not matches:
            missing.append(labels[event_type])
            continue
        if event_type == "water":
            amount = sum(float(item.payload.amount_ml) for item in matches)
            facts.append(f"饮水 {amount:g} ml")
        elif event_type == "exercise":
            minutes = sum(float(item.payload.duration_minutes) for item in matches)
            facts.append(f"运动 {minutes:g} 分钟")
        elif event_type == "meal":
            facts.append(f"饮食 {len(matches)} 条")
        else:
            latest = max(matches, key=lambda item: item.occurred_at)
            facts.append(f"最近体重 {float(latest.payload.weight_kg):g} kg")

    fact_text = "、".join(facts) if facts else "暂时没有看到相关记录"
    if missing:
        question = f"暂未看到{'、'.join(missing)}记录。是今天没有发生，还是需要补记？"
    else:
        question = "这些记录已经比较完整。还有需要补记或修改的吗？"
    return (
        "小满主动 check-in\n"
        f"今天已确认：{fact_text}。\n"
        f"{question}\n"
        "回复这条消息不会自动写入；请回到 HealthOS 对话确认保存。"
    )


def dispatch_due_reminders(
    *,
    store: HealthOSStore,
    notifier: ReminderNotifier,
    event_store: HealthEventStore | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """发送一批到期提醒；未确认、本地渠道和免打扰时段均跳过。"""

    active_now = now or datetime.now(timezone.utc)
    if active_now.tzinfo is None or active_now.utcoffset() is None:
        raise ValueError("now 必须包含时区")
    snapshot = store.read()
    outcomes: list[dict[str, Any]] = []
    for candidate in sorted(snapshot.reminders, key=lambda item: item.scheduled_for):
        if candidate.delivery_channel != notifier.channel:
            continue
        if candidate.status not in {ReminderStatus.SCHEDULED, ReminderStatus.SNOOZED}:
            continue
        if candidate.scheduled_for.astimezone(timezone.utc) > active_now.astimezone(timezone.utc):
            continue
        profile = snapshot.profiles.get(candidate.user_id)
        if profile is not None and (not profile.reminders_enabled or _in_quiet_hours(profile, active_now)):
            continue
        claimed = _claim_due_reminder(
            store,
            reminder_id=candidate.reminder_id,
            user_id=candidate.user_id,
            now=active_now,
        )
        if claimed is None:
            continue
        if claimed.destination_key != notifier.destination_key:
            error = "发送目标配置已改变，请重新创建提醒并确认"
            _finish_delivery(
                store,
                reminder=claimed,
                now=active_now,
                error=error,
            )
            outcomes.append(
                {
                    "reminder_id": str(claimed.reminder_id),
                    "status": "failed",
                    "error": error,
                }
            )
            continue
        if claimed.reminder_type == "check_in":
            if event_store is None:
                error = "主动 check-in 缺少健康事件存储，未发送"
                _finish_delivery(
                    store,
                    reminder=claimed,
                    now=active_now,
                    error=error,
                )
                outcomes.append({"reminder_id": str(claimed.reminder_id), "status": "failed", "error": error})
                continue
            message = build_check_in_message(claimed, event_store=event_store, now=active_now)
        else:
            message = (
                f"健康提醒\n{claimed.content}\n"
                f"计划时间：{claimed.scheduled_for.isoformat()}"
            )
        error: str | None = None
        outcome_unknown = False
        try:
            notifier.send(message)
        except FeishuDeliveryUnknown as exc:
            error = f"{type(exc).__name__}: {exc}"
            outcome_unknown = True
        except Exception as exc:  # Provider 错误必须转为可追踪状态，不能终止调度循环。
            error = f"{type(exc).__name__}: {exc}"
        finished_at = active_now
        _finish_delivery(
            store,
            reminder=claimed,
            now=finished_at,
            error=error,
            outcome_unknown=outcome_unknown,
        )
        outcomes.append(
            {
                "reminder_id": str(claimed.reminder_id),
                "status": "unknown" if outcome_unknown else ("failed" if error else "completed"),
                "error": error,
            }
        )
    return outcomes


class ReminderScheduler:
    """随本地网页进程运行的轻量轮询线程。"""

    def __init__(
        self,
        *,
        store: HealthOSStore,
        notifier: ReminderNotifier,
        event_store: HealthEventStore | None = None,
        poll_seconds: float = 30.0,
    ) -> None:
        self._store = store
        self._notifier = notifier
        self._event_store = event_store
        self._poll_seconds = max(float(poll_seconds), 5.0)
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="healthos-reminders", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                dispatch_due_reminders(
                    store=self._store,
                    notifier=self._notifier,
                    event_store=self._event_store,
                )
            except Exception:
                # 下一轮继续；单次存储故障不能让后台线程永久退出。
                pass
            self._stop_event.wait(self._poll_seconds)
