"""飞书 Provider 与确认后提醒调度的边界测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from src.automation.feishu import (
    FeishuConfigurationError,
    FeishuWebhookConfig,
    FeishuWebhookNotifier,
    FeishuDeliveryUnknown,
    build_feishu_signature,
)
from src.automation.reminder_scheduler import dispatch_due_reminders
from src.health.models import HealthEvent
from src.healthos.models import Reminder, ReminderStatus, ReminderTransition, UserProfile
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore


NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


class FakeNotifier:
    channel = "feishu"
    destination_label = "飞书群「健康提醒」"
    destination_key = "feishu-test-target"

    def __init__(self, *, fail: bool = False, unknown: bool = False) -> None:
        self.fail = fail
        self.unknown = unknown
        self.messages: list[str] = []

    def send(self, text: str) -> dict[str, Any]:
        self.messages.append(text)
        if self.unknown:
            raise FeishuDeliveryUnknown("timeout")
        if self.fail:
            raise RuntimeError("temporary failure")
        return {"ok": True}


def _store_with_due_reminder(
    tmp_path: Path,
    *,
    quiet: bool = False,
    check_in: bool = False,
) -> HealthOSStore:
    store = HealthOSStore(tmp_path / "healthos.json")

    def seed(state: Any) -> None:
        state.profiles["user-1"] = UserProfile(
            user_id="user-1",
            timezone_name="UTC",
            quiet_hours_start="00:00" if quiet else None,
            quiet_hours_end="23:59" if quiet else None,
            updated_at=NOW,
        )
        state.reminders.append(
            Reminder(
                reminder_id=uuid4(),
                user_id="user-1",
                content="起来喝水",
                scheduled_for=NOW - timedelta(minutes=1),
                timezone_name="UTC",
                delivery_channel="feishu",
                reminder_type="check_in" if check_in else "standard",
                recurrence="daily" if check_in else "once",
                check_in_focus=["meal", "water", "exercise"] if check_in else [],
                destination_label="飞书群「健康提醒」",
                destination_key="feishu-test-target",
                status=ReminderStatus.SCHEDULED,
                created_at=NOW - timedelta(hours=1),
                updated_at=NOW - timedelta(hours=1),
                transitions=[
                    ReminderTransition(
                        status=ReminderStatus.SCHEDULED,
                        occurred_at=NOW - timedelta(hours=1),
                        reason="用户确认创建",
                    )
                ],
            )
        )

    store.update(seed)
    return store


def test_due_confirmed_feishu_reminder_is_sent_once(tmp_path: Path) -> None:
    store = _store_with_due_reminder(tmp_path)
    notifier = FakeNotifier()

    first = dispatch_due_reminders(store=store, notifier=notifier, now=NOW)
    second = dispatch_due_reminders(store=store, notifier=notifier, now=NOW)

    reminder = store.read().reminders[0]
    assert first[0]["status"] == "completed"
    assert second == []
    assert len(notifier.messages) == 1
    assert reminder.status == ReminderStatus.COMPLETED
    assert reminder.delivery_attempts == 1
    assert reminder.last_delivery_error is None


def test_delivery_failure_is_persisted_and_quiet_hours_skip(tmp_path: Path) -> None:
    failing_store = _store_with_due_reminder(tmp_path / "failure")
    failed = dispatch_due_reminders(
        store=failing_store,
        notifier=FakeNotifier(fail=True),
        now=NOW,
    )
    assert failed[0]["status"] == "failed"
    assert failing_store.read().reminders[0].status == ReminderStatus.FAILED

    quiet_store = _store_with_due_reminder(tmp_path / "quiet", quiet=True)
    notifier = FakeNotifier()
    assert dispatch_due_reminders(store=quiet_store, notifier=notifier, now=NOW) == []
    assert notifier.messages == []
    assert quiet_store.read().reminders[0].status == ReminderStatus.SCHEDULED


def test_active_check_in_uses_confirmed_facts_and_schedules_next_day(tmp_path: Path) -> None:
    store = _store_with_due_reminder(tmp_path, check_in=True)
    notifier = FakeNotifier()
    event_store = HealthEventStore(tmp_path / "events.jsonl")
    event_store.append(
        HealthEvent.model_validate(
            {
                "schema_version": "1.1",
                "event_id": str(uuid4()),
                "user_id": "user-1",
                "event_type": "water",
                "occurred_at": NOW - timedelta(hours=1),
                "payload": {"amount_ml": 350, "beverage": "水", "note": ""},
                "source_refs": [],
                "input_source": "chat",
                "created_at": NOW - timedelta(hours=1),
                "updated_at": NOW - timedelta(hours=1),
            }
        )
    )

    result = dispatch_due_reminders(
        store=store,
        notifier=notifier,
        event_store=event_store,
        now=NOW,
    )

    reminder = store.read().reminders[0]
    assert result[0]["status"] == "completed"
    assert reminder.status == ReminderStatus.SCHEDULED
    assert reminder.scheduled_for == NOW + timedelta(days=1, minutes=-1)
    assert "今天已确认：饮水 350 ml" in notifier.messages[0]
    assert "暂未看到饮食、运动记录" in notifier.messages[0]
    assert "不会自动写入" in notifier.messages[0]


def test_uncertain_delivery_is_not_automatically_retried(tmp_path: Path) -> None:
    store = _store_with_due_reminder(tmp_path)

    result = dispatch_due_reminders(
        store=store,
        notifier=FakeNotifier(unknown=True),
        now=NOW,
    )

    assert result[0]["status"] == "unknown"
    assert store.read().reminders[0].status == ReminderStatus.UNKNOWN
    assert dispatch_due_reminders(store=store, notifier=FakeNotifier(), now=NOW) == []


def test_weekday_check_in_skips_the_weekend(tmp_path: Path) -> None:
    store = _store_with_due_reminder(tmp_path, check_in=True)

    def use_weekdays(state: Any) -> None:
        state.reminders[0] = state.reminders[0].model_copy(update={"recurrence": "weekdays"})

    store.update(use_weekdays)
    dispatch_due_reminders(
        store=store,
        notifier=FakeNotifier(),
        event_store=HealthEventStore(tmp_path / "events.jsonl"),
        now=NOW,
    )

    next_run = store.read().reminders[0].scheduled_for
    assert next_run.weekday() == 0
    assert next_run.date().isoformat() == "2026-09-07"


def test_changed_webhook_target_requires_new_confirmation(tmp_path: Path) -> None:
    store = _store_with_due_reminder(tmp_path)
    notifier = FakeNotifier()
    notifier.destination_key = "a-different-target"

    result = dispatch_due_reminders(store=store, notifier=notifier, now=NOW)

    assert result[0]["status"] == "failed"
    assert "重新创建提醒并确认" in result[0]["error"]
    assert notifier.messages == []
    assert store.read().reminders[0].status == ReminderStatus.FAILED


def test_feishu_provider_validates_destination_and_posts_text(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FeishuConfigurationError):
        FeishuWebhookNotifier(
            FeishuWebhookConfig(enabled=True, webhook_url="https://example.com/hook")
        )

    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self, _: int) -> bytes:
            return b'{"code":0,"msg":"success"}'

    def fake_urlopen(request: Any, timeout: float) -> Response:
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notifier = FeishuWebhookNotifier(
        FeishuWebhookConfig(
            enabled=True,
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/example",
            signing_secret="secret",
            destination_label="飞书测试群",
        )
    )
    result = notifier.send("健康提醒\n起来喝水")

    assert result["ok"] is True
    assert captured["payload"]["msg_type"] == "text"
    assert captured["payload"]["content"]["text"] == "健康提醒\n起来喝水"
    assert captured["payload"]["sign"]
    assert build_feishu_signature(123, "secret") == build_feishu_signature(123, "secret")
