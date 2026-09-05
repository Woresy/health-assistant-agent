"""SQLite 主存储、旧文件迁移和回滚边界。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from src.agent.models import AgentMessage, SessionState
from src.health.models import HealthEvent, WaterPayload
from src.healthos.models import HealthOSState, UserProfile
from src.storage.conversation_store import ConversationStore
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore
from src.storage.sqlite_store import (
    SQLiteConversationStore,
    SQLiteDatabase,
    SQLiteHealthEventStore,
    SQLiteHealthOSStore,
    migrate_legacy_storage,
)


NOW = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
SESSION_ID = "conversation-0123456789abcdef0123456789abcdef"


def _event(amount: float = 350) -> HealthEvent:
    return HealthEvent(
        event_id=uuid4(),
        user_id="user-1",
        event_type="water",
        occurred_at=NOW,
        payload=WaterPayload(amount_ml=amount),
        input_source="chat",
        created_at=NOW,
        updated_at=NOW,
    )


def _session() -> SessionState:
    return SessionState(
        session_id=SESSION_ID,
        user_id="user-1",
        messages=(
            AgentMessage(role="system", content="rules"),
            AgentMessage(role="user", content="我喝了水"),
        ),
        turn_count=1,
    )


def test_sqlite_event_crud_and_conversation_persistence(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "healthos.db")
    events = SQLiteHealthEventStore(database)
    conversations = SQLiteConversationStore(database)
    original = _event()
    events.append(original)

    assert events.query(user_id="user-1") == [original]
    replacement = original.model_copy(
        update={
            "payload": WaterPayload(amount_ml=500),
            "updated_at": NOW + timedelta(minutes=1),
        }
    )
    events.update(
        event_id=original.event_id,
        replacement=replacement,
        expected_updated_at=original.updated_at,
    )
    assert events.find_by_event_id(original.event_id) == replacement

    conversations.save(_session())
    assert conversations.load(SESSION_ID) == _session()
    conversations.delete(SESSION_ID)
    assert conversations.load(SESSION_ID) is None
    assert database.integrity_check()["ok"] is True


def test_sqlite_healthos_update_is_transactional(tmp_path: Path) -> None:
    store = SQLiteHealthOSStore(tmp_path / "healthos.db")

    def save_profile(state: HealthOSState) -> str:
        state.profiles["user-1"] = UserProfile(
            user_id="user-1",
            timezone_name="Asia/Shanghai",
            updated_at=NOW,
        )
        return "saved"

    assert store.update(save_profile) == "saved"
    assert store.get_profile("user-1", "UTC").timezone_name == "Asia/Shanghai"

    def fail_after_change(state: HealthOSState) -> None:
        state.profiles.clear()
        raise RuntimeError("stop")

    try:
        store.update(fail_after_change)
    except RuntimeError:
        pass
    assert "user-1" in store.read().profiles


def test_legacy_migration_is_idempotent_and_keeps_sources(tmp_path: Path) -> None:
    event_path = tmp_path / "health_events.jsonl"
    healthos_path = tmp_path / "healthos_state.json"
    conversation_dir = tmp_path / "conversations"
    legacy_event_store = HealthEventStore(event_path)
    legacy_event_store.append(_event())
    legacy_healthos_store = HealthOSStore(healthos_path)
    legacy_healthos_store.update(
        lambda state: state.profiles.update(
            {
                "user-1": UserProfile(
                    user_id="user-1",
                    timezone_name="Asia/Shanghai",
                    updated_at=NOW,
                )
            }
        )
    )
    ConversationStore(conversation_dir).save(_session())
    database = SQLiteDatabase(tmp_path / "healthos.db")

    first = migrate_legacy_storage(
        database,
        events_path=event_path,
        healthos_path=healthos_path,
        conversations_path=conversation_dir,
    )
    second = migrate_legacy_storage(
        database,
        events_path=event_path,
        healthos_path=healthos_path,
        conversations_path=conversation_dir,
    )

    assert first == {"health_events": 1, "healthos_entities": 1, "conversations": 1}
    assert second == {"health_events": 0, "healthos_entities": 0, "conversations": 0}
    assert event_path.exists() and healthos_path.exists()
    assert SQLiteHealthEventStore(database).read_all() == legacy_event_store.read_all()
    assert SQLiteConversationStore(database).load(SESSION_ID) == _session()
