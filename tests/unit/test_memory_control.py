"""长期记忆查看、遗忘、导出和全部清除。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.models import PendingConfirmation
from src.agent.tool_router import HealthToolRouter
from src.healthos.memory_control import (
    clear_user_memories,
    delete_user_memory,
    export_user_memories,
    list_user_memories,
)
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore
from src.storage.sqlite_store import SQLiteDatabase, SQLiteHealthOSStore


@pytest.fixture(params=["json", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> HealthOSStore:
    if request.param == "sqlite":
        return SQLiteHealthOSStore(SQLiteDatabase(tmp_path / "healthos.db"))
    return HealthOSStore(tmp_path / "healthos.json")


def _confirm_profile(store: HealthOSStore, tmp_path: Path) -> None:
    router = HealthToolRouter(
        HealthEventStore(tmp_path / "events.jsonl"),
        healthos_store=store,
    )
    prepared = router.dispatch(
        tool_name="prepare_profile_update",
        arguments={
            "patch": {
                "coach_style": "rational",
                "dietary_preferences": ["少油"],
                "exclusions": ["花生"],
                "reminders_enabled": False,
            }
        },
        user_id="memory-user",
        timezone_name="Asia/Shanghai",
        session_id="memory-session",
        call_id="memory-profile",
    )
    data = prepared.result["data"]
    router.confirm(
        PendingConfirmation(
            action=data["action"],
            tool_name="prepare_profile_update",
            draft_data=data,
        )
    )


def test_confirmed_profile_is_visible_and_exportable(store: HealthOSStore, tmp_path: Path) -> None:
    _confirm_profile(store, tmp_path)
    listed = list_user_memories(user_id="memory-user", store=store)
    assert listed["ok"] is True
    assert listed["data"]["count"] == 4
    assert {item["label"] for item in listed["data"]["memories"]} == {"教练风格", "饮食偏好", "忌口", "提醒偏好"}
    exported = export_user_memories(user_id="memory-user", store=store)
    assert exported["ok"] is True
    assert exported["data"]["schema_version"] == "1.0"
    assert len(exported["data"]["memories"]) == 4


def test_delete_requires_confirmation_and_removes_prompt_preference(store: HealthOSStore, tmp_path: Path) -> None:
    _confirm_profile(store, tmp_path)
    memory = next(
        item
        for item in list_user_memories(user_id="memory-user", store=store)["data"]["memories"]
        if item["label"] == "饮食偏好"
    )
    rejected = delete_user_memory(
        user_id="memory-user",
        memory_id=memory["memory_id"],
        confirmed=False,
        store=store,
    )
    assert rejected["error"]["error_code"] == "CONFIRMATION_REQUIRED"
    assert store.get_profile("memory-user", "Asia/Shanghai").dietary_preferences == ["少油"]
    deleted = delete_user_memory(
        user_id="memory-user",
        memory_id=memory["memory_id"],
        confirmed=True,
        store=store,
    )
    assert deleted["ok"] is True
    assert store.get_profile("memory-user", "Asia/Shanghai").dietary_preferences == []


def test_clear_requires_confirmation_and_resets_preferences_only(store: HealthOSStore, tmp_path: Path) -> None:
    _confirm_profile(store, tmp_path)
    rejected = clear_user_memories(user_id="memory-user", confirmed=False, store=store)
    assert rejected["error"]["error_code"] == "CONFIRMATION_REQUIRED"
    cleared = clear_user_memories(user_id="memory-user", confirmed=True, store=store)
    assert cleared["data"]["cleared_count"] == 4
    assert list_user_memories(user_id="memory-user", store=store)["data"]["count"] == 0
    profile = store.get_profile("memory-user", "Asia/Shanghai")
    assert profile.coach_style.value == "gentle"
    assert profile.dietary_preferences == []
    assert profile.exclusions == []
    assert profile.reminders_enabled is True
