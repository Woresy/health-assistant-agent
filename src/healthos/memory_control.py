"""用户可直接查看、遗忘和导出的长期记忆控制。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

from src.healthos.models import CoachStyle, MemoryType, UserMemory, UserProfile
from src.storage.healthos_store import HealthOSStore, HealthOSStoreError


MEMORY_LABELS = {
    MemoryType.DIETARY_PREFERENCE: "饮食偏好",
    MemoryType.EXCLUSION: "忌口",
    MemoryType.COACH_STYLE: "教练风格",
    MemoryType.REMINDER_PREFERENCE: "提醒偏好",
    MemoryType.USER_NOTE: "长期偏好",
}


def _memory_id(user_id: str, memory_type: MemoryType, content: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"healthos-memory:{user_id}:{memory_type.value}:{content}")


def sync_profile_memories(state: Any, profile: UserProfile) -> None:
    """将已确认档案偏好同步为可逐条删除的记忆。"""

    retained = [
        item
        for item in state.memories
        if item.user_id != profile.user_id or item.source != "profile_confirmation"
    ]
    now = datetime.now(timezone.utc)
    values: list[tuple[MemoryType, str]] = [
        (MemoryType.COACH_STYLE, profile.coach_style.value),
        *[(MemoryType.DIETARY_PREFERENCE, item) for item in profile.dietary_preferences],
        *[(MemoryType.EXCLUSION, item) for item in profile.exclusions],
        (
            MemoryType.REMINDER_PREFERENCE,
            (
                f"enabled={str(profile.reminders_enabled).lower()};"
                f"quiet={profile.quiet_hours_start or '-'}~{profile.quiet_hours_end or '-'}"
            ),
        ),
    ]
    state.memories = retained + [
        UserMemory(
            memory_id=_memory_id(profile.user_id, memory_type, content),
            user_id=profile.user_id,
            memory_type=memory_type,
            content=content,
            source="profile_confirmation",
            confirmed_at=now,
            updated_at=now,
        )
        for memory_type, content in values
    ]


def list_user_memories(*, user_id: str, store: HealthOSStore) -> dict[str, Any]:
    try:
        items = [item for item in store.read().memories if item.user_id == user_id]
    except HealthOSStoreError as exc:
        return {"ok": False, "data": None, "error": {"error_code": "MEMORY_READ_FAILED", "message": str(exc)}}
    items.sort(key=lambda item: (item.memory_type.value, item.content))
    return {
        "ok": True,
        "data": {
            "memories": [
                {
                    **item.model_dump(mode="json"),
                    "label": MEMORY_LABELS[item.memory_type],
                }
                for item in items
            ],
            "count": len(items),
        },
        "error": None,
    }


def export_user_memories(*, user_id: str, store: HealthOSStore) -> dict[str, Any]:
    result = list_user_memories(user_id=user_id, store=store)
    if not result["ok"]:
        return result
    return {
        "ok": True,
        "data": {
            "schema_version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "memories": result["data"]["memories"],
        },
        "error": None,
    }


def delete_user_memory(
    *, user_id: str, memory_id: str, confirmed: bool, store: HealthOSStore
) -> dict[str, Any]:
    if not confirmed:
        return {"ok": False, "data": None, "error": {"error_code": "CONFIRMATION_REQUIRED", "message": "请先确认要遗忘这条内容"}}

    def mutation(state: Any) -> dict[str, Any]:
        index = store.find_memory_index(state, user_id, UUID(memory_id))
        memory = state.memories[index]
        profile = state.profiles.get(user_id)
        if profile is not None:
            update: dict[str, Any] = {"version": profile.version + 1, "updated_at": datetime.now(timezone.utc)}
            if memory.memory_type == MemoryType.DIETARY_PREFERENCE:
                update["dietary_preferences"] = [item for item in profile.dietary_preferences if item != memory.content]
            elif memory.memory_type == MemoryType.EXCLUSION:
                update["exclusions"] = [item for item in profile.exclusions if item != memory.content]
            elif memory.memory_type == MemoryType.COACH_STYLE:
                update["coach_style"] = CoachStyle.GENTLE
            elif memory.memory_type == MemoryType.REMINDER_PREFERENCE:
                update.update({"reminders_enabled": True, "quiet_hours_start": None, "quiet_hours_end": None})
            state.profiles[user_id] = profile.model_copy(update=update)
        del state.memories[index]
        return {"memory_id": memory_id, "label": MEMORY_LABELS[memory.memory_type], "content": memory.content}

    try:
        deleted = store.update(mutation)
    except (ValueError, HealthOSStoreError) as exc:
        return {"ok": False, "data": None, "error": {"error_code": "MEMORY_DELETE_FAILED", "message": str(exc)}}
    return {"ok": True, "data": deleted, "error": None}


def clear_user_memories(*, user_id: str, confirmed: bool, store: HealthOSStore) -> dict[str, Any]:
    if not confirmed:
        return {"ok": False, "data": None, "error": {"error_code": "CONFIRMATION_REQUIRED", "message": "请再次确认清除全部长期记忆"}}

    def mutation(state: Any) -> int:
        count = sum(1 for item in state.memories if item.user_id == user_id)
        state.memories = [item for item in state.memories if item.user_id != user_id]
        profile = state.profiles.get(user_id)
        if profile is not None:
            state.profiles[user_id] = UserProfile(
                user_id=user_id,
                timezone_name=profile.timezone_name,
                updated_at=datetime.now(timezone.utc),
                version=profile.version + 1,
            )
        return count

    try:
        count = store.update(mutation)
    except HealthOSStoreError as exc:
        return {"ok": False, "data": None, "error": {"error_code": "MEMORY_CLEAR_FAILED", "message": str(exc)}}
    return {"ok": True, "data": {"cleared_count": count}, "error": None}
