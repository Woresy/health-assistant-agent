"""保存健康事件的统一工具入口。"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from src.health.models import HealthEvent
from src.storage.jsonl_store import (
    HealthEventStore,
    JsonlReadError,
    JsonlWriteError,
)
from src.tools.confirmation import verify_confirmation_token


def _error(error_code: str, message: str) -> dict[str, Any]:
    """生成统一失败结构。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def save_health_event(
    event_input: HealthEvent | dict[str, Any],
    confirmation_token: str,
    idempotency_key: str,
    store: HealthEventStore,
) -> dict[str, Any]:
    """
    校验确认、执行持久幂等检查并追加一个完整 HealthEvent。

    返回结构始终为 {ok, data, error}。
    """

    if not confirmation_token:
        return _error(
            "CONFIRMATION_REQUIRED",
            "必须先查看估算结果并明确确认保存",
        )

    normalized_key = idempotency_key.strip()
    if not normalized_key:
        return _error(
            "IDEMPOTENCY_KEY_REQUIRED",
            "idempotency_key 不能为空",
        )
    if len(normalized_key) > 128:
        return _error(
            "IDEMPOTENCY_KEY_INVALID",
            "idempotency_key 长度不得超过 128",
        )

    try:
        event = (
            event_input
            if isinstance(event_input, HealthEvent)
            else HealthEvent.model_validate(event_input)
        )
    except ValidationError as exc:
        return _error(
            "VALIDATION_ERROR",
            f"健康事件校验失败：{exc}",
        )

    token_valid, token_message = verify_confirmation_token(
        confirmation_token,
        event,
    )
    if not token_valid:
        return _error(
            "INVALID_CONFIRMATION_TOKEN",
            token_message,
        )

    stable_event_id = uuid5(
        NAMESPACE_URL,
        f"health-event:{event.user_id}:{normalized_key}",
    )
    persisted_event = event.model_copy(
        update={"event_id": stable_event_id}
    )

    try:
        existing = store.find_by_event_id(stable_event_id)
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )

    if existing is not None:
        return {
            "ok": True,
            "data": {
                "event": existing.model_dump(mode="json"),
                "idempotent": True,
            },
            "error": None,
        }

    try:
        store.append(persisted_event)
    except JsonlWriteError as exc:
        return _error(
            "WRITE_FAILED",
            str(exc),
        )

    return {
        "ok": True,
        "data": {
            "event": persisted_event.model_dump(mode="json"),
            "idempotent": False,
        },
        "error": None,
    }