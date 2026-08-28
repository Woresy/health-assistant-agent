"""确认后原子删除健康事件。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from src.storage.jsonl_store import (
    HealthEventConflictError,
    HealthEventNotFoundError,
    HealthEventStore,
    JsonlReadError,
    JsonlWriteError,
)
from src.tools.confirmation import (
    health_event_digest,
    verify_delete_confirmation_token,
)


def _error(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """构造统一工具错误。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def _parse_event_id(
    event_id: str | UUID,
) -> UUID | None:
    """解析事件 UUID。"""

    try:
        return (
            event_id
            if isinstance(
                event_id,
                UUID,
            )
            else UUID(
                str(event_id)
            )
        )
    except (
        ValueError,
        TypeError,
        AttributeError,
    ):
        return None


def delete_health_event(
    *,
    event_id: str | UUID,
    user_id: str,
    confirmation_token: str,
    idempotency_key: str,
    store: HealthEventStore,
) -> dict[str, Any]:
    """
    验证确认令牌、目标版本和幂等状态后删除事件。
    """

    if not confirmation_token:
        return _error(
            "CONFIRMATION_REQUIRED",
            "必须先查看待删除记录"
            "并明确确认删除",
        )

    if not isinstance(
        user_id,
        str,
    ):
        return _error(
            "USER_ID_REQUIRED",
            "user_id 必须是字符串",
        )

    normalized_user_id = (
        user_id.strip()
    )

    if not normalized_user_id:
        return _error(
            "USER_ID_REQUIRED",
            "user_id 不能为空",
        )

    if not isinstance(
        idempotency_key,
        str,
    ):
        return _error(
            "IDEMPOTENCY_KEY_REQUIRED",
            "idempotency_key "
            "必须是字符串",
        )

    normalized_key = (
        idempotency_key.strip()
    )

    if not normalized_key:
        return _error(
            "IDEMPOTENCY_KEY_REQUIRED",
            "idempotency_key 不能为空",
        )

    if len(normalized_key) > 128:
        return _error(
            "IDEMPOTENCY_KEY_INVALID",
            "idempotency_key "
            "长度不得超过 128",
        )

    parsed_event_id = (
        _parse_event_id(
            event_id
        )
    )

    if parsed_event_id is None:
        return _error(
            "EVENT_ID_INVALID",
            "event_id 必须是有效 UUID",
        )

    (
        token_valid,
        token_message,
        token_body,
    ) = verify_delete_confirmation_token(
        token=confirmation_token,
        event_id=parsed_event_id,
        user_id=normalized_user_id,
        idempotency_key=(
            normalized_key
        ),
    )

    if (
        not token_valid
        or token_body is None
    ):
        return _error(
            "INVALID_CONFIRMATION_TOKEN",
            token_message,
        )

    try:
        current_event = (
            store.find_by_event_id(
                parsed_event_id
            )
        )
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )

    if current_event is None:
        return {
            "ok": True,
            "data": {
                "event_id": str(
                    parsed_event_id
                ),
                "deleted_event": None,
                "deleted": True,
                "idempotent": True,
            },
            "error": None,
        }

    if (
        current_event.user_id
        != normalized_user_id
    ):
        return _error(
            "EVENT_NOT_FOUND",
            "没有找到可删除的健康事件",
        )

    expected_before_digest = str(
        token_body[
            "before_digest"
        ]
    )

    if (
        health_event_digest(
            current_event
        )
        != expected_before_digest
    ):
        return _error(
            "EVENT_VERSION_CONFLICT",
            "事件在确认前已经发生变化，"
            "请重新读取并生成删除草稿",
        )

    try:
        expected_updated_at = (
            datetime.fromisoformat(
                str(
                    token_body[
                        "expected_updated_at"
                    ]
                )
            )
        )
    except (
        ValueError,
        TypeError,
        KeyError,
    ):
        return _error(
            "INVALID_CONFIRMATION_TOKEN",
            "确认令牌中的版本时间无效",
        )

    try:
        deleted_event = store.delete(
            event_id=parsed_event_id,
            expected_updated_at=(
                expected_updated_at
            ),
        )
    except HealthEventNotFoundError:
        return {
            "ok": True,
            "data": {
                "event_id": str(
                    parsed_event_id
                ),
                "deleted_event": None,
                "deleted": True,
                "idempotent": True,
            },
            "error": None,
        }
    except HealthEventConflictError:
        return _error(
            "EVENT_VERSION_CONFLICT",
            "事件版本已经变化，"
            "请重新读取并确认",
        )
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )
    except JsonlWriteError as exc:
        return _error(
            "WRITE_FAILED",
            str(exc),
        )
    except ValueError as exc:
        return _error(
            "VALIDATION_ERROR",
            str(exc),
        )

    return {
        "ok": True,
        "data": {
            "event_id": str(
                parsed_event_id
            ),
            "deleted_event": (
                deleted_event.model_dump(
                    mode="json"
                )
            ),
            "deleted": True,
            "idempotent": False,
        },
        "error": None,
    }