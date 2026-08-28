"""确认后原子更新健康事件。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import (
    ValidationError,
)

from src.health.models import (
    HealthEvent,
)
from src.storage.jsonl_store import (
    HealthEventConflictError,
    HealthEventNotFoundError,
    HealthEventStore,
    JsonlReadError,
    JsonlWriteError,
)
from src.tools.confirmation import (
    health_event_digest,
    verify_update_confirmation_token,
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


def update_health_event(
    *,
    event_id: str | UUID,
    user_id: str,
    replacement_event_input: (
        HealthEvent
        | dict[str, Any]
    ),
    confirmation_token: str,
    idempotency_key: str,
    store: HealthEventStore,
) -> dict[str, Any]:
    """
    验证确认令牌、版本和幂等状态后更新事件。

    replacement_event_input 必须来自准备阶段返回的
    proposed_event，不能临时重新拼接。
    """

    if not confirmation_token:
        return _error(
            "CONFIRMATION_REQUIRED",
            "必须先查看更新前后对比"
            "并明确确认修改",
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

    try:
        replacement_event = (
            replacement_event_input
            if isinstance(
                replacement_event_input,
                HealthEvent,
            )
            else (
                HealthEvent
                .model_validate(
                    replacement_event_input
                )
            )
        )
    except ValidationError as exc:
        return _error(
            "VALIDATION_ERROR",
            "更新后的健康事件"
            f"校验失败：{exc}",
        )

    if (
        replacement_event.event_id
        != parsed_event_id
    ):
        return _error(
            "EVENT_ID_MISMATCH",
            "replacement_event 的 "
            "event_id 与目标不一致",
        )

    if (
        replacement_event.user_id
        != normalized_user_id
    ):
        return _error(
            "USER_ID_MISMATCH",
            "replacement_event 的 "
            "user_id 与当前用户不一致",
        )

    (
        token_valid,
        token_message,
        token_body,
    ) = verify_update_confirmation_token(
        token=confirmation_token,
        event_id=parsed_event_id,
        user_id=normalized_user_id,
        replacement_event=(
            replacement_event
        ),
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

    if (
        current_event is None
        or current_event.user_id
        != normalized_user_id
    ):
        return _error(
            "EVENT_NOT_FOUND",
            "没有找到可更新的健康事件",
        )

    current_digest = (
        health_event_digest(
            current_event
        )
    )

    expected_after_digest = str(
        token_body[
            "after_digest"
        ]
    )

    if (
        current_digest
        == expected_after_digest
    ):
        return {
            "ok": True,
            "data": {
                "event": (
                    current_event
                    .model_dump(
                        mode="json"
                    )
                ),
                "idempotent": True,
            },
            "error": None,
        }

    expected_before_digest = str(
        token_body[
            "before_digest"
        ]
    )

    if (
        current_digest
        != expected_before_digest
    ):
        return _error(
            "EVENT_VERSION_CONFLICT",
            "事件在确认前已经发生变化，"
            "请重新读取并生成更新草稿",
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
        updated_event = store.update(
            event_id=parsed_event_id,
            replacement=(
                replacement_event
            ),
            expected_updated_at=(
                expected_updated_at
            ),
        )
    except HealthEventNotFoundError:
        return _error(
            "EVENT_NOT_FOUND",
            "没有找到可更新的健康事件",
        )
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
            "event": (
                updated_event.model_dump(
                    mode="json"
                )
            ),
            "idempotent": False,
        },
        "error": None,
    }