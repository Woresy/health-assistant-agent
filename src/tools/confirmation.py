"""健康事件保存、更新和删除操作的确认令牌。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any
from uuid import UUID

from src.health.models import (
    HealthEvent,
)


CONFIRMATION_VERSION = "1.0"
_PROCESS_SECRET = secrets.token_bytes(
    32
)


def _secret() -> bytes:
    """
    获取确认令牌签名密钥。

    正式运行时建议设置：
    HEALTH_CONFIRMATION_SECRET

    没有设置时使用当前 Python 进程中的临时密钥。
    """

    configured = os.getenv(
        "HEALTH_CONFIRMATION_SECRET",
        "",
    ).strip()

    if configured:
        return configured.encode(
            "utf-8"
        )

    return _PROCESS_SECRET


def health_event_digest(
    event: HealthEvent,
) -> str:
    """
    计算 HealthEvent 的稳定摘要。

    内容只要变化一个字段，摘要就会不同。
    """

    canonical = json.dumps(
        event.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        canonical
    ).hexdigest()


def _encode_body(
    body: dict[str, Any],
) -> str:
    """将令牌正文编码为 URL 安全字符串。"""

    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return (
        base64
        .urlsafe_b64encode(raw)
        .decode("ascii")
        .rstrip("=")
    )


def _issue_signed_token(
    *,
    operation_body: dict[str, Any],
    ttl_seconds: int,
) -> str:
    """为一个无副作用草稿签发确认令牌。"""

    if (
        isinstance(
            ttl_seconds,
            bool,
        )
        or not isinstance(
            ttl_seconds,
            int,
        )
        or ttl_seconds <= 0
    ):
        raise ValueError(
            "ttl_seconds 必须是大于 0 的整数"
        )

    issued_at = int(
        time.time()
    )

    body = {
        "confirmation_version": (
            CONFIRMATION_VERSION
        ),
        "issued_at": issued_at,
        "expires_at": (
            issued_at
            + ttl_seconds
        ),
        **operation_body,
    }

    encoded_body = _encode_body(
        body
    )

    signature = hmac.new(
        _secret(),
        encoded_body.encode(
            "ascii"
        ),
        hashlib.sha256,
    ).hexdigest()

    return (
        f"{encoded_body}."
        f"{signature}"
    )


def _decode_and_verify_token(
    token: str,
) -> tuple[
    bool,
    str,
    dict[str, Any] | None,
]:
    """验证签名、格式、版本和有效期。"""

    if (
        not token
        or "." not in token
    ):
        return (
            False,
            "确认令牌缺失或格式错误",
            None,
        )

    try:
        (
            encoded_body,
            supplied_signature,
        ) = token.split(
            ".",
            maxsplit=1,
        )

        expected_signature = (
            hmac.new(
                _secret(),
                encoded_body.encode(
                    "ascii"
                ),
                hashlib.sha256,
            ).hexdigest()
        )

        if not hmac.compare_digest(
            supplied_signature,
            expected_signature,
        ):
            return (
                False,
                "确认令牌签名无效",
                None,
            )

        padding = (
            "="
            * (
                -len(encoded_body)
                % 4
            )
        )

        decoded = (
            base64
            .urlsafe_b64decode(
                encoded_body
                + padding
            )
            .decode("utf-8")
        )

        body = json.loads(
            decoded
        )

        if not isinstance(
            body,
            dict,
        ):
            return (
                False,
                "确认令牌正文必须是对象",
                None,
            )

        if (
            body.get(
                "confirmation_version"
            )
            != CONFIRMATION_VERSION
        ):
            return (
                False,
                "不支持的确认令牌版本",
                None,
            )

        expires_at = int(
            body["expires_at"]
        )

        if expires_at < int(
            time.time()
        ):
            return (
                False,
                "确认令牌已过期",
                None,
            )
    except (
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        binascii.Error,
    ):
        return (
            False,
            "确认令牌无法解析",
            None,
        )

    return (
        True,
        "",
        body,
    )


def _same_text(
    left: Any,
    right: Any,
) -> bool:
    """使用恒定时间比较两个字符串。"""

    return hmac.compare_digest(
        str(left),
        str(right),
    )


def _validate_idempotency_key(
    idempotency_key: str,
) -> str:
    """签发令牌前验证幂等键。"""

    if not isinstance(
        idempotency_key,
        str,
    ):
        raise ValueError(
            "idempotency_key 必须是字符串"
        )

    normalized = (
        idempotency_key.strip()
    )

    if not normalized:
        raise ValueError(
            "idempotency_key 不能为空"
        )

    if len(normalized) > 128:
        raise ValueError(
            "idempotency_key 长度不得超过 128"
        )

    return normalized


def issue_confirmation_token(
    event: HealthEvent,
    ttl_seconds: int = 900,
) -> str:
    """
    为新增健康事件签发确认令牌。

    保留原函数名，兼容现有饮食保存流程。
    """

    return _issue_signed_token(
        operation_body={
            "action": "save",
            "event_digest": (
                health_event_digest(
                    event
                )
            ),
        },
        ttl_seconds=ttl_seconds,
    )


def verify_confirmation_token(
    token: str,
    event: HealthEvent,
) -> tuple[bool, str]:
    """验证新增健康事件确认令牌。"""

    (
        valid,
        message,
        body,
    ) = _decode_and_verify_token(
        token
    )

    if (
        not valid
        or body is None
    ):
        return (
            False,
            message,
        )

    if body.get("action") != "save":
        return (
            False,
            "确认令牌操作类型不匹配",
        )

    if not _same_text(
        body.get(
            "event_digest"
        ),
        health_event_digest(
            event
        ),
    ):
        return (
            False,
            "确认令牌与待保存事件不匹配",
        )

    return (
        True,
        "",
    )


def issue_update_confirmation_token(
    *,
    current_event: HealthEvent,
    replacement_event: HealthEvent,
    idempotency_key: str,
    ttl_seconds: int = 900,
) -> str:
    """为一个确定的事件更新草稿签发令牌。"""

    normalized_key = (
        _validate_idempotency_key(
            idempotency_key
        )
    )

    if (
        replacement_event.event_id
        != current_event.event_id
    ):
        raise ValueError(
            "更新草稿不得修改 event_id"
        )

    if (
        replacement_event.user_id
        != current_event.user_id
    ):
        raise ValueError(
            "更新草稿不得修改 user_id"
        )

    if (
        replacement_event.created_at
        != current_event.created_at
    ):
        raise ValueError(
            "更新草稿不得修改 created_at"
        )

    if (
        replacement_event.updated_at
        <= current_event.updated_at
    ):
        raise ValueError(
            "新版本 updated_at "
            "必须晚于旧版本"
        )

    return _issue_signed_token(
        operation_body={
            "action": "update",
            "event_id": str(
                current_event.event_id
            ),
            "user_id": (
                current_event.user_id
            ),
            "idempotency_key": (
                normalized_key
            ),
            "expected_updated_at": (
                current_event
                .updated_at
                .isoformat()
            ),
            "before_digest": (
                health_event_digest(
                    current_event
                )
            ),
            "after_digest": (
                health_event_digest(
                    replacement_event
                )
            ),
        },
        ttl_seconds=ttl_seconds,
    )


def verify_update_confirmation_token(
    *,
    token: str,
    event_id: UUID,
    user_id: str,
    replacement_event: HealthEvent,
    idempotency_key: str,
) -> tuple[
    bool,
    str,
    dict[str, Any] | None,
]:
    """验证更新令牌与本次更新参数是否完全一致。"""

    (
        valid,
        message,
        body,
    ) = _decode_and_verify_token(
        token
    )

    if (
        not valid
        or body is None
    ):
        return (
            False,
            message,
            None,
        )

    if body.get("action") != "update":
        return (
            False,
            "确认令牌操作类型不匹配",
            None,
        )

    expected_values = {
        "event_id": str(event_id),
        "user_id": user_id,
        "idempotency_key": (
            idempotency_key
        ),
        "after_digest": (
            health_event_digest(
                replacement_event
            )
        ),
    }

    for (
        field_name,
        expected_value,
    ) in expected_values.items():
        if not _same_text(
            body.get(field_name),
            expected_value,
        ):
            return (
                False,
                "确认令牌与当前更新请求不匹配",
                None,
            )

    if (
        not body.get(
            "before_digest"
        )
        or not body.get(
            "expected_updated_at"
        )
    ):
        return (
            False,
            "确认令牌缺少更新版本信息",
            None,
        )

    return (
        True,
        "",
        body,
    )


def issue_delete_confirmation_token(
    *,
    current_event: HealthEvent,
    idempotency_key: str,
    ttl_seconds: int = 900,
) -> str:
    """为指定事件删除草稿签发确认令牌。"""

    normalized_key = (
        _validate_idempotency_key(
            idempotency_key
        )
    )

    return _issue_signed_token(
        operation_body={
            "action": "delete",
            "event_id": str(
                current_event.event_id
            ),
            "user_id": (
                current_event.user_id
            ),
            "idempotency_key": (
                normalized_key
            ),
            "expected_updated_at": (
                current_event
                .updated_at
                .isoformat()
            ),
            "before_digest": (
                health_event_digest(
                    current_event
                )
            ),
        },
        ttl_seconds=ttl_seconds,
    )


def verify_delete_confirmation_token(
    *,
    token: str,
    event_id: UUID,
    user_id: str,
    idempotency_key: str,
) -> tuple[
    bool,
    str,
    dict[str, Any] | None,
]:
    """验证删除令牌与本次删除请求是否一致。"""

    (
        valid,
        message,
        body,
    ) = _decode_and_verify_token(
        token
    )

    if (
        not valid
        or body is None
    ):
        return (
            False,
            message,
            None,
        )

    if body.get("action") != "delete":
        return (
            False,
            "确认令牌操作类型不匹配",
            None,
        )

    expected_values = {
        "event_id": str(event_id),
        "user_id": user_id,
        "idempotency_key": (
            idempotency_key
        ),
    }

    for (
        field_name,
        expected_value,
    ) in expected_values.items():
        if not _same_text(
            body.get(field_name),
            expected_value,
        ):
            return (
                False,
                "确认令牌与当前删除请求不匹配",
                None,
            )

    if (
        not body.get(
            "before_digest"
        )
        or not body.get(
            "expected_updated_at"
        )
    ):
        return (
            False,
            "确认令牌缺少删除版本信息",
            None,
        )

    return (
        True,
        "",
        body,
    )