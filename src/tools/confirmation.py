"""保存确认令牌的签发与验证。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from src.health.models import HealthEvent


_PROCESS_SECRET = secrets.token_bytes(32)


def _secret() -> bytes:
    """生产环境可使用环境变量；本地缺省值不阻塞启动。"""

    configured = os.getenv("HEALTH_CONFIRMATION_SECRET", "").strip()
    return configured.encode("utf-8") if configured else _PROCESS_SECRET


def _event_digest(event: HealthEvent) -> str:
    canonical = json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _encode_body(body: dict[str, Any]) -> str:
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def issue_confirmation_token(
    event: HealthEvent,
    ttl_seconds: int = 900,
) -> str:
    """为当前待保存事件签发默认有效期 15 分钟的令牌。"""

    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds 必须大于 0")

    encoded_body = _encode_body(
        {
            "event_digest": _event_digest(event),
            "expires_at": int(time.time()) + ttl_seconds,
        }
    )
    signature = hmac.new(
        _secret(),
        encoded_body.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_body}.{signature}"


def verify_confirmation_token(
    token: str,
    event: HealthEvent,
) -> tuple[bool, str]:
    """校验签名、有效期以及令牌绑定的事件内容。"""

    if not token or "." not in token:
        return False, "确认令牌缺失或格式错误"

    try:
        encoded_body, supplied_signature = token.split(".", maxsplit=1)
        expected_signature = hmac.new(
            _secret(),
            encoded_body.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            supplied_signature,
            expected_signature,
        ):
            return False, "确认令牌签名无效"

        padding = "=" * (-len(encoded_body) % 4)
        decoded = base64.urlsafe_b64decode(
            encoded_body + padding
        ).decode("utf-8")
        body = json.loads(decoded)

        if int(body["expires_at"]) < int(time.time()):
            return False, "确认令牌已过期"

        if not hmac.compare_digest(
            str(body["event_digest"]),
            _event_digest(event),
        ):
            return False, "确认令牌与待保存事件不匹配"
    except (
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return False, "确认令牌无法解析"

    return True, ""