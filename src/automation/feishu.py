"""飞书自定义机器人 Webhook Provider。

Webhook 与签名密钥只从本机环境读取，不进入模型上下文、业务数据库或 Trace。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class FeishuConfigurationError(ValueError):
    """飞书机器人配置不完整或目标地址不安全。"""


class FeishuDeliveryError(RuntimeError):
    """飞书机器人拒绝消息或网络发送失败。"""


class FeishuDeliveryUnknown(FeishuDeliveryError):
    """请求可能已经到达飞书，但客户端没有拿到确定结果。"""


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FeishuWebhookConfig:
    """不含任何可持久化方法的本机飞书配置。"""

    enabled: bool = False
    webhook_url: str = ""
    signing_secret: str = ""
    destination_label: str = "飞书健康提醒群"
    timeout_seconds: float = 8.0

    @classmethod
    def from_environment(cls) -> "FeishuWebhookConfig":
        timeout_text = os.getenv("FEISHU_REQUEST_TIMEOUT", "8").strip()
        try:
            timeout = float(timeout_text)
        except ValueError:
            timeout = 8.0
        return cls(
            enabled=_enabled(os.getenv("FEISHU_REMINDER_ENABLED")),
            webhook_url=os.getenv("FEISHU_WEBHOOK_URL", "").strip(),
            signing_secret=os.getenv("FEISHU_SIGNING_SECRET", "").strip(),
            destination_label=(
                os.getenv("FEISHU_DESTINATION_LABEL", "飞书健康提醒群").strip()
                or "飞书健康提醒群"
            ),
            timeout_seconds=min(max(timeout, 1.0), 30.0),
        )

    @property
    def available(self) -> bool:
        if not self.enabled or not self.webhook_url:
            return False
        try:
            self.validate()
        except FeishuConfigurationError:
            return False
        return True

    @property
    def destination_key(self) -> str:
        """不可逆目标指纹，用于阻止确认后静默切换群聊。"""

        if not self.webhook_url:
            return "unconfigured"
        return hashlib.sha256(self.webhook_url.encode("utf-8")).hexdigest()[:24]

    def validate(self) -> None:
        if not self.enabled:
            raise FeishuConfigurationError("飞书提醒未启用")
        parsed = urlparse(self.webhook_url)
        if parsed.scheme != "https":
            raise FeishuConfigurationError("飞书 Webhook 必须使用 HTTPS")
        if parsed.hostname not in {"open.feishu.cn", "open.larksuite.com"}:
            raise FeishuConfigurationError("飞书 Webhook 域名不在允许列表中")
        if not parsed.path.startswith("/open-apis/bot/v2/hook/"):
            raise FeishuConfigurationError("飞书 Webhook 路径格式无效")


def build_feishu_signature(timestamp: int, secret: str) -> str:
    """按飞书自定义机器人签名协议生成 Base64 HMAC-SHA256。"""

    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


class FeishuWebhookNotifier:
    """发送纯文本健康提醒，不把 Webhook 暴露给上层。"""

    channel = "feishu"

    def __init__(self, config: FeishuWebhookConfig) -> None:
        config.validate()
        self._config = config

    @property
    def destination_label(self) -> str:
        return self._config.destination_label

    @property
    def destination_key(self) -> str:
        return self._config.destination_key

    def send(self, text: str) -> dict[str, Any]:
        normalized = text.strip()
        if not normalized:
            raise FeishuDeliveryError("提醒内容为空")
        payload: dict[str, Any] = {
            "msg_type": "text",
            "content": {"text": normalized},
        }
        if self._config.signing_secret:
            timestamp = int(time.time())
            payload.update(
                {
                    "timestamp": str(timestamp),
                    "sign": build_feishu_signature(timestamp, self._config.signing_secret),
                }
            )
        request = urllib.request.Request(
            self._config.webhook_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as response:
                raw = response.read(64 * 1024).decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise FeishuDeliveryError(f"飞书通知请求失败：HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FeishuDeliveryUnknown(
                f"飞书通知结果待核实：{type(exc).__name__}"
            ) from exc
            raise FeishuDeliveryError(f"飞书通知网络失败：{type(exc).__name__}") from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FeishuDeliveryError("飞书返回了无法识别的响应") from exc
        code = result.get("code", result.get("StatusCode", -1))
        if str(code) != "0":
            message = str(result.get("msg", result.get("StatusMessage", "发送失败")))
            raise FeishuDeliveryError(f"飞书机器人拒绝消息：{message[:160]}")
        return {"ok": True, "provider": "feishu", "destination": self.destination_label}
