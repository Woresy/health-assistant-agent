"""测试环境的公共外部服务配置。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def configured_test_feishu(monkeypatch: pytest.MonkeyPatch) -> None:
    """提醒测试只生成草稿，使用不会发送的占位飞书配置。"""

    monkeypatch.setenv("FEISHU_REMINDER_ENABLED", "true")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test-only",
    )
    monkeypatch.setenv("FEISHU_DESTINATION_LABEL", "飞书群「测试提醒」")
