"""LangSmith 可观测性配置。

健康数据默认不离开本机：启用时若未显式配置，自动隐藏 Trace 输入与输出。
LangSmith 故障不得阻断健康记录、确认或提醒流程。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LangSmithStatus:
    enabled: bool
    ready: bool
    project: str
    message: str


def configure_langsmith() -> LangSmithStatus:
    """校验环境并应用健康场景的隐私安全默认值。"""

    tracing_enabled = _enabled(os.getenv("LANGSMITH_TRACING"))
    project = os.getenv(
        "LANGSMITH_PROJECT",
        "health-assistant-agent-local",
    ).strip() or "health-assistant-agent-local"

    if not tracing_enabled:
        return LangSmithStatus(
            enabled=False,
            ready=False,
            project=project,
            message="LangSmith 观测未启用",
        )

    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        # 配置错误时降级到本地 Trace，不能让远程观测影响核心业务。
        os.environ["LANGSMITH_TRACING"] = "false"
        return LangSmithStatus(
            enabled=True,
            ready=False,
            project=project,
            message="LangSmith 缺少 API Key，已安全降级为本地 Trace",
        )

    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "true")
    os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "true")

    privacy = (
        _enabled(os.getenv("LANGSMITH_HIDE_INPUTS"))
        and _enabled(os.getenv("LANGSMITH_HIDE_OUTPUTS"))
    )
    privacy_text = "输入输出已隐藏" if privacy else "请确认健康数据上传授权"
    return LangSmithStatus(
        enabled=True,
        ready=True,
        project=project,
        message=f"LangSmith 已启用：{project}；{privacy_text}",
    )


def wrap_openai_client(client: Any, *, model: str) -> Any:
    """为 OpenAI-compatible Client 增加 LLM 子运行；失败时原样降级。"""

    if not _enabled(os.getenv("LANGSMITH_TRACING")):
        return client
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return client

    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(
            client,
            tracing_extra={
                "tags": ["healthos", "llm"],
                "metadata": {
                    "component": "agent-model",
                    "model": model,
                    "privacy": "inputs-outputs-hidden-by-default",
                },
            },
            chat_name="HealthOS Agent Model",
        )
    except (ImportError, RuntimeError, TypeError):
        return client
