"""LangSmith 可选观测的隐私与降级测试。"""

from __future__ import annotations

from src.observability.langsmith import configure_langsmith, wrap_openai_client


def _clear(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_langsmith_disabled_is_a_noop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear(monkeypatch)
    client = object()

    status = configure_langsmith()

    assert status.ready is False
    assert wrap_openai_client(client, model="test") is client


def test_missing_key_safely_disables_remote_tracing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    status = configure_langsmith()

    assert status.enabled is True
    assert status.ready is False
    assert "缺少 API Key" in status.message


def test_health_privacy_defaults_hide_inputs_and_outputs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    status = configure_langsmith()

    assert status.ready is True
    assert status.project == "health-assistant-agent-local"
    assert status.message.endswith("输入输出已隐藏")
