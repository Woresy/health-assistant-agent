"""OpenAI-compatible Agent 模型适配器测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent.models import (
    AgentMessage,
)
from src.agent.openai_model import (
    AgentConfigurationError,
    AgentProtocolError,
    OpenAICompatibleAgentModel,
    load_agent_settings,
)


class FakeCompletions:
    """模拟 client.chat.completions。"""

    def __init__(
        self,
        response: object,
    ) -> None:
        self.response = response
        self.requests: list[
            dict[str, Any]
        ] = []

    def create(
        self,
        **kwargs: Any,
    ) -> object:
        self.requests.append(
            kwargs
        )
        return self.response


class FakeClient:
    """最小 OpenAI Client 替身。"""

    def __init__(
        self,
        response: object,
    ) -> None:
        self.completions = (
            FakeCompletions(
                response
            )
        )
        self.chat = SimpleNamespace(
            completions=(
                self.completions
            )
        )


def _response(
    *,
    content: str | None,
    tool_calls: (
        list[object]
        | None
    ) = None,
    finish_reason: str = "stop",
) -> object:
    """构造模拟 Chat Completion。"""

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=(
                    finish_reason
                ),
                message=(
                    SimpleNamespace(
                        content=content,
                        tool_calls=(
                            tool_calls
                            or []
                        ),
                    )
                ),
            )
        ]
    )


def test_provider_performance_contract_has_a_real_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER_MODE", "openai_compatible")
    monkeypatch.setenv("AGENT_API_KEY", "test-only")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.delenv("AGENT_TARGET_RESPONSE_SECONDS", raising=False)
    monkeypatch.delenv("AGENT_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("AGENT_MAX_RETRIES", raising=False)

    settings = load_agent_settings(tmp_path / "missing.env")

    assert settings is not None
    assert settings.target_response_seconds == 15
    assert settings.timeout == 60
    assert settings.max_retries == 0


def test_provider_rejects_retries_that_multiply_the_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("AGENT_PROVIDER_MODE", "openai_compatible")
    monkeypatch.setenv("AGENT_API_KEY", "test-only")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("AGENT_MAX_RETRIES", "1")

    with pytest.raises(AgentConfigurationError, match="AGENT_MAX_RETRIES"):
        load_agent_settings(tmp_path / "missing.env")


def test_plain_text_reply() -> None:
    client = FakeClient(
        _response(
            content="你好。",
        )
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model="test-model",
        )
    )

    reply = model.complete(
        [
            AgentMessage(
                role="system",
                content="系统提示",
            ),
            AgentMessage(
                role="user",
                content="你好",
            ),
        ],
        [],
    )

    assert reply.content == "你好。"
    assert reply.tool_calls == ()

    request = (
        client.completions
        .requests[0]
    )

    assert request["model"] == (
        "test-model"
    )
    assert (
        request["tool_choice"]
        == "auto"
    )


def test_thinking_mode_is_forwarded_for_deepseek_compatible_providers() -> None:
    client = FakeClient(
        _response(content="完成。")
    )
    model = OpenAICompatibleAgentModel(
        client=client,
        model="deepseek-v4-flash",
        thinking_mode="disabled",
    )

    model.complete(
        [AgentMessage(role="user", content="创建提醒")],
        [],
    )

    request = client.completions.requests[0]
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_tool_call_is_parsed() -> None:
    raw_tool_call = (
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name=(
                    "prepare_health_event"
                ),
                arguments=(
                    json.dumps(
                        {
                            "event_type": (
                                "water"
                            ),
                            "amount_ml": 500,
                        },
                        ensure_ascii=False,
                    )
                ),
            ),
        )
    )

    client = FakeClient(
        _response(
            content=None,
            tool_calls=[
                raw_tool_call
            ],
        )
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model="test-model",
        )
    )

    reply = model.complete(
        [
            AgentMessage(
                role="system",
                content="系统提示",
            ),
            AgentMessage(
                role="user",
                content=(
                    "记录喝水500毫升"
                ),
            ),
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": (
                        "prepare_health_event"
                    ),
                    "parameters": {
                        "type": "object"
                    },
                },
            }
        ],
    )

    assert len(
        reply.tool_calls
    ) == 1

    tool_call = (
        reply.tool_calls[0]
    )

    assert (
        tool_call.call_id
        == "call-1"
    )
    assert (
        tool_call.name
        == "prepare_health_event"
    )
    assert (
        tool_call.arguments[
            "amount_ml"
        ]
        == 500
    )


def test_previous_tool_result_is_converted_back_to_provider_messages(
) -> None:
    client = FakeClient(
        _response(
            content="查询完成。",
        )
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model="test-model",
        )
    )

    assistant_tool_message = (
        AgentMessage(
            role="assistant",
            content=json.dumps(
                {
                    "tool_call": {
                        "call_id": (
                            "query-1"
                        ),
                        "name": (
                            "query_health_events"
                        ),
                        "arguments": {
                            "date": (
                                "2026-08-28"
                            )
                        },
                    }
                },
                ensure_ascii=False,
            ),
            tool_call_id="query-1",
            tool_name=(
                "query_health_events"
            ),
        )
    )

    tool_result_message = (
        AgentMessage(
            role="tool",
            content=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "events": []
                    },
                    "error": None,
                }
            ),
            tool_call_id="query-1",
            tool_name=(
                "query_health_events"
            ),
        )
    )

    model.complete(
        [
            AgentMessage(
                role="system",
                content="系统提示",
            ),
            AgentMessage(
                role="user",
                content="查询记录",
            ),
            assistant_tool_message,
            tool_result_message,
        ],
        [],
    )

    request_messages = (
        client.completions
        .requests[0][
            "messages"
        ]
    )

    assistant_message = (
        request_messages[2]
    )
    tool_message = (
        request_messages[3]
    )

    assert (
        assistant_message["role"]
        == "assistant"
    )
    assert (
        assistant_message[
            "tool_calls"
        ][0]["id"]
        == "query-1"
    )
    assert (
        tool_message["role"]
        == "tool"
    )
    assert (
        tool_message[
            "tool_call_id"
        ]
        == "query-1"
    )


def test_invalid_tool_arguments_json_is_rejected(
) -> None:
    raw_tool_call = (
        SimpleNamespace(
            id="call-bad",
            function=SimpleNamespace(
                name=(
                    "prepare_health_event"
                ),
                arguments="{invalid",
            ),
        )
    )

    client = FakeClient(
        _response(
            content=None,
            tool_calls=[
                raw_tool_call
            ],
        )
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model="test-model",
        )
    )

    with pytest.raises(
        AgentProtocolError,
        match="不是有效 JSON",
    ):
        model.complete(
            [
                AgentMessage(
                    role="system",
                    content="系统提示",
                ),
                AgentMessage(
                    role="user",
                    content="记录喝水",
                ),
            ],
            [],
        )


def test_length_truncation_is_rejected(
) -> None:
    client = FakeClient(
        _response(
            content="不完整",
            finish_reason="length",
        )
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model="test-model",
        )
    )

    with pytest.raises(
        AgentProtocolError,
        match="长度限制",
    ):
        model.complete(
            [
                AgentMessage(
                    role="system",
                    content="系统提示",
                ),
                AgentMessage(
                    role="user",
                    content="测试",
                ),
            ],
            [],
        )
