"""Agent 必须调用健康工具的协议门控测试。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.models import (
    AgentMessage,
    AgentModelReply,
    ModelToolCall,
)
from src.agent.runner import (
    AgentRunner,
    ConversationSession,
)
from src.agent.tool_router import (
    HealthToolRouter,
)
from src.storage.jsonl_store import (
    HealthEventStore,
)


class FakeAgentModel:
    """按照给定顺序返回模型响应。"""

    def __init__(
        self,
        replies: list[AgentModelReply],
    ) -> None:
        self._replies = list(replies)
        self.call_count = 0

    def complete(
        self,
        messages: Sequence[AgentMessage],
        tool_definitions: Sequence[
            dict[str, Any]
        ],
    ) -> AgentModelReply:
        del messages
        del tool_definitions

        self.call_count += 1

        if not self._replies:
            raise AssertionError(
                "FakeAgentModel 没有剩余响应。"
            )

        return self._replies.pop(0)


def _build_session(
    tmp_path: Path,
    replies: list[AgentModelReply],
    *,
    max_model_rounds: int = 4,
) -> tuple[
    ConversationSession,
    HealthEventStore,
    FakeAgentModel,
]:
    """创建相互隔离的测试会话。"""

    store = HealthEventStore(
        tmp_path / "health_events.jsonl"
    )
    model = FakeAgentModel(replies)
    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(store),
        max_model_rounds=max_model_rounds,
    )
    session = ConversationSession(
        runner=runner,
        session_id="guard-session",
        user_id="guard-user",
        timezone_name="Asia/Shanghai",
    )

    return session, store, model


def test_fake_weight_draft_is_retried_with_tool(
    tmp_path: Path,
) -> None:
    """模型假装生成草稿时必须重新调用工具。"""

    session, store, model = _build_session(
        tmp_path,
        [
            AgentModelReply(
                content=(
                    "已生成体重事件保存草稿，"
                    "请确认保存。"
                )
            ),
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="weight-tool-call",
                        name="prepare_health_event",
                        arguments={
                            "event_type": "weight",
                            "weight_kg": 64.8,
                        },
                    ),
                )
            ),
        ],
    )

    result = session.send(
        "记录体重64.8公斤"
    )

    assert model.call_count == 2
    assert (
        result.state.value
        == "awaiting_confirmation"
    )
    assert (
        result.finish_reason.value
        == "awaiting_confirmation"
    )
    assert result.pending_confirmation is not None
    assert (
        result.tool_steps[0].tool_name
        == "prepare_health_event"
    )
    assert store.read_all() == []

    confirmed = session.confirm()
    events = store.read_all()

    assert confirmed.state.value == "completed"
    assert len(events) == 1
    assert events[0].event_type.value == "weight"
    assert events[0].payload.weight_kg == 64.8


def test_repeated_fake_draft_fails_without_data(
    tmp_path: Path,
) -> None:
    """模型持续拒绝工具调用时必须安全失败。"""

    session, store, model = _build_session(
        tmp_path,
        [
            AgentModelReply(
                content="已经生成草稿。"
            ),
            AgentModelReply(
                content="已经为你保存。"
            ),
        ],
        max_model_rounds=2,
    )

    result = session.send(
        "记录喝水500毫升"
    )

    assert model.call_count == 2
    assert result.state.value == "failed"
    assert result.finish_reason.value == "tool_error"
    assert result.pending_confirmation is None
    assert store.read_all() == []


def test_normal_greeting_can_return_plain_text(
    tmp_path: Path,
) -> None:
    """不涉及健康工具的问候允许文本回答。"""

    session, store, model = _build_session(
        tmp_path,
        [
            AgentModelReply(
                content=(
                    "你好，我可以帮助你"
                    "管理健康记录。"
                )
            )
        ],
    )

    result = session.send("你好")

    assert model.call_count == 1
    assert result.state.value == "completed"
    assert result.finish_reason.value == "completed"
    assert store.read_all() == []


def test_query_can_return_text_after_tool_execution(
    tmp_path: Path,
) -> None:
    """查询工具执行后允许模型用文本汇总。"""

    session, store, model = _build_session(
        tmp_path,
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="query-water",
                        name="query_health_events",
                        arguments={
                            "event_type": "water",
                            "timezone_name": (
                                "Asia/Shanghai"
                            ),
                        },
                    ),
                )
            ),
            AgentModelReply(
                content="今天暂时没有饮水记录。"
            ),
        ],
    )

    result = session.send(
        "查询今天的饮水记录"
    )

    assert model.call_count == 2
    assert result.state.value == "completed"
    assert result.finish_reason.value == "completed"
    assert len(result.tool_steps) == 1
    assert (
        result.tool_steps[0].tool_name
        == "query_health_events"
    )
    assert store.read_all() == []
