"""LangGraph 编排器的关键健康流程验收。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.langgraph_runner import LangGraphAgentRunner
from src.agent.models import (
    AgentMessage,
    AgentModelReply,
    ModelToolCall,
)
from src.agent.runner import ConversationSession
from src.agent.tool_router import HealthToolRouter
from src.agent.trace import AgentTraceStore, TracedConversationSession
from src.storage.jsonl_store import HealthEventStore


class FakeAgentModel:
    def __init__(self, replies: list[AgentModelReply]) -> None:
        self._replies = list(replies)
        self.received_messages: list[list[AgentMessage]] = []

    def complete(
        self,
        messages: Sequence[AgentMessage],
        tool_definitions: Sequence[dict[str, Any]],
    ) -> AgentModelReply:
        del tool_definitions
        self.received_messages.append(list(messages))
        if not self._replies:
            raise AssertionError("FakeAgentModel 没有剩余响应")
        return self._replies.pop(0)


class FailOnceConfirmationRouter(HealthToolRouter):
    """第一次确认失败，第二次使用真实工具执行。"""

    def __init__(self, store: HealthEventStore) -> None:
        super().__init__(store)
        self.confirm_count = 0

    def confirm(self, pending):  # type: ignore[no-untyped-def]
        self.confirm_count += 1
        if self.confirm_count == 1:
            return {
                "ok": False,
                "data": None,
                "error": {
                    "error_code": "TEMPORARY_WRITE_FAILURE",
                    "message": "模拟临时写入失败",
                },
            }
        return super().confirm(pending)


def tool_reply(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str,
) -> AgentModelReply:
    return AgentModelReply(
        tool_calls=(
            ModelToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        )
    )


def build_session(
    tmp_path: Path,
    replies: list[AgentModelReply],
    *,
    traced: bool = False,
) -> tuple[
    ConversationSession | TracedConversationSession,
    HealthEventStore,
    LangGraphAgentRunner,
]:
    store = HealthEventStore(tmp_path / "health_events.jsonl")
    runner = LangGraphAgentRunner(
        model=FakeAgentModel(replies),
        router=HealthToolRouter(store),
        max_model_rounds=4,
    )
    if traced:
        session: ConversationSession | TracedConversationSession = (
            TracedConversationSession(
                runner=runner,
                session_id="langgraph-e2e",
                user_id="user-1",
                trace_store=AgentTraceStore(
                    tmp_path / "agent_traces.jsonl"
                ),
            )
        )
    else:
        session = ConversationSession(
            runner=runner,
            session_id="langgraph-e2e",
            user_id="user-1",
        )
    return session, store, runner


def test_langgraph_water_interrupt_confirm_and_trace(
    tmp_path: Path,
) -> None:
    session, store, runner = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": "water",
                    "amount_ml": 500,
                    "occurred_at": "2026-08-30T09:00:00+08:00",
                },
                call_id="water-1",
            )
        ],
        traced=True,
    )

    prepared = session.send("记录喝水500毫升")

    assert prepared.state.value == "awaiting_confirmation"
    assert prepared.pending_confirmation is not None
    assert store.read_all() == []

    snapshot = runner.graph.get_state(
        {"configurable": {"thread_id": "langgraph-e2e"}}
    )
    assert snapshot.next == ("await_confirmation",)
    assert snapshot.interrupts[0].value["kind"] == "confirmation"
    assert "confirmation_token" not in str(snapshot.interrupts[0].value)

    confirmed = session.confirm()

    assert confirmed.state.value == "completed"
    assert len(store.read_all()) == 1
    assert store.read_all()[0].payload.amount_ml == 500

    trace_store = AgentTraceStore(tmp_path / "agent_traces.jsonl")
    traces = trace_store.read_recent(limit=10)
    assert [trace.action for trace in traces] == ["confirm", "send"]


def test_langgraph_clarification_resumes_same_graph(
    tmp_path: Path,
) -> None:
    session, store, runner = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": "exercise",
                    "activity_type": "跑步",
                },
                call_id="exercise-missing",
            ),
            tool_reply(
                "prepare_health_event",
                {"duration_minutes": 30},
                call_id="exercise-complete",
            ),
        ],
    )

    missing = session.send("我跑步了")
    assert missing.state.value == "awaiting_clarification"
    assert missing.pending_task is not None
    assert missing.pending_task.missing_parameters == ["duration_minutes"]
    assert store.read_all() == []

    snapshot = runner.graph.get_state(
        {"configurable": {"thread_id": "langgraph-e2e"}}
    )
    assert snapshot.next == ("await_clarification",)

    prepared = session.send("30分钟")
    assert prepared.state.value == "awaiting_confirmation"
    assert prepared.pending_confirmation is not None
    assert session.state.turn_count == 2

    cancelled = session.cancel()
    assert cancelled.state.value == "cancelled"
    assert store.read_all() == []


def test_langgraph_read_tool_loops_back_to_model(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "query_health_events",
                {"event_type": "water"},
                call_id="query-1",
            ),
            AgentModelReply(content="当前没有饮水记录。"),
        ],
    )

    result = session.send("查询饮水记录")

    assert result.state.value == "completed"
    assert result.model_rounds == 2
    assert [step.tool_name for step in result.tool_steps] == [
        "query_health_events"
    ]
    assert store.read_all() == []


def test_langgraph_unknown_tool_fails_without_side_effect(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "delete_everything",
                {},
                call_id="unknown-1",
            )
        ],
    )

    result = session.send("删除全部健康数据")

    assert result.state.value == "failed"
    assert result.finish_reason.value == "invalid_arguments"
    assert "白名单" in result.answer
    assert store.read_all() == []


def test_langgraph_required_tool_guard_retries_model(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            AgentModelReply(content="已经替你记录喝水。"),
            tool_reply(
                "prepare_health_event",
                {"event_type": "water", "amount_ml": 300},
                call_id="guard-retry",
            ),
        ],
    )

    result = session.send("记录喝水300毫升")

    assert result.state.value == "awaiting_confirmation"
    assert result.model_rounds == 2
    assert result.pending_confirmation is not None
    assert store.read_all() == []


def test_langgraph_confirmation_failure_interrupts_again_for_retry(
    tmp_path: Path,
) -> None:
    store = HealthEventStore(tmp_path / "health_events.jsonl")
    router = FailOnceConfirmationRouter(store)
    runner = LangGraphAgentRunner(
        model=FakeAgentModel(
            [
                tool_reply(
                    "prepare_health_event",
                    {"event_type": "water", "amount_ml": 250},
                    call_id="retry-confirm",
                )
            ]
        ),
        router=router,
    )
    session = ConversationSession(
        runner=runner,
        session_id="confirmation-retry",
        user_id="user-1",
    )

    prepared = session.send("记录喝水250毫升")
    assert prepared.state.value == "awaiting_confirmation"

    failed = session.confirm()
    assert failed.state.value == "awaiting_confirmation"
    assert failed.finish_reason.value == "tool_error"
    assert failed.pending_confirmation is not None
    assert store.read_all() == []

    snapshot = runner.graph.get_state(
        {"configurable": {"thread_id": "confirmation-retry"}}
    )
    assert snapshot.next == ("await_confirmation",)

    confirmed = session.confirm()
    assert confirmed.state.value == "completed"
    assert len(store.read_all()) == 1
