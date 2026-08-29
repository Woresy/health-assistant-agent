"""健康管理 Agent 完整流程测试。

覆盖八条完整流程：

1. 饮水草稿、确认和保存；
2. 体重查询不修改数据；
3. 运动缺参、补参和取消；
4. 未知工具失败不落数据；
5. 非法体重失败不落数据；
6. 待确认期间阻止新的写请求；
7. 删除草稿取消后保留原事件；
8. Agent Trace 写入失败不影响健康事件保存。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.agent.models import (
    AgentMessage,
    AgentModelReply,
    ModelToolCall,
)
from src.agent.runner import AgentRunner
from src.agent.tool_router import (
    HealthToolRouter,
)
from src.agent.trace import (
    AgentTrace,
    AgentTraceStore,
    AgentTraceWriteError,
    TracedConversationSession,
)
from src.health.models import (
    EventType,
    HealthEvent,
    InputSource,
    WeightPayload,
)
from src.storage.jsonl_store import (
    HealthEventStore,
)


class FakeAgentModel:
    """按照给定顺序返回模型响应。"""

    def __init__(
        self,
        replies: list[
            AgentModelReply
        ],
    ) -> None:
        self._replies = list(
            replies
        )

        self.received_messages: list[
            list[AgentMessage]
        ] = []

        self.received_tools: list[
            list[dict[str, Any]]
        ] = []

    def complete(
        self,
        messages: Sequence[
            AgentMessage
        ],
        tool_definitions: Sequence[
            dict[str, Any]
        ],
    ) -> AgentModelReply:
        """返回下一条预设模型响应。"""

        self.received_messages.append(
            list(messages)
        )

        self.received_tools.append(
            list(tool_definitions)
        )

        if not self._replies:
            raise AssertionError(
                "FakeAgentModel "
                "没有剩余响应。"
            )

        return self._replies.pop(
            0
        )


class FailingTraceStore:
    """模拟 Agent Trace 写入失败。"""

    def append(
        self,
        trace: AgentTrace,
    ) -> None:
        del trace

        raise AgentTraceWriteError(
            "模拟 Agent Trace "
            "写入失败。"
        )


def tool_reply(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call-1",
) -> AgentModelReply:
    """构造模型工具调用响应。"""

    return AgentModelReply(
        content=None,
        tool_calls=(
            ModelToolCall(
                call_id=call_id,
                name=tool_name,
                arguments=arguments,
            ),
        ),
    )


def text_reply(
    content: str,
) -> AgentModelReply:
    """构造模型普通文本响应。"""

    return AgentModelReply(
        content=content,
        tool_calls=(),
    )


def build_session(
    tmp_path: Path,
    *,
    replies: list[
        AgentModelReply
    ],
    trace_store: (
        AgentTraceStore
        | FailingTraceStore
        | None
    ) = None,
) -> tuple[
    TracedConversationSession,
    HealthEventStore,
    AgentTraceStore | FailingTraceStore,
]:
    """创建相互隔离的 Agent E2E 会话。"""

    event_store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    router = HealthToolRouter(
        event_store
    )

    model = FakeAgentModel(
        replies
    )

    runner = AgentRunner(
        model=model,
        router=router,
        max_model_rounds=4,
    )

    effective_trace_store: (
        AgentTraceStore
        | FailingTraceStore
    )

    if trace_store is None:
        effective_trace_store = (
            AgentTraceStore(
                tmp_path
                / "agent_traces.jsonl"
            )
        )
    else:
        effective_trace_store = (
            trace_store
        )

    session = TracedConversationSession(
        runner=runner,
        session_id="e2e-session",
        user_id="e2e-user",
        timezone_name="Asia/Shanghai",
        trace_store=(
            effective_trace_store
        ),
    )

    return (
        session,
        event_store,
        effective_trace_store,
    )


def existing_weight_event() -> HealthEvent:
    """构造一条预置的体重事件。"""

    occurred_at = datetime(
        2026,
        8,
        29,
        8,
        30,
        tzinfo=timezone.utc,
    )

    return HealthEvent(
        event_id=uuid4(),
        user_id="e2e-user",
        event_type=(
            EventType.WEIGHT
        ),
        occurred_at=occurred_at,
        payload=WeightPayload(
            weight_kg=64.8,
        ),
        source_refs=[],
        input_source=(
            InputSource.CHAT
        ),
        created_at=occurred_at,
        updated_at=occurred_at,
    )


def test_water_prepare_confirm_and_save(
    tmp_path: Path,
) -> None:
    """饮水事件必须先生成草稿，再确认保存。"""

    (
        session,
        event_store,
        trace_store,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": (
                        "water"
                    ),
                    "amount_ml": 500,
                    "occurred_at": (
                        "2026-08-29"
                        "T09:00:00+08:00"
                    ),
                },
            )
        ],
    )

    prepared = session.send(
        "记录喝水500毫升"
    )

    assert (
        prepared.state.value
        == "awaiting_confirmation"
    )

    assert (
        prepared
        .pending_confirmation
        is not None
    )

    assert (
        event_store.read_all()
        == []
    )

    confirmed = session.confirm()

    events = (
        event_store.read_all()
    )

    assert (
        confirmed.state.value
        == "completed"
    )

    assert len(events) == 1

    assert (
        events[0].event_type
        == EventType.WATER
    )

    assert (
        events[0]
        .payload
        .amount_ml
        == 500
    )

    assert isinstance(
        trace_store,
        AgentTraceStore,
    )

    traces = trace_store.read_recent(
        limit=20
    )

    assert len(traces) == 2

    assert (
        traces[0].action
        == "confirm"
    )

    assert (
        traces[1].action
        == "send"
    )


def test_weight_query_does_not_modify_data(
    tmp_path: Path,
) -> None:
    """查询体重事件不能修改 JSONL。"""

    (
        session,
        event_store,
        _,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "query_health_events",
                {
                    "event_type": (
                        "weight"
                    ),
                    "timezone_name": (
                        "Asia/Shanghai"
                    ),
                },
            ),
            text_reply(
                "最近一次体重记录"
                "是64.8公斤。"
            ),
        ],
    )

    original = (
        existing_weight_event()
    )

    event_store.append(
        original
    )

    before = (
        event_store.read_all()
    )

    result = session.send(
        "查询我的体重记录"
    )

    after = (
        event_store.read_all()
    )

    assert (
        result.state.value
        == "completed"
    )

    assert (
        result.model_rounds
        == 2
    )

    assert len(
        result.tool_steps
    ) == 1

    assert (
        result.tool_steps[0]
        .tool_name
        == "query_health_events"
    )

    assert before == after

    assert len(after) == 1

    assert (
        after[0].event_id
        == original.event_id
    )


def test_exercise_missing_parameter_can_be_completed(
    tmp_path: Path,
) -> None:
    """运动缺少时长时追问，补充后生成草稿。"""

    (
        session,
        event_store,
        trace_store,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": (
                        "exercise"
                    ),
                    "activity_type": (
                        "跑步"
                    ),
                },
                call_id=(
                    "exercise-missing"
                ),
            ),
            tool_reply(
                "prepare_health_event",
                {
                    "duration_minutes": (
                        30
                    ),
                },
                call_id=(
                    "exercise-completed"
                ),
            ),
        ],
    )

    missing = session.send(
        "记录跑步"
    )

    assert (
        missing.state.value
        == "awaiting_clarification"
    )

    assert (
        missing.pending_task
        is not None
    )

    assert (
        event_store.read_all()
        == []
    )

    completed = session.send(
        "30分钟"
    )

    assert (
        completed.state.value
        == "awaiting_confirmation"
    )

    assert (
        completed
        .pending_confirmation
        is not None
    )

    assert (
        event_store.read_all()
        == []
    )

    cancelled = session.cancel()

    assert (
        cancelled.state.value
        == "cancelled"
    )

    assert (
        event_store.read_all()
        == []
    )

    assert isinstance(
        trace_store,
        AgentTraceStore,
    )

    traces = trace_store.read_recent(
        limit=20
    )

    assert len(traces) == 3

    assert (
        traces[0].action
        == "cancel"
    )


def test_unknown_tool_fails_without_writing_event(
    tmp_path: Path,
) -> None:
    """白名单之外的工具必须失败且不写数据。"""

    (
        session,
        event_store,
        trace_store,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                (
                    "delete_everything_"
                    "without_confirmation"
                ),
                {
                    "user_id": (
                        "e2e-user"
                    ),
                },
            )
        ],
    )

    result = session.send(
        "执行不存在的工具"
    )

    assert (
        result.state.value
        == "failed"
    )

    assert (
        result.finish_reason.value
        == "invalid_arguments"
    )

    assert (
        event_store.read_all()
        == []
    )

    assert isinstance(
        trace_store,
        AgentTraceStore,
    )

    traces = trace_store.read_recent(
        limit=20
    )

    assert len(traces) == 1

    assert (
        traces[0].state
        == "failed"
    )

    assert (
        traces[0].finish_reason
        == "invalid_arguments"
    )

    assert (
        traces[0].tool_steps
        == ()
    )


def test_invalid_weight_fails_without_writing_event(
    tmp_path: Path,
) -> None:
    """非法体重参数必须失败且不写数据。"""

    (
        session,
        event_store,
        trace_store,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": (
                        "weight"
                    ),
                    "weight_kg": -2,
                    "occurred_at": (
                        "2026-08-29"
                        "T08:00:00+08:00"
                    ),
                },
            )
        ],
    )

    result = session.send(
        "记录体重负二公斤"
    )

    assert (
        result.state.value
        == "failed"
    )

    assert (
        result.finish_reason.value
        == "invalid_arguments"
    )

    assert (
        event_store.read_all()
        == []
    )

    assert isinstance(
        trace_store,
        AgentTraceStore,
    )

    traces = trace_store.read_recent(
        limit=20
    )

    assert len(traces) == 1

    assert (
        traces[0].state
        == "failed"
    )

    assert (
        traces[0].finish_reason
        == "invalid_arguments"
    )


def test_pending_confirmation_blocks_second_request(
    tmp_path: Path,
) -> None:
    """存在待确认草稿时不能执行第二个写请求。"""

    (
        session,
        event_store,
        _,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": (
                        "water"
                    ),
                    "amount_ml": 300,
                    "occurred_at": (
                        "2026-08-29"
                        "T10:00:00+08:00"
                    ),
                },
            )
        ],
    )

    first = session.send(
        "记录喝水300毫升"
    )

    assert (
        first.state.value
        == "awaiting_confirmation"
    )

    assert (
        event_store.read_all()
        == []
    )

    blocked = session.send(
        "再记录喝水900毫升"
    )

    assert (
        blocked.state.value
        == "awaiting_confirmation"
    )

    assert (
        blocked.model_rounds
        == 0
    )

    assert (
        blocked
        .pending_confirmation
        is not None
    )

    assert (
        event_store.read_all()
        == []
    )

    confirmed = session.confirm()

    assert (
        confirmed.state.value
        == "completed"
    )

    events = (
        event_store.read_all()
    )

    assert len(events) == 1

    assert (
        events[0]
        .payload
        .amount_ml
        == 300
    )


def test_delete_cancel_keeps_original_event(
    tmp_path: Path,
) -> None:
    """删除草稿取消后，原事件必须继续存在。"""

    original = (
        existing_weight_event()
    )

    (
        session,
        event_store,
        _,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                (
                    "prepare_delete_"
                    "health_event"
                ),
                {
                    "event_id": str(
                        original.event_id
                    ),
                },
            )
        ],
    )

    event_store.append(
        original
    )

    prepared = session.send(
        "删除事件"
        f"{original.event_id}"
    )

    assert (
        prepared.state.value
        == "awaiting_confirmation"
    )

    assert (
        prepared
        .pending_confirmation
        is not None
    )

    cancelled = session.cancel()

    remaining = (
        event_store
        .find_by_event_id(
            original.event_id
        )
    )

    assert (
        cancelled.state.value
        == "cancelled"
    )

    assert remaining is not None

    assert (
        remaining.event_id
        == original.event_id
    )

    assert len(
        event_store.read_all()
    ) == 1


def test_trace_failure_does_not_break_health_event_save(
    tmp_path: Path,
) -> None:
    """Trace 写入失败不能破坏健康事件保存。"""

    (
        session,
        event_store,
        _,
    ) = build_session(
        tmp_path,
        replies=[
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": (
                        "water"
                    ),
                    "amount_ml": 450,
                    "occurred_at": (
                        "2026-08-29"
                        "T11:00:00+08:00"
                    ),
                },
            )
        ],
        trace_store=(
            FailingTraceStore()
        ),
    )

    prepared = session.send(
        "记录喝水450毫升"
    )

    assert (
        prepared.state.value
        == "awaiting_confirmation"
    )

    assert (
        session.last_trace_warning
        is not None
    )

    confirmed = session.confirm()

    events = (
        event_store.read_all()
    )

    assert (
        confirmed.state.value
        == "completed"
    )

    assert len(events) == 1

    assert (
        events[0]
        .payload
        .amount_ml
        == 450
    )

    assert (
        session.last_trace_warning
        is not None
    )

    assert (
        session.last_trace_warning
        .startswith(
            "TRACE_WRITE_FAILED:"
        )
    )