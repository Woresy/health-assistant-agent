"""Agent Trace 模型和 JSONL 存储测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.trace import (
    AgentTrace,
    AgentTraceReadError,
    AgentTraceStore,
    AgentTraceToolStep,
)


def make_trace(
    *,
    action: str = "send",
    turn_count: int = 1,
    state: str = "completed",
) -> AgentTrace:
    """构造一条不含隐私数据的 Trace。"""

    return AgentTrace(
        created_at="2026-08-29T10:00:00+00:00",
        session_hash="a" * 64,
        user_hash="b" * 64,
        action=action,
        turn_count=turn_count,
        input_length=8,
        input_sha256="c" * 64,
        state=state,
        finish_reason="completed",
        model_rounds=1,
        duration_ms=12.5,
        tool_steps=(
            AgentTraceToolStep(
                tool_name="query_health_events",
                argument_names=(
                    "event_type",
                    "user_id",
                ),
                ok=True,
                error_code=None,
            ),
        ),
        has_pending_task=False,
        pending_tool_name=None,
        has_pending_confirmation=False,
        confirmation_action=None,
        error_type=None,
    )


def test_trace_store_appends_and_reads_newest_first(
    tmp_path: Path,
) -> None:
    store = AgentTraceStore(
        tmp_path / "agent_traces.jsonl"
    )

    first = make_trace(
        action="send",
        turn_count=1,
    )
    second = make_trace(
        action="confirm",
        turn_count=2,
    )

    store.append(first)
    store.append(second)

    traces = store.read_recent(limit=20)

    assert len(traces) == 2
    assert traces[0].action == "confirm"
    assert traces[0].turn_count == 2
    assert traces[1].action == "send"
    assert traces[1].turn_count == 1


def test_trace_file_does_not_contain_raw_health_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_traces.jsonl"
    store = AgentTraceStore(path)

    trace = make_trace()
    store.append(trace)

    raw_text = path.read_text(encoding="utf-8")

    assert "记录喝水500毫升" not in raw_text
    assert "500毫升" not in raw_text
    assert "64.8公斤" not in raw_text
    assert "confirmation_token" not in raw_text
    assert "query_health_events" in raw_text


def test_read_recent_returns_empty_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    store = AgentTraceStore(
        tmp_path / "missing.jsonl"
    )

    assert store.read_recent(limit=20) == ()


def test_read_recent_rejects_non_positive_limit(
    tmp_path: Path,
) -> None:
    store = AgentTraceStore(
        tmp_path / "agent_traces.jsonl"
    )

    with pytest.raises(
        ValueError,
        match="limit 必须大于 0",
    ):
        store.read_recent(limit=0)


def test_read_recent_reports_invalid_json_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_traces.jsonl"
    path.write_text(
        '{"invalid": true}\n',
        encoding="utf-8",
    )

    store = AgentTraceStore(path)

    with pytest.raises(
        AgentTraceReadError,
        match="第 1 行格式错误",
    ):
        store.read_recent(limit=20)