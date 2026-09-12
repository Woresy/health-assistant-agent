"""LangGraph 编排器的关键健康流程验收。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

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
from src.storage.healthos_store import HealthOSStore


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


def test_relative_reminder_does_not_wait_for_model(tmp_path: Path) -> None:
    event_store = HealthEventStore(tmp_path / "health_events.jsonl")
    router = HealthToolRouter(
        event_store,
        healthos_store=HealthOSStore(tmp_path / "healthos.json"),
    )
    model = FakeAgentModel([])
    runner = LangGraphAgentRunner(model=model, router=router)
    session = ConversationSession(
        runner=runner,
        session_id="direct-reminder",
        user_id="user-1",
    )

    prepared = session.send("两分钟后提醒我喝水")

    assert prepared.model_rounds == 0
    assert prepared.pending_confirmation is not None
    assert model.received_messages == []
    confirmed = session.confirm()
    assert confirmed.answer == "提醒已确认安排。"


def test_daily_reminder_with_time_and_content_does_not_ask_for_time(
    tmp_path: Path,
) -> None:
    """完整的日常提醒描述应直接生成草稿，不能再次追问时间。"""

    event_store = HealthEventStore(tmp_path / "health_events.jsonl")
    router = HealthToolRouter(
        event_store,
        healthos_store=HealthOSStore(tmp_path / "healthos.json"),
    )
    model = FakeAgentModel([])
    runner = LangGraphAgentRunner(model=model, router=router)
    session = ConversationSession(
        runner=runner,
        session_id="direct-daily-reminder",
        user_id="user-1",
        timezone_name="Asia/Shanghai",
    )

    prepared = session.send("每天晚上9点提醒我拉伸")

    assert prepared.model_rounds == 0
    assert prepared.pending_confirmation is not None
    assert "请直接说明提醒时间" not in prepared.answer
    preview = prepared.pending_confirmation.draft_data["preview"]
    assert preview["content"] == "拉伸"
    assert preview["recurrence"] == "daily"
    assert datetime.fromisoformat(preview["scheduled_for"]).hour == 21
    assert model.received_messages == []


def test_remind_me_prefix_relative_reminder_does_not_wait_for_model(
    tmp_path: Path,
) -> None:
    """“提醒我 2 分钟后”语序也必须走确定性的未来时间解析。"""

    event_store = HealthEventStore(tmp_path / "health_events.jsonl")
    router = HealthToolRouter(
        event_store,
        healthos_store=HealthOSStore(tmp_path / "healthos.json"),
    )
    model = FakeAgentModel([])
    runner = LangGraphAgentRunner(model=model, router=router)
    session = ConversationSession(
        runner=runner,
        session_id="direct-reminder-prefix",
        user_id="user-1",
    )

    prepared = session.send("提醒我2分钟后喝水")

    assert prepared.model_rounds == 0
    assert prepared.pending_confirmation is not None
    assert prepared.answer != "提醒时间必须晚于当前时间"
    assert model.received_messages == []


def test_relative_feishu_reminder_uses_local_current_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自然语言里的相对时间应按用户时区计算，无需 ISO 8601。"""

    monkeypatch.setenv("FEISHU_REMINDER_ENABLED", "true")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/example",
    )
    event_store = HealthEventStore(tmp_path / "health_events.jsonl")
    router = HealthToolRouter(
        event_store,
        healthos_store=HealthOSStore(tmp_path / "healthos.json"),
    )
    model = FakeAgentModel([])
    runner = LangGraphAgentRunner(model=model, router=router)
    session = ConversationSession(
        runner=runner,
        session_id="direct-feishu-relative-reminder",
        user_id="user-1",
        timezone_name="Asia/Shanghai",
    )
    before = datetime.now(ZoneInfo("Asia/Shanghai"))

    prepared = session.send(
        "我想创建一个提醒：两分钟后在飞书上提醒我去喝水"
    )

    after = datetime.now(ZoneInfo("Asia/Shanghai"))
    assert prepared.model_rounds == 0
    assert prepared.pending_confirmation is not None
    preview = prepared.pending_confirmation.draft_data["preview"]
    scheduled_for = datetime.fromisoformat(preview["scheduled_for"])
    assert before + timedelta(minutes=2, seconds=-1) <= scheduled_for
    assert scheduled_for <= after + timedelta(minutes=2, seconds=1)
    assert preview["timezone_name"] == "Asia/Shanghai"
    assert preview["delivery_channel"] == "feishu"
    assert preview["content"] == "去喝水"
    assert model.received_messages == []


def test_active_check_in_does_not_wait_for_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEISHU_REMINDER_ENABLED", "true")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/example",
    )
    healthos_store = HealthOSStore(tmp_path / "healthos.json")
    router = HealthToolRouter(
        HealthEventStore(tmp_path / "health_events.jsonl"),
        healthos_store=healthos_store,
    )
    model = FakeAgentModel([])
    session = ConversationSession(
        runner=LangGraphAgentRunner(model=model, router=router),
        session_id="direct-active-check-in",
        user_id="user-1",
    )

    prepared = session.send(
        "我想每天晚上 9 点通过飞书主动 check-in，关注饮食、饮水和运动"
    )

    assert prepared.model_rounds == 0
    assert prepared.pending_confirmation is not None
    assert "目前还没有启用" in prepared.answer
    assert healthos_store.read().reminders == []
    session.confirm()
    reminder = healthos_store.read().reminders[0]
    assert reminder.reminder_type == "check_in"
    assert reminder.recurrence == "daily"
    assert reminder.check_in_focus == ["meal", "water", "exercise"]
    assert model.received_messages == []


def test_pending_draft_can_be_revised_and_confirmed_in_chat(
    tmp_path: Path,
) -> None:
    """用户不必先点取消；自然语言修订会让旧草稿失效并生成新草稿。"""

    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": "water",
                    "amount_ml": 500,
                    "occurred_at": "2026-09-12T09:00:00+08:00",
                },
                call_id="water-original",
            ),
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": "water",
                    "amount_ml": 600,
                    "occurred_at": "2026-09-12T09:00:00+08:00",
                },
                call_id="water-revised",
            ),
        ],
    )

    original = session.send("记录喝水 500 毫升")
    original_token = original.pending_confirmation.draft_data["confirmation_token"]
    revised = session.send("不是 500 毫升，改成 600 毫升")

    assert revised.state.value == "awaiting_confirmation"
    assert revised.pending_confirmation is not None
    assert revised.pending_confirmation.draft_data["confirmation_token"] != original_token
    assert store.read_all() == []

    confirmed = session.send("确认")
    assert confirmed.state.value == "completed"
    assert [event.payload.amount_ml for event in store.read_all()] == [600]


def test_text_cancel_then_new_intent_starts_cleanly(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {"event_type": "weight", "weight_kg": 70},
                call_id="weight-cancelled",
            ),
            tool_reply(
                "prepare_health_event",
                {"event_type": "water", "amount_ml": 350},
                call_id="water-after-cancel",
            ),
        ],
    )

    assert session.send("记录体重 70 公斤").pending_confirmation is not None
    assert session.send("算了").state.value == "cancelled"
    assert store.read_all() == []

    next_draft = session.send("那就记录喝水 350 毫升")
    assert next_draft.pending_confirmation is not None
    session.confirm()
    assert [event.event_type.value for event in store.read_all()] == ["water"]


def test_clarification_can_span_three_user_turns(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {"event_type": "exercise", "activity_type": "跑步"},
                call_id="exercise-start",
            ),
            tool_reply(
                "prepare_health_event",
                {"distance_km": 5},
                call_id="exercise-distance",
            ),
            tool_reply(
                "prepare_health_event",
                {"duration_minutes": 32},
                call_id="exercise-duration",
            ),
        ],
    )

    first = session.send("我跑步了")
    assert first.pending_task is not None
    second = session.send("大概 5 公里")
    assert second.pending_task is not None
    assert second.pending_task.arguments["distance_km"] == 5
    third = session.send("用了 32 分钟")
    assert third.pending_confirmation is not None
    assert third.pending_confirmation.draft_data["event"]["payload"]["distance_km"] == 5
    assert store.read_all() == []
    session.confirm()
    event = store.read_all()[0]
    assert event.payload.distance_km == 5
    assert event.payload.duration_minutes == 32


def test_confirmed_fact_can_be_queried_in_the_same_conversation(
    tmp_path: Path,
) -> None:
    session, store, _ = build_session(
        tmp_path,
        [
            tool_reply(
                "prepare_health_event",
                {
                    "event_type": "water",
                    "amount_ml": 420,
                    "occurred_at": "2026-09-12T10:00:00+08:00",
                },
                call_id="water-save",
            ),
            tool_reply(
                "get_health_events",
                {"event_type": "water", "date": "2026-09-12"},
                call_id="water-query",
            ),
            AgentModelReply(content="今天已确认记录中有一笔 420 毫升饮水。"),
        ],
    )

    session.send("记录喝水 420 毫升")
    session.confirm()
    queried = session.send("那我今天喝了多少？")

    assert queried.state.value == "completed"
    assert "420" in queried.answer
    assert len(store.read_all()) == 1


def test_langgraph_keeps_business_scenario_for_elliptical_follow_up(
    tmp_path: Path,
) -> None:
    store = HealthEventStore(tmp_path / "scenario-events.jsonl")
    model = FakeAgentModel(
        [
            tool_reply(
                "get_period_summary",
                {"days": 7},
                call_id="scenario-review",
            ),
            AgentModelReply(content="记录已经汇总。你最想先改善哪一项？"),
            AgentModelReply(content="可以，我们先把这一项变成下周的小行动。"),
        ]
    )
    runner = LangGraphAgentRunner(
        model=model,
        router=HealthToolRouter(store),
        max_model_rounds=4,
    )
    session = ConversationSession(
        runner=runner,
        session_id="scenario-follow-up",
        user_id="user-1",
    )

    session.send("帮我复盘最近一周的健康情况")
    follow_up = session.send("那下一步呢？")

    assert "小行动" in follow_up.answer
    assert "period_review" in model.received_messages[-1][0].content


def test_langgraph_recovers_when_model_requests_multiple_tools(
    tmp_path: Path,
) -> None:
    """餐食建议可能同时需要当日记录和健康知识。

    Provider 一次返回多个工具请求时，编排器应在内部让模型
    按顺序重试，不能把协议错误暴露给用户。
    """
    parallel_reply = AgentModelReply(
        tool_calls=(
            ModelToolCall(
                call_id="meal-daily-parallel",
                name="get_daily_summary",
                arguments={"date": "2026-09-12"},
            ),
            ModelToolCall(
                call_id="meal-knowledge-parallel",
                name="retrieve_health_knowledge",
                arguments={"question": "午餐怎么搭配更均衡"},
            ),
        )
    )
    session, _, runner = build_session(
        tmp_path,
        [
            parallel_reply,
            tool_reply(
                "get_daily_summary",
                {"date": "2026-09-12"},
                call_id="meal-daily-sequential",
            ),
            tool_reply(
                "retrieve_health_knowledge",
                {"question": "午餐怎么搭配更均衡"},
                call_id="meal-knowledge-sequential",
            ),
            AgentModelReply(
                content="午餐可以在主食外加一份蛋白质和一份蔬菜。"
            ),
        ],
    )

    result = session.send("那午餐应该怎么搭配？")

    assert result.state.value == "completed"
    assert [step.tool_name for step in result.tool_steps] == [
        "get_daily_summary",
        "retrieve_health_knowledge",
    ]
    assert "一次模型响应只能" not in result.answer
    assert "蛋白质" in result.answer
    assert any(
        "请一次只选择一个工具" in message.content
        for message in runner._model.received_messages[1]  # type: ignore[attr-defined]
    )
