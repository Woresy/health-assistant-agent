"""阶段 D：四类草稿、pending_task 和 Agent Loop 测试。"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import pytest

from src.agent.models import (
    AgentMessage,
    AgentModelReply,
    AgentState,
    ModelToolCall,
)
from src.agent.runner import (
    AgentRunner,
    ConversationSession,
)
from src.agent.tool_router import (
    HealthToolRouter,
)
from src.health.models import (
    MealFood,
    MealNutrition,
    MealPayload,
    MealPortion,
)
from src.storage.jsonl_store import (
    HealthEventStore,
)
from src.tools.prepare_health_event import (
    prepare_health_event,
)
from src.tools.save_health_event import (
    save_health_event,
)


FIXED_TIME = datetime(
    2026,
    8,
    28,
    8,
    0,
    tzinfo=timezone.utc,
)


class FakeAgentModel:
    """按顺序返回测试预设响应。"""

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
        self.received_messages.append(
            list(messages)
        )
        self.received_tools.append(
            list(tool_definitions)
        )

        if not self._replies:
            raise AssertionError(
                "FakeAgentModel "
                "没有剩余响应"
            )

        return self._replies.pop(0)


def _meal_payload() -> MealPayload:
    """构造已经完成营养计算的饮食 payload。"""

    return MealPayload(
        food=MealFood(
            food_id="FOOD_001",
            name="番茄",
            category="蔬菜",
        ),
        portion=MealPortion(
            grams=200,
        ),
        nutrition=MealNutrition(
            calories_kcal=30,
            protein_g=1.8,
            fat_g=0.4,
            carbs_g=6.6,
            source_ref=(
                "sample:FOOD_001"
            ),
            retrieval_query="番茄",
            selected_food_code=(
                "FOOD_001"
            ),
            portion_assumption=(
                "可食部分 200g"
            ),
        ),
        retrieval_query="番茄",
        candidate_source="manual",
    )


@pytest.mark.parametrize(
    (
        "event_input",
        "event_type",
    ),
    [
        (
            {
                "event_type": "meal",
                "payload": (
                    _meal_payload()
                    .model_dump(
                        mode="json"
                    )
                ),
                "source_refs": [
                    "sample:FOOD_001"
                ],
                "input_source": (
                    "image"
                ),
            },
            "meal",
        ),
        (
            {
                "event_type": "water",
                "amount_ml": 500,
            },
            "water",
        ),
        (
            {
                "event_type": "weight",
                "weight_kg": 65.2,
            },
            "weight",
        ),
        (
            {
                "event_type": (
                    "exercise"
                ),
                "activity_type": (
                    "跑步"
                ),
                "duration_minutes": 30,
                "distance_km": 5,
                "intensity": "medium",
            },
            "exercise",
        ),
    ],
)
def test_four_event_drafts_can_be_confirmed_and_saved(
    tmp_path: Path,
    event_input: dict[str, Any],
    event_type: str,
) -> None:
    """四类草稿都遵守先预览后保存。"""

    store = HealthEventStore(
        tmp_path
        / (
            f"{event_type}_"
            "events.jsonl"
        )
    )

    draft = prepare_health_event(
        event_input=event_input,
        user_id="user-1",
        idempotency_key=(
            f"save-{event_type}-001"
        ),
        now=FIXED_TIME,
    )

    assert draft["ok"] is True
    assert store.read_all() == []

    result = save_health_event(
        event_input=(
            draft["data"]["event"]
        ),
        confirmation_token=(
            draft["data"][
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            draft["data"][
                "idempotency_key"
            ]
        ),
        store=store,
    )

    assert result["ok"] is True
    assert len(
        store.read_all()
    ) == 1
    assert (
        store.read_all()[0]
        .event_type
        .value
        == event_type
    )


def test_missing_exercise_duration_is_completed_next_turn(
    tmp_path: Path,
) -> None:
    """缺少运动时长时追问，下一轮继续同一任务。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    model = FakeAgentModel(
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id=(
                            "exercise-missing"
                        ),
                        name=(
                            "prepare_health_event"
                        ),
                        arguments={
                            "event_type": (
                                "exercise"
                            ),
                            "activity_type": (
                                "跑步"
                            ),
                        },
                    ),
                )
            ),
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id=(
                            "exercise-complete"
                        ),
                        name=(
                            "prepare_health_event"
                        ),
                        arguments={
                            "duration_minutes": 30
                        },
                    ),
                )
            ),
        ]
    )

    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(
            store
        ),
    )

    session = ConversationSession(
        runner=runner,
        session_id="session-1",
        user_id="user-1",
        timezone_name="UTC",
    )

    first = session.send(
        "记录跑步"
    )

    assert (
        first.finish_reason.value
        == "needs_clarification"
    )
    assert "多少分钟" in (
        first.answer
    )
    assert (
        session.state
        .pending_task
        is not None
    )
    assert (
        session.state
        .pending_task
        .arguments[
            "activity_type"
        ]
        == "跑步"
    )
    assert store.read_all() == []

    second = session.send(
        "30分钟"
    )

    assert (
        second.finish_reason.value
        == "awaiting_confirmation"
    )
    assert (
        second.state
        == AgentState
        .AWAITING_CONFIRMATION
    )
    assert (
        session.state
        .pending_confirmation
        is not None
    )
    assert store.read_all() == []

    confirmed = session.confirm()

    assert (
        confirmed.finish_reason.value
        == "completed"
    )
    assert len(
        store.read_all()
    ) == 1

    saved = store.read_all()[0]

    assert (
        saved.event_type.value
        == "exercise"
    )
    assert (
        saved.payload
        .duration_minutes
        == 30
    )


def test_query_tool_result_returns_to_model(
    tmp_path: Path,
) -> None:
    """只读工具执行后，结果回传给模型组织最终回答。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    draft = prepare_health_event(
        event_input={
            "event_type": "water",
            "amount_ml": 500,
            "occurred_at": (
                FIXED_TIME
                .isoformat()
            ),
        },
        user_id="user-1",
        idempotency_key=(
            "existing-water"
        ),
        now=FIXED_TIME,
    )

    assert draft["ok"] is True

    saved = save_health_event(
        event_input=(
            draft["data"]["event"]
        ),
        confirmation_token=(
            draft["data"][
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            draft["data"][
                "idempotency_key"
            ]
        ),
        store=store,
    )

    assert saved["ok"] is True

    model = FakeAgentModel(
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="query-1",
                        name=(
                            "query_health_events"
                        ),
                        arguments={
                            "event_type": (
                                "water"
                            ),
                            "date": (
                                "2026-08-28"
                            ),
                            "timezone_name": (
                                "UTC"
                            ),
                        },
                    ),
                )
            ),
            AgentModelReply(
                content=(
                    "当天有 1 条饮水记录，"
                    "饮水量为 500 毫升。"
                )
            ),
        ]
    )

    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(
            store
        ),
    )

    session = ConversationSession(
        runner=runner,
        session_id="session-query",
        user_id="user-1",
        timezone_name="UTC",
    )

    result = session.send(
        "我这天喝了多少水？"
    )

    assert (
        result.finish_reason.value
        == "completed"
    )
    assert result.model_rounds == 2
    assert len(
        result.tool_steps
    ) == 1
    assert (
        result.tool_steps[0]
        .tool_name
        == "query_health_events"
    )
    assert (
        result.tool_steps[0]
        .result["ok"]
        is True
    )
    assert "500" in result.answer

    second_model_messages = (
        model.received_messages[1]
    )

    assert any(
        message.role == "tool"
        for message
        in second_model_messages
    )


def test_unknown_tool_is_blocked(
    tmp_path: Path,
) -> None:
    """模型提出白名单外工具时不得执行。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    model = FakeAgentModel(
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="danger-1",
                        name=(
                            "delete_database"
                        ),
                        arguments={},
                    ),
                )
            )
        ]
    )

    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(
            store
        ),
    )

    session = ConversationSession(
        runner=runner,
        session_id="session-danger",
        user_id="user-1",
    )

    result = session.send(
        "删除全部数据库"
    )

    assert (
        result.finish_reason.value
        == "invalid_arguments"
    )
    assert (
        result.state
        == AgentState.FAILED
    )
    assert "白名单" in result.answer
    assert store.read_all() == []


def test_pending_confirmation_blocks_new_turn_until_cancelled(
    tmp_path: Path,
) -> None:
    """存在待确认草稿时不能偷偷开始另一个写操作。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    model = FakeAgentModel(
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="water-1",
                        name=(
                            "prepare_health_event"
                        ),
                        arguments={
                            "event_type": (
                                "water"
                            ),
                            "amount_ml": 300,
                        },
                    ),
                )
            )
        ]
    )

    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(
            store
        ),
    )

    session = ConversationSession(
        runner=runner,
        session_id="session-block",
        user_id="user-1",
    )

    first = session.send(
        "记录喝水300毫升"
    )

    assert (
        first.finish_reason.value
        == "awaiting_confirmation"
    )
    assert store.read_all() == []

    blocked = session.send(
        "再记录体重65公斤"
    )

    assert (
        blocked.finish_reason.value
        == "awaiting_confirmation"
    )
    assert "确认或取消" in (
        blocked.answer
    )
    assert store.read_all() == []

    cancelled = session.cancel()

    assert (
        cancelled.finish_reason.value
        == "cancelled"
    )
    assert (
        session.state
        .pending_confirmation
        is None
    )
    assert store.read_all() == []


def test_delete_draft_requires_user_confirmation(
    tmp_path: Path,
) -> None:
    """Agent 只能准备删除，确认后才真正删除。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    draft = prepare_health_event(
        event_input={
            "event_type": "weight",
            "weight_kg": 65.2,
        },
        user_id="user-1",
        idempotency_key=(
            "existing-weight"
        ),
        now=FIXED_TIME,
    )

    assert draft["ok"] is True

    saved = save_health_event(
        event_input=(
            draft["data"]["event"]
        ),
        confirmation_token=(
            draft["data"][
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            draft["data"][
                "idempotency_key"
            ]
        ),
        store=store,
    )

    assert saved["ok"] is True

    event_id = (
        saved["data"]["event"][
            "event_id"
        ]
    )

    model = FakeAgentModel(
        [
            AgentModelReply(
                tool_calls=(
                    ModelToolCall(
                        call_id="delete-1",
                        name=(
                            "prepare_delete_health_event"
                        ),
                        arguments={
                            "event_id": event_id
                        },
                    ),
                )
            )
        ]
    )

    runner = AgentRunner(
        model=model,
        router=HealthToolRouter(
            store
        ),
    )

    session = ConversationSession(
        runner=runner,
        session_id="session-delete",
        user_id="user-1",
    )

    prepared = session.send(
        "删除这条体重记录"
    )

    assert (
        prepared.finish_reason.value
        == "awaiting_confirmation"
    )
    assert len(
        store.read_all()
    ) == 1

    confirmed = session.confirm()

    assert (
        confirmed.finish_reason.value
        == "completed"
    )
    assert store.read_all() == []