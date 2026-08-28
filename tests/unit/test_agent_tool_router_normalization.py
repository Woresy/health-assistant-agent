"""Agent 工具参数归一化回归测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from src.agent.models import (
    PendingConfirmation,
)
from src.agent.tool_router import (
    HealthToolRouter,
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


def test_water_event_ignores_model_input_source(
    tmp_path: Path,
) -> None:
    """
    饮水来源由应用管理。

    即使模型传入 input_source=chat，
    也应该正常生成草稿并在确认后保存。
    """

    store = HealthEventStore(
        tmp_path / "health_events.jsonl"
    )
    router = HealthToolRouter(
        store=store
    )

    dispatch = router.dispatch(
        tool_name="prepare_health_event",
        arguments={
            "event_type": "water",
            "amount_ml": 500,
            "input_source": "chat",
        },
        user_id="test-user",
        timezone_name="Asia/Shanghai",
        session_id="test-session",
        call_id="test-water-call",
    )

    assert dispatch.status == "executed"
    assert dispatch.result is not None
    assert dispatch.result["ok"] is True

    draft_data = dispatch.result["data"]

    assert isinstance(
        draft_data,
        dict,
    )

    event = draft_data["event"]

    assert event["event_type"] == "water"
    assert event["payload"]["amount_ml"] == 500
    assert event["input_source"] == "chat"

    assert store.read_all() == []

    pending = PendingConfirmation(
        action="save",
        tool_name="prepare_health_event",
        draft_data=draft_data,
    )

    saved = router.confirm(
        pending
    )

    assert saved["ok"] is True

    stored_events = store.read_all()

    assert len(stored_events) == 1
    assert (
        stored_events[0]
        .event_type
        .value
        == "water"
    )
    assert (
        stored_events[0]
        .input_source
        .value
        == "chat"
    )


def test_unknown_water_field_is_still_rejected(
    tmp_path: Path,
) -> None:
    """
    只忽略应用负责的 input_source。

    其他未知参数仍然必须被严格拒绝。
    """

    store = HealthEventStore(
        tmp_path / "health_events.jsonl"
    )
    router = HealthToolRouter(
        store=store
    )

    dispatch = router.dispatch(
        tool_name="prepare_health_event",
        arguments={
            "event_type": "water",
            "amount_ml": 500,
            "unknown_field": "不应该被接受",
        },
        user_id="test-user",
        timezone_name="Asia/Shanghai",
        session_id="test-session",
        call_id="test-invalid-call",
    )

    assert dispatch.status == "invalid"
    assert dispatch.result is not None
    assert dispatch.result["ok"] is False
    assert (
        dispatch.result["error"][
            "error_code"
        ]
        == "VALIDATION_ERROR"
    )

    assert store.read_all() == []


@pytest.mark.parametrize(
    (
        "event_input",
        "flat_patch",
        "field_name",
        "original_value",
        "expected_value",
    ),
    [
        (
            {
                "event_type": "weight",
                "weight_kg": 65.2,
            },
            {
                "weight_kg": 64.8,
            },
            "weight_kg",
            65.2,
            64.8,
        ),
        (
            {
                "event_type": "exercise",
                "activity_type": "跑步",
                "duration_minutes": 30,
                "distance_km": 5,
                "intensity": "medium",
            },
            {
                "duration_minutes": 40,
            },
            "duration_minutes",
            30,
            40,
        ),
    ],
)
def test_flat_update_fields_are_nested_in_payload(
    tmp_path: Path,
    event_input: dict[str, Any],
    flat_patch: dict[str, Any],
    field_name: str,
    original_value: float,
    expected_value: float,
) -> None:
    """
    模型传入扁平修改字段时，
    路由器应将其归一化到 payload。
    """

    event_type = str(
        event_input["event_type"]
    )

    store = HealthEventStore(
        tmp_path
        / (
            f"{event_type}_"
            "events.jsonl"
        )
    )

    create_draft = (
        prepare_health_event(
            event_input=event_input,
            user_id="test-user",
            idempotency_key=(
                f"create-{event_type}"
            ),
        )
    )

    assert create_draft["ok"] is True

    create_result = save_health_event(
        event_input=(
            create_draft["data"][
                "event"
            ]
        ),
        confirmation_token=(
            create_draft["data"][
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            create_draft["data"][
                "idempotency_key"
            ]
        ),
        store=store,
    )

    assert create_result["ok"] is True

    event_id_text = str(
        create_result["data"][
            "event"
        ]["event_id"]
    )
    event_id = UUID(
        event_id_text
    )

    original_event = (
        store.find_by_event_id(
            event_id
        )
    )

    assert original_event is not None
    assert getattr(
        original_event.payload,
        field_name,
    ) == pytest.approx(
        original_value
    )

    router = HealthToolRouter(
        store=store
    )

    dispatch = router.dispatch(
        tool_name=(
            "prepare_update_health_event"
        ),
        arguments={
            "event_id": event_id_text,
            "patch": flat_patch,
        },
        user_id="test-user",
        timezone_name="Asia/Shanghai",
        session_id="test-session",
        call_id=(
            f"update-{event_type}"
        ),
    )

    assert dispatch.status == "executed"
    assert dispatch.result is not None
    assert dispatch.result["ok"] is True

    draft_data = dispatch.result[
        "data"
    ]

    proposed_event = draft_data[
        "proposed_event"
    ]

    assert (
        proposed_event[
            "payload"
        ][field_name]
        == pytest.approx(
            expected_value
        )
    )

    # 生成草稿时，原记录不能改变。
    unchanged_event = (
        store.find_by_event_id(
            event_id
        )
    )

    assert unchanged_event is not None
    assert getattr(
        unchanged_event.payload,
        field_name,
    ) == pytest.approx(
        original_value
    )

    pending = PendingConfirmation(
        action="update",
        tool_name=(
            "prepare_update_health_event"
        ),
        draft_data=draft_data,
    )

    update_result = router.confirm(
        pending
    )

    assert update_result["ok"] is True

    updated_event = (
        store.find_by_event_id(
            event_id
        )
    )

    assert updated_event is not None
    assert getattr(
        updated_event.payload,
        field_name,
    ) == pytest.approx(
        expected_value
    )
    assert len(store.read_all()) == 1


def test_duplicate_flat_and_nested_update_is_rejected(
    tmp_path: Path,
) -> None:
    """
    同一字段不能同时出现在两个层级，
    防止产生含义不明确的修改。
    """

    store = HealthEventStore(
        tmp_path / "events.jsonl"
    )
    router = HealthToolRouter(
        store=store
    )

    dispatch = router.dispatch(
        tool_name=(
            "prepare_update_health_event"
        ),
        arguments={
            "event_id": (
                "6f23793d-decd-5bdc-"
                "ba1c-2f6e85d30288"
            ),
            "patch": {
                "weight_kg": 64.8,
                "payload": {
                    "weight_kg": 63.0,
                },
            },
        },
        user_id="test-user",
        timezone_name="Asia/Shanghai",
        session_id="test-session",
        call_id="ambiguous-update",
    )

    assert dispatch.status == "invalid"
    assert dispatch.result is not None
    assert dispatch.result["ok"] is False
    assert (
        dispatch.result["error"][
            "error_code"
        ]
        == "VALIDATION_ERROR"
    )