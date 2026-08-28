"""健康事件安全更新和删除工具测试。"""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from uuid import uuid4

import pytest

from src.health.models import (
    ExercisePayload,
    HealthEvent,
    MealFood,
    MealNutrition,
    MealPayload,
    MealPortion,
    WaterPayload,
    WeightPayload,
)
from src.storage.jsonl_store import (
    HealthEventStore,
)
from src.tools.delete_health_event import (
    delete_health_event,
)
from src.tools.prepare_health_event_mutation import (
    prepare_delete_health_event,
    prepare_update_health_event,
)
from src.tools.update_health_event import (
    update_health_event,
)


BASE_TIME = datetime(
    2026,
    8,
    28,
    8,
    0,
    tzinfo=timezone.utc,
)


def _meal_payload(
    *,
    grams: float = 200,
    calories_kcal: float = 30,
    protein_g: float = 1.8,
    fat_g: float = 0.4,
    carbs_g: float = 6.6,
) -> MealPayload:
    """生成已经确定性计算过的饮食 payload。"""

    return MealPayload(
        food=MealFood(
            food_id="FOOD_001",
            name="番茄",
            category="蔬菜",
        ),
        portion=MealPortion(
            grams=grams,
        ),
        nutrition=MealNutrition(
            calories_kcal=(
                calories_kcal
            ),
            protein_g=protein_g,
            fat_g=fat_g,
            carbs_g=carbs_g,
            source_ref=(
                "sample:FOOD_001"
            ),
            retrieval_query="番茄",
            selected_food_code=(
                "FOOD_001"
            ),
            portion_assumption=(
                f"可食部分 {grams:g}g"
            ),
        ),
        retrieval_query="番茄",
        candidate_source="manual",
    )


def _event(
    *,
    event_type: str,
    user_id: str = "user-1",
) -> HealthEvent:
    """生成四类测试事件。"""

    if event_type == "meal":
        payload = _meal_payload()
        source_refs = [
            "sample:FOOD_001"
        ]
        input_source = "image"
    elif event_type == "water":
        payload = WaterPayload(
            amount_ml=300,
        )
        source_refs = []
        input_source = "chat"
    elif event_type == "weight":
        payload = WeightPayload(
            weight_kg=65.2,
        )
        source_refs = []
        input_source = "chat"
    elif event_type == "exercise":
        payload = ExercisePayload(
            activity_type="快走",
            duration_minutes=20,
            distance_km=2,
            intensity="medium",
        )
        source_refs = []
        input_source = "chat"
    else:
        raise ValueError(
            "不支持的测试事件类型"
        )

    return HealthEvent.model_validate(
        {
            "schema_version": "1.1",
            "event_id": str(
                uuid4()
            ),
            "user_id": user_id,
            "event_type": event_type,
            "occurred_at": BASE_TIME,
            "payload": (
                payload.model_dump(
                    mode="json"
                )
            ),
            "source_refs": source_refs,
            "input_source": input_source,
            "created_at": BASE_TIME,
            "updated_at": BASE_TIME,
        }
    )


def test_prepare_update_does_not_write(
    tmp_path: Path,
) -> None:
    """准备更新时 JSONL 不应发生变化。"""

    store_path = (
        tmp_path
        / "health_events.jsonl"
    )
    store = HealthEventStore(
        store_path
    )

    event = _event(
        event_type="water"
    )
    store.append(event)

    file_before = (
        store_path.read_text(
            encoding="utf-8"
        )
    )

    result = (
        prepare_update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            patch={
                "payload": {
                    "amount_ml": 600
                }
            },
            idempotency_key=(
                "update-water-001"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert result["ok"] is True
    assert result["error"] is None

    proposed = result["data"][
        "proposed_event"
    ]

    assert (
        proposed["payload"][
            "amount_ml"
        ]
        == 600
    )
    assert result["data"][
        "confirmation_token"
    ]

    assert (
        store_path.read_text(
            encoding="utf-8"
        )
        == file_before
    )


def test_update_requires_confirmation(
    tmp_path: Path,
) -> None:
    """没有确认令牌时不得更新。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="water"
    )
    store.append(event)

    proposed = event.model_copy(
        update={
            "payload": WaterPayload(
                amount_ml=600
            ),
            "updated_at": (
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        }
    )

    result = update_health_event(
        event_id=event.event_id,
        user_id="user-1",
        replacement_event_input=(
            proposed
        ),
        confirmation_token="",
        idempotency_key=(
            "update-water-002"
        ),
        store=store,
    )

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == "CONFIRMATION_REQUIRED"
    )

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        WaterPayload,
    )
    assert (
        persisted.payload.amount_ml
        == 300
    )


def test_confirmed_update_is_idempotent(
    tmp_path: Path,
) -> None:
    """同一更新重复提交只产生一个最终状态。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="water"
    )
    store.append(event)

    draft = (
        prepare_update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            patch={
                "payload": {
                    "amount_ml": 600
                }
            },
            idempotency_key=(
                "update-water-003"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert draft["ok"] is True

    draft_data = draft["data"]

    first_result = (
        update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            replacement_event_input=(
                draft_data[
                    "proposed_event"
                ]
            ),
            confirmation_token=(
                draft_data[
                    "confirmation_token"
                ]
            ),
            idempotency_key=(
                draft_data[
                    "idempotency_key"
                ]
            ),
            store=store,
        )
    )

    assert first_result["ok"] is True
    assert (
        first_result["data"][
            "idempotent"
        ]
        is False
    )

    second_result = (
        update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            replacement_event_input=(
                draft_data[
                    "proposed_event"
                ]
            ),
            confirmation_token=(
                draft_data[
                    "confirmation_token"
                ]
            ),
            idempotency_key=(
                draft_data[
                    "idempotency_key"
                ]
            ),
            store=store,
        )
    )

    assert second_result["ok"] is True
    assert (
        second_result["data"][
            "idempotent"
        ]
        is True
    )

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        WaterPayload,
    )
    assert (
        persisted.payload.amount_ml
        == 600
    )
    assert len(
        store.read_all()
    ) == 1


@pytest.mark.parametrize(
    (
        "event_type",
        "patch",
        "expected_value",
    ),
    [
        (
            "weight",
            {
                "payload": {
                    "weight_kg": 64.8,
                }
            },
            64.8,
        ),
        (
            "exercise",
            {
                "payload": {
                    "duration_minutes": 40,
                }
            },
            40,
        ),
    ],
)
def test_weight_and_exercise_updates_require_confirmation_and_are_idempotent(
    tmp_path: Path,
    event_type: str,
    patch: dict[str, object],
    expected_value: float,
) -> None:
    """
    体重和运动修改必须先生成草稿，
    确认后才更新，重复确认保持幂等。
    """

    store = HealthEventStore(
        tmp_path
        / (
            f"{event_type}_"
            "update_events.jsonl"
        )
    )

    original_event = _event(
        event_type=event_type
    )
    store.append(
        original_event
    )

    draft = (
        prepare_update_health_event(
            event_id=(
                original_event.event_id
            ),
            user_id="user-1",
            patch=patch,
            idempotency_key=(
                f"update-{event_type}-"
                "confirmation"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert draft["ok"] is True
    assert draft["error"] is None

    before_confirmation = (
        store.find_by_event_id(
            original_event.event_id
        )
    )

    assert before_confirmation is not None

    if event_type == "weight":
        assert isinstance(
            before_confirmation.payload,
            WeightPayload,
        )
        assert (
            before_confirmation
            .payload
            .weight_kg
            == pytest.approx(65.2)
        )
    else:
        assert isinstance(
            before_confirmation.payload,
            ExercisePayload,
        )
        assert (
            before_confirmation
            .payload
            .duration_minutes
            == pytest.approx(20)
        )

    first_result = (
        update_health_event(
            event_id=(
                original_event.event_id
            ),
            user_id="user-1",
            replacement_event_input=(
                draft["data"][
                    "proposed_event"
                ]
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
    )

    assert first_result["ok"] is True
    assert first_result["error"] is None
    assert (
        first_result["data"][
            "idempotent"
        ]
        is False
    )

    updated_event = (
        store.find_by_event_id(
            original_event.event_id
        )
    )

    assert updated_event is not None

    if event_type == "weight":
        assert isinstance(
            updated_event.payload,
            WeightPayload,
        )
        assert (
            updated_event
            .payload
            .weight_kg
            == pytest.approx(
                expected_value
            )
        )
    else:
        assert isinstance(
            updated_event.payload,
            ExercisePayload,
        )
        assert (
            updated_event
            .payload
            .duration_minutes
            == pytest.approx(
                expected_value
            )
        )
        assert (
            updated_event
            .payload
            .activity_type
            == "快走"
        )
        assert (
            updated_event
            .payload
            .distance_km
            == pytest.approx(2)
        )
        assert (
            updated_event
            .payload
            .intensity
            .value
            == "medium"
        )

    second_result = (
        update_health_event(
            event_id=(
                original_event.event_id
            ),
            user_id="user-1",
            replacement_event_input=(
                draft["data"][
                    "proposed_event"
                ]
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
    )

    assert second_result["ok"] is True
    assert second_result["error"] is None
    assert (
        second_result["data"][
            "idempotent"
        ]
        is True
    )
    assert len(
        store.read_all()
    ) == 1


def test_stale_update_does_not_overwrite_newer_data(
    tmp_path: Path,
) -> None:
    """草稿生成后事件变化，应返回版本冲突。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="water"
    )
    store.append(event)

    draft = (
        prepare_update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            patch={
                "payload": {
                    "amount_ml": 600
                }
            },
            idempotency_key=(
                "update-water-stale"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert draft["ok"] is True

    competing_event = (
        event.model_copy(
            update={
                "payload": (
                    WaterPayload(
                        amount_ml=700
                    )
                ),
                "updated_at": (
                    BASE_TIME
                    + timedelta(
                        minutes=2
                    )
                ),
            }
        )
    )

    store.update(
        event_id=event.event_id,
        replacement=competing_event,
        expected_updated_at=(
            event.updated_at
        ),
    )

    result = update_health_event(
        event_id=event.event_id,
        user_id="user-1",
        replacement_event_input=(
            draft["data"][
                "proposed_event"
            ]
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

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == "EVENT_VERSION_CONFLICT"
    )

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        WaterPayload,
    )
    assert (
        persisted.payload.amount_ml
        == 700
    )


def test_partial_meal_update_requires_recalculation(
    tmp_path: Path,
) -> None:
    """
    只改克重会导致营养值过期，
    因此必须提交重新计算后的完整 meal payload。
    """

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="meal"
    )
    store.append(event)

    result = (
        prepare_update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            patch={
                "payload": {
                    "portion": {
                        "grams": 150
                    }
                }
            },
            idempotency_key=(
                "update-meal-partial"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == (
            "MEAL_RECALCULATION_REQUIRED"
        )
    )

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        MealPayload,
    )
    assert (
        persisted.payload.portion.grams
        == 200
    )
    assert (
        persisted
        .payload
        .nutrition
        .calories_kcal
        == 30
    )


def test_full_recalculated_meal_update_succeeds(
    tmp_path: Path,
) -> None:
    """完整重算后的饮食 payload 可以更新。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="meal"
    )
    store.append(event)

    recalculated_payload = (
        _meal_payload(
            grams=150,
            calories_kcal=22.5,
            protein_g=1.35,
            fat_g=0.3,
            carbs_g=4.95,
        )
    )

    draft = (
        prepare_update_health_event(
            event_id=event.event_id,
            user_id="user-1",
            patch={
                "payload": (
                    recalculated_payload
                    .model_dump(
                        mode="json"
                    )
                ),
                "source_refs": [
                    "sample:FOOD_001"
                ],
            },
            idempotency_key=(
                "update-meal-full"
            ),
            store=store,
            now=(
                BASE_TIME
                + timedelta(
                    minutes=1
                )
            ),
        )
    )

    assert draft["ok"] is True

    result = update_health_event(
        event_id=event.event_id,
        user_id="user-1",
        replacement_event_input=(
            draft["data"][
                "proposed_event"
            ]
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

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        MealPayload,
    )
    assert (
        persisted.payload.portion.grams
        == 150
    )
    assert (
        persisted
        .payload
        .nutrition
        .calories_kcal
        == pytest.approx(22.5)
    )


@pytest.mark.parametrize(
    "event_type",
    [
        "meal",
        "water",
        "weight",
        "exercise",
    ],
)
def test_four_event_types_can_be_deleted_idempotently(
    tmp_path: Path,
    event_type: str,
) -> None:
    """四类事件都遵守预览、确认和幂等删除。"""

    store = HealthEventStore(
        tmp_path
        / (
            f"{event_type}_"
            "health_events.jsonl"
        )
    )

    event = _event(
        event_type=event_type
    )
    store.append(event)

    before_draft = (
        store.read_all()
    )

    draft = (
        prepare_delete_health_event(
            event_id=event.event_id,
            user_id="user-1",
            idempotency_key=(
                f"delete-{event_type}-001"
            ),
            store=store,
        )
    )

    assert draft["ok"] is True
    assert (
        store.read_all()
        == before_draft
    )

    first_result = (
        delete_health_event(
            event_id=event.event_id,
            user_id="user-1",
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
    )

    assert first_result["ok"] is True
    assert (
        first_result["data"][
            "idempotent"
        ]
        is False
    )
    assert store.read_all() == []

    second_result = (
        delete_health_event(
            event_id=event.event_id,
            user_id="user-1",
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
    )

    assert second_result["ok"] is True
    assert (
        second_result["data"][
            "idempotent"
        ]
        is True
    )
    assert store.read_all() == []


def test_delete_requires_confirmation(
    tmp_path: Path,
) -> None:
    """没有确认令牌不能删除。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="weight"
    )
    store.append(event)

    result = delete_health_event(
        event_id=event.event_id,
        user_id="user-1",
        confirmation_token="",
        idempotency_key=(
            "delete-weight-no-confirm"
        ),
        store=store,
    )

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == "CONFIRMATION_REQUIRED"
    )
    assert store.read_all() == [
        event
    ]


def test_changed_event_cannot_be_deleted_with_old_token(
    tmp_path: Path,
) -> None:
    """删除确认后目标发生变化，应要求重新确认。"""

    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    event = _event(
        event_type="weight"
    )
    store.append(event)

    draft = (
        prepare_delete_health_event(
            event_id=event.event_id,
            user_id="user-1",
            idempotency_key=(
                "delete-weight-stale"
            ),
            store=store,
        )
    )

    assert draft["ok"] is True

    changed_event = (
        event.model_copy(
            update={
                "payload": (
                    WeightPayload(
                        weight_kg=64.9
                    )
                ),
                "updated_at": (
                    BASE_TIME
                    + timedelta(
                        minutes=1
                    )
                ),
            }
        )
    )

    store.update(
        event_id=event.event_id,
        replacement=changed_event,
        expected_updated_at=(
            event.updated_at
        ),
    )

    result = delete_health_event(
        event_id=event.event_id,
        user_id="user-1",
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

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == "EVENT_VERSION_CONFLICT"
    )

    persisted = (
        store.find_by_event_id(
            event.event_id
        )
    )

    assert persisted is not None
    assert isinstance(
        persisted.payload,
        WeightPayload,
    )
    assert (
        persisted.payload.weight_kg
        == pytest.approx(64.9)
    )