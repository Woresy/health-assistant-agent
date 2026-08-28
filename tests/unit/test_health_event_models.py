"""四类 HealthEvent Schema 和兼容迁移测试。"""

from __future__ import annotations

import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

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


BASE_TIME = datetime(
    2026,
    8,
    28,
    8,
    0,
    tzinfo=timezone.utc,
)


def _meal_payload() -> MealPayload:
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
            retrieval_query=(
                "西红柿"
            ),
            selected_food_code=(
                "FOOD_001"
            ),
            portion_assumption=(
                "可食部分 200g"
            ),
        ),
        retrieval_query="西红柿",
        candidate_source="manual",
    )


def _event(
    *,
    event_type: str,
    payload: object,
    source_refs: (
        list[str]
        | None
    ) = None,
    input_source: str = "chat",
) -> HealthEvent:
    return HealthEvent.model_validate(
        {
            "schema_version": "1.1",
            "event_id": str(
                uuid4()
            ),
            "user_id": "test-user",
            "event_type": (
                event_type
            ),
            "occurred_at": (
                BASE_TIME
            ),
            "payload": (
                payload.model_dump()
                if hasattr(
                    payload,
                    "model_dump",
                )
                else payload
            ),
            "source_refs": (
                source_refs
                if source_refs
                is not None
                else []
            ),
            "input_source": (
                input_source
            ),
            "created_at": (
                BASE_TIME
            ),
            "updated_at": (
                BASE_TIME
            ),
        }
    )


def test_four_event_payloads_are_supported() -> None:
    meal = _event(
        event_type="meal",
        payload=_meal_payload(),
        source_refs=[
            "sample:FOOD_001"
        ],
        input_source="image",
    )
    water = _event(
        event_type="water",
        payload=WaterPayload(
            amount_ml=500,
            beverage="饮用水",
        ),
    )
    weight = _event(
        event_type="weight",
        payload=WeightPayload(
            weight_kg=65.2,
        ),
    )
    exercise = _event(
        event_type="exercise",
        payload=ExercisePayload(
            activity_type="跑步",
            duration_minutes=30,
            distance_km=5,
            intensity="medium",
        ),
    )

    assert isinstance(
        meal.payload,
        MealPayload,
    )
    assert isinstance(
        water.payload,
        WaterPayload,
    )
    assert isinstance(
        weight.payload,
        WeightPayload,
    )
    assert isinstance(
        exercise.payload,
        ExercisePayload,
    )

    assert (
        water.payload.amount_ml
        == 500
    )
    assert (
        weight.payload.weight_kg
        == 65.2
    )
    assert (
        exercise.payload.distance_km
        == 5
    )


def test_event_type_must_match_payload() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "event_type 与 payload "
            "类型不一致"
        ),
    ):
        _event(
            event_type="meal",
            payload=WaterPayload(
                amount_ml=300,
            ),
            source_refs=[
                "test-source"
            ],
            input_source="image",
        )


def test_meal_requires_source_refs() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "meal 事件必须保留"
            "营养数据来源"
        ),
    ):
        _event(
            event_type="meal",
            payload=(
                _meal_payload()
            ),
            source_refs=[],
            input_source="image",
        )


@pytest.mark.parametrize(
    ("payload", "event_type"),
    [
        (
            {
                "amount_ml": 0,
                "unit": "ml",
            },
            "water",
        ),
        (
            {
                "weight_kg": -1,
                "unit": "kg",
            },
            "weight",
        ),
        (
            {
                "activity_type": (
                    "跑步"
                ),
                "duration_minutes": 0,
            },
            "exercise",
        ),
        (
            {
                "activity_type": (
                    "跑步"
                ),
                "duration_minutes": 30,
                "distance_km": 0,
            },
            "exercise",
        ),
    ],
)
def test_invalid_numeric_ranges_are_rejected(
    payload: dict[str, object],
    event_type: str,
) -> None:
    with pytest.raises(
        ValidationError
    ):
        _event(
            event_type=event_type,
            payload=payload,
        )


def test_exercise_distance_and_intensity_are_optional() -> None:
    event = _event(
        event_type="exercise",
        payload=ExercisePayload(
            activity_type="快走",
            duration_minutes=20,
        ),
    )

    assert isinstance(
        event.payload,
        ExercisePayload,
    )
    assert (
        event.payload.distance_km
        is None
    )
    assert (
        event.payload.intensity
        is None
    )


def test_time_fields_require_timezone() -> None:
    naive_time = datetime(
        2026,
        8,
        28,
        8,
        0,
    )

    with pytest.raises(
        ValidationError,
        match="时间字段必须包含时区",
    ):
        HealthEvent.model_validate(
            {
                "event_id": str(
                    uuid4()
                ),
                "user_id": (
                    "test-user"
                ),
                "event_type": (
                    "water"
                ),
                "occurred_at": (
                    naive_time
                ),
                "payload": {
                    "amount_ml": 500,
                    "unit": "ml",
                },
                "source_refs": [],
                "input_source": (
                    "chat"
                ),
                "created_at": (
                    BASE_TIME
                ),
                "updated_at": (
                    BASE_TIME
                ),
            }
        )


def test_legacy_meal_is_migrated_to_schema_1_1() -> None:
    legacy = {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "user_id": "legacy-user",
        "event_type": "meal",
        "occurred_at": (
            BASE_TIME.isoformat()
        ),
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
            "image_manual"
        ),
        "created_at": (
            BASE_TIME.isoformat()
        ),
        "updated_at": (
            BASE_TIME.isoformat()
        ),
    }

    event = (
        HealthEvent.model_validate(
            legacy
        )
    )

    assert (
        event.schema_version
        == "1.1"
    )
    assert (
        event.input_source.value
        == "image"
    )
    assert (
        legacy["schema_version"]
        == "1.0"
    )
    assert (
        legacy["input_source"]
        == "image_manual"
    )


def test_store_reads_legacy_jsonl(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema_version": "1.0",
        "event_id": str(uuid4()),
        "user_id": "legacy-user",
        "event_type": "meal",
        "occurred_at": (
            BASE_TIME.isoformat()
        ),
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
            "image_manual"
        ),
        "created_at": (
            BASE_TIME.isoformat()
        ),
        "updated_at": (
            BASE_TIME.isoformat()
        ),
    }

    store_path = (
        tmp_path
        / "health_events.jsonl"
    )

    store_path.write_text(
        json.dumps(
            legacy,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    store = HealthEventStore(
        store_path
    )
    events = store.read_all()

    assert len(events) == 1
    assert (
        events[0].schema_version
        == "1.1"
    )
    assert (
        events[0]
        .input_source
        .value
        == "image"
    )

    raw_after_read = (
        store_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        '"schema_version": "1.0"'
        in raw_after_read
    )


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "不支持的 HealthEvent "
            "schema_version"
        ),
    ):
        HealthEvent.model_validate(
            {
                "schema_version": (
                    "9.9"
                ),
                "event_id": str(
                    uuid4()
                ),
                "user_id": (
                    "test-user"
                ),
                "event_type": (
                    "water"
                ),
                "occurred_at": (
                    BASE_TIME
                ),
                "payload": {
                    "amount_ml": 500,
                    "unit": "ml",
                },
                "source_refs": [],
                "input_source": (
                    "chat"
                ),
                "created_at": (
                    BASE_TIME
                ),
                "updated_at": (
                    BASE_TIME
                ),
            }
        )


def test_extra_payload_fields_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        _event(
            event_type="water",
            payload={
                "amount_ml": 500,
                "unit": "ml",
                "unknown_field": (
                    "不允许"
                ),
            },
        )