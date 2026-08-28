"""健康时间线查询和每日确定性汇总测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

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
from src.tools.get_daily_health_summary import (
    get_daily_health_summary,
)
from src.tools.query_health_events import (
    query_health_events,
)


LOCAL_TIMEZONE = ZoneInfo(
    "Asia/Shanghai"
)


def _local_time(
    hour: int,
    *,
    day: int = 28,
) -> datetime:
    return datetime(
        2026,
        8,
        day,
        hour,
        0,
        tzinfo=LOCAL_TIMEZONE,
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
                "番茄"
            ),
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


def _event(
    *,
    user_id: str,
    event_type: str,
    payload: object,
    occurred_at: datetime,
) -> HealthEvent:
    source_refs = (
        ["sample:FOOD_001"]
        if event_type == "meal"
        else []
    )

    input_source = (
        "image"
        if event_type == "meal"
        else "chat"
    )

    payload_data = (
        payload.model_dump(
            mode="json"
        )
        if hasattr(
            payload,
            "model_dump",
        )
        else payload
    )

    return HealthEvent.model_validate(
        {
            "schema_version": "1.1",
            "event_id": str(
                uuid4()
            ),
            "user_id": user_id,
            "event_type": event_type,
            "occurred_at": (
                occurred_at
            ),
            "payload": payload_data,
            "source_refs": (
                source_refs
            ),
            "input_source": (
                input_source
            ),
            "created_at": occurred_at,
            "updated_at": occurred_at,
        }
    )


def _prepare_store(
    tmp_path: Path,
) -> HealthEventStore:
    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    events = [
        _event(
            user_id="user-1",
            event_type="weight",
            payload=WeightPayload(
                weight_kg=65.4,
            ),
            occurred_at=_local_time(7),
        ),
        _event(
            user_id="user-1",
            event_type="meal",
            payload=_meal_payload(),
            occurred_at=_local_time(8),
        ),
        _event(
            user_id="user-1",
            event_type="water",
            payload=WaterPayload(
                amount_ml=300,
            ),
            occurred_at=_local_time(9),
        ),
        _event(
            user_id="user-1",
            event_type="water",
            payload=WaterPayload(
                amount_ml=500,
            ),
            occurred_at=_local_time(10),
        ),
        _event(
            user_id="user-1",
            event_type="exercise",
            payload=ExercisePayload(
                activity_type="跑步",
                duration_minutes=30,
                distance_km=5,
                intensity="medium",
            ),
            occurred_at=_local_time(12),
        ),
        _event(
            user_id="user-1",
            event_type="weight",
            payload=WeightPayload(
                weight_kg=65.1,
            ),
            occurred_at=_local_time(18),
        ),
        _event(
            user_id="user-1",
            event_type="water",
            payload=WaterPayload(
                amount_ml=1000,
            ),
            occurred_at=_local_time(
                8,
                day=29,
            ),
        ),
        _event(
            user_id="user-2",
            event_type="water",
            payload=WaterPayload(
                amount_ml=2000,
            ),
            occurred_at=_local_time(11),
        ),
    ]

    for event in events:
        store.append(event)

    return store


def test_daily_summary_uses_only_saved_events(
    tmp_path: Path,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = (
        get_daily_health_summary(
            user_id="user-1",
            date="2026-08-28",
            timezone_name=(
                "Asia/Shanghai"
            ),
            store=store,
        )
    )

    assert result["ok"] is True
    assert result["error"] is None

    data = result["data"]
    summary = data["summary"]

    assert len(
        data["events"]
    ) == 6
    assert (
        summary["event_count"]
        == 6
    )

    assert (
        summary["meal"]["count"]
        == 1
    )
    assert (
        summary["meal"][
            "calories_kcal"
        ]
        == pytest.approx(30)
    )
    assert (
        summary["meal"]["protein_g"]
        == pytest.approx(1.8)
    )
    assert (
        summary["meal"]["fat_g"]
        == pytest.approx(0.4)
    )
    assert (
        summary["meal"]["carbs_g"]
        == pytest.approx(6.6)
    )

    assert (
        summary["water"]["count"]
        == 2
    )
    assert (
        summary["water"]["total_ml"]
        == pytest.approx(800)
    )

    assert (
        summary["weight"]["count"]
        == 2
    )
    assert (
        summary["weight"][
            "latest_weight_kg"
        ]
        == pytest.approx(65.1)
    )
    assert (
        summary["weight"][
            "latest_occurred_at"
        ].startswith(
            "2026-08-28T18:00:00"
        )
    )

    assert (
        summary["exercise"]["count"]
        == 1
    )
    assert (
        summary["exercise"][
            "total_duration_minutes"
        ]
        == pytest.approx(30)
    )
    assert (
        summary["exercise"][
            "total_distance_km"
        ]
        == pytest.approx(5)
    )


def test_timeline_can_filter_water_events(
    tmp_path: Path,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = query_health_events(
        user_id="user-1",
        event_type="water",
        date="2026-08-28",
        timezone_name=(
            "Asia/Shanghai"
        ),
        newest_first=False,
        limit=100,
        store=store,
    )

    assert result["ok"] is True

    data = result["data"]

    assert (
        data["matched_count"]
        == 2
    )
    assert (
        data["returned_count"]
        == 2
    )
    assert [
        event["payload"][
            "amount_ml"
        ]
        for event in data["events"]
    ] == [
        300,
        500,
    ]


def test_timeline_can_sort_newest_first(
    tmp_path: Path,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = query_health_events(
        user_id="user-1",
        date="2026-08-28",
        timezone_name=(
            "Asia/Shanghai"
        ),
        newest_first=True,
        limit=2,
        store=store,
    )

    assert result["ok"] is True

    data = result["data"]

    assert (
        data["matched_count"]
        == 6
    )
    assert (
        data["returned_count"]
        == 2
    )

    assert (
        data["events"][0][
            "event_type"
        ]
        == "weight"
    )
    assert (
        data["events"][1][
            "event_type"
        ]
        == "exercise"
    )


def test_empty_day_returns_zero_summary(
    tmp_path: Path,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = (
        get_daily_health_summary(
            user_id="user-1",
            date="2026-08-30",
            timezone_name=(
                "Asia/Shanghai"
            ),
            store=store,
        )
    )

    assert result["ok"] is True

    data = result["data"]
    summary = data["summary"]

    assert data["events"] == []
    assert (
        summary["event_count"]
        == 0
    )
    assert (
        summary["meal"][
            "calories_kcal"
        ]
        == 0
    )
    assert (
        summary["water"]["total_ml"]
        == 0
    )
    assert (
        summary["exercise"][
            "total_duration_minutes"
        ]
        == 0
    )
    assert (
        summary["weight"][
            "latest_weight_kg"
        ]
        is None
    )


@pytest.mark.parametrize(
    (
        "date_value",
        "expected_error_code",
    ),
    [
        (
            "2026-13-99",
            "DATE_INVALID",
        ),
        (
            "",
            "DATE_INVALID",
        ),
    ],
)
def test_invalid_date_returns_stable_error(
    tmp_path: Path,
    date_value: str,
    expected_error_code: str,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = (
        get_daily_health_summary(
            user_id="user-1",
            date=date_value,
            timezone_name=(
                "Asia/Shanghai"
            ),
            store=store,
        )
    )

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == expected_error_code
    )


def test_invalid_timezone_returns_stable_error(
    tmp_path: Path,
) -> None:
    store = _prepare_store(
        tmp_path
    )

    result = (
        get_daily_health_summary(
            user_id="user-1",
            date="2026-08-28",
            timezone_name=(
                "Invalid/Timezone"
            ),
            store=store,
        )
    )

    assert result["ok"] is False
    assert (
        result["error"][
            "error_code"
        ]
        == "TIMEZONE_INVALID"
    )