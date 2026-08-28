"""HealthEventStore 查询、更新、删除和原子性测试。"""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
from uuid import (
    UUID,
    uuid4,
)

import pytest

from src.health.models import (
    ExercisePayload,
    HealthEvent,
    WaterPayload,
)
from src.storage.jsonl_store import (
    HealthEventConflictError,
    HealthEventNotFoundError,
    HealthEventStore,
    JsonlWriteError,
)


BASE_TIME = datetime(
    2026,
    8,
    28,
    8,
    0,
    tzinfo=timezone.utc,
)


def _event(
    *,
    event_type: str,
    occurred_at: datetime,
    user_id: str = "user-1",
    event_id: UUID | None = None,
) -> HealthEvent:
    """生成测试事件。"""

    if event_type == "water":
        payload = WaterPayload(
            amount_ml=300,
        )
    elif event_type == "exercise":
        payload = ExercisePayload(
            activity_type="快走",
            duration_minutes=20,
            distance_km=2,
            intensity="medium",
        )
    else:
        raise ValueError(
            "测试只支持 water/exercise"
        )

    return HealthEvent.model_validate(
        {
            "schema_version": "1.1",
            "event_id": str(
                event_id or uuid4()
            ),
            "user_id": user_id,
            "event_type": event_type,
            "occurred_at": occurred_at,
            "payload": (
                payload.model_dump(
                    mode="json"
                )
            ),
            "source_refs": [],
            "input_source": "chat",
            "created_at": occurred_at,
            "updated_at": occurred_at,
        }
    )


def test_query_filters_and_sorts_events(
    tmp_path: Path,
) -> None:
    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    later_water = _event(
        event_type="water",
        occurred_at=(
            BASE_TIME
            + timedelta(hours=2)
        ),
    )
    earlier_exercise = _event(
        event_type="exercise",
        occurred_at=(
            BASE_TIME
            + timedelta(hours=1)
        ),
    )
    another_user = _event(
        event_type="water",
        occurred_at=(
            BASE_TIME
            + timedelta(hours=3)
        ),
        user_id="user-2",
    )
    next_day = _event(
        event_type="water",
        occurred_at=(
            BASE_TIME
            + timedelta(days=1)
        ),
    )

    store.append(later_water)
    store.append(another_user)
    store.append(next_day)
    store.append(earlier_exercise)

    events = store.query(
        user_id="user-1",
        occurred_from=BASE_TIME,
        occurred_to=(
            BASE_TIME
            + timedelta(days=1)
        ),
    )

    assert [
        event.event_id
        for event in events
    ] == [
        earlier_exercise.event_id,
        later_water.event_id,
    ]

    water_events = store.query(
        user_id="user-1",
        event_type="water",
        occurred_from=BASE_TIME,
        occurred_to=(
            BASE_TIME
            + timedelta(days=1)
        ),
    )

    assert [
        event.event_id
        for event in water_events
    ] == [
        later_water.event_id
    ]

    newest_first = store.query(
        user_id="user-1",
        occurred_from=BASE_TIME,
        occurred_to=(
            BASE_TIME
            + timedelta(days=1)
        ),
        newest_first=True,
    )

    assert [
        event.event_id
        for event in newest_first
    ] == [
        later_water.event_id,
        earlier_exercise.event_id,
    ]


def test_update_rewrites_exactly_one_event(
    tmp_path: Path,
) -> None:
    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    original = _event(
        event_type="water",
        occurred_at=BASE_TIME,
    )
    untouched = _event(
        event_type="exercise",
        occurred_at=(
            BASE_TIME
            + timedelta(hours=1)
        ),
    )

    store.append(original)
    store.append(untouched)

    replacement = (
        original.model_copy(
            update={
                "payload": (
                    WaterPayload(
                        amount_ml=600,
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

    updated = store.update(
        event_id=original.event_id,
        replacement=replacement,
        expected_updated_at=(
            original.updated_at
        ),
    )

    assert isinstance(
        updated.payload,
        WaterPayload,
    )
    assert (
        updated.payload.amount_ml
        == 600
    )

    persisted = store.read_all()

    assert len(persisted) == 2
    assert (
        persisted[0].event_id
        == original.event_id
    )
    assert isinstance(
        persisted[0].payload,
        WaterPayload,
    )
    assert (
        persisted[0]
        .payload
        .amount_ml
        == 600
    )
    assert (
        persisted[1]
        == untouched
    )


def test_stale_update_is_rejected_without_writing(
    tmp_path: Path,
) -> None:
    store_path = (
        tmp_path
        / "health_events.jsonl"
    )
    store = HealthEventStore(
        store_path
    )

    original = _event(
        event_type="water",
        occurred_at=BASE_TIME,
    )

    store.append(original)

    first_replacement = (
        original.model_copy(
            update={
                "payload": (
                    WaterPayload(
                        amount_ml=500,
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

    current = store.update(
        event_id=original.event_id,
        replacement=(
            first_replacement
        ),
        expected_updated_at=(
            original.updated_at
        ),
    )

    file_before_conflict = (
        store_path.read_text(
            encoding="utf-8"
        )
    )

    stale_replacement = (
        current.model_copy(
            update={
                "payload": (
                    WaterPayload(
                        amount_ml=700,
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

    with pytest.raises(
        HealthEventConflictError,
    ):
        store.update(
            event_id=original.event_id,
            replacement=(
                stale_replacement
            ),
            expected_updated_at=(
                original.updated_at
            ),
        )

    assert (
        store_path.read_text(
            encoding="utf-8"
        )
        == file_before_conflict
    )


def test_delete_removes_only_target_event(
    tmp_path: Path,
) -> None:
    store = HealthEventStore(
        tmp_path
        / "health_events.jsonl"
    )

    target = _event(
        event_type="water",
        occurred_at=BASE_TIME,
    )
    remaining = _event(
        event_type="exercise",
        occurred_at=(
            BASE_TIME
            + timedelta(hours=1)
        ),
    )

    store.append(target)
    store.append(remaining)

    deleted = store.delete(
        event_id=target.event_id,
        expected_updated_at=(
            target.updated_at
        ),
    )

    assert deleted == target
    assert store.read_all() == [
        remaining
    ]

    with pytest.raises(
        HealthEventNotFoundError,
    ):
        store.delete(
            event_id=target.event_id,
            expected_updated_at=(
                target.updated_at
            ),
        )

    assert store.read_all() == [
        remaining
    ]


def test_atomic_rewrite_failure_preserves_original_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = (
        tmp_path
        / "health_events.jsonl"
    )
    store = HealthEventStore(
        store_path
    )

    original = _event(
        event_type="water",
        occurred_at=BASE_TIME,
    )

    store.append(original)

    original_file = (
        store_path.read_text(
            encoding="utf-8"
        )
    )

    replacement = (
        original.model_copy(
            update={
                "payload": (
                    WaterPayload(
                        amount_ml=900,
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

    def fail_replace(
        source: object,
        destination: object,
    ) -> None:
        del source
        del destination
        raise OSError(
            "simulated replace failure"
        )

    monkeypatch.setattr(
        "src.storage.jsonl_store."
        "os.replace",
        fail_replace,
    )

    with pytest.raises(
        JsonlWriteError,
        match=(
            "simulated replace failure"
        ),
    ):
        store.update(
            event_id=original.event_id,
            replacement=replacement,
            expected_updated_at=(
                original.updated_at
            ),
        )

    assert (
        store_path.read_text(
            encoding="utf-8"
        )
        == original_file
    )

    temporary_files = list(
        tmp_path.glob(
            ".health_events.jsonl.*.tmp"
        )
    )

    assert temporary_files == []