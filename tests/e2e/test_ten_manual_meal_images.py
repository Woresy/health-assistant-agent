"""十张固定图片的人工饮食闭环验收。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.health.models import HealthEvent
from src.nutrition.calculator import calculate_nutrition
from src.nutrition.repository import FoodRepository
from src.storage.jsonl_store import HealthEventStore
from src.tools.confirmation import issue_confirmation_token
from src.tools.save_health_event import save_health_event
from src.ui.image_input import validate_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "meals"
CASES = json.loads((FIXTURE_DIR / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["image"] for case in CASES])
def test_fixed_image_completes_manual_meal_flow(case: dict[str, object], tmp_path: Path) -> None:
    image_result = validate_image(FIXTURE_DIR / str(case["image"]))
    assert image_result.ok is True

    repository = FoodRepository(PROJECT_ROOT / "data" / "samples" / "foods_sample.json")
    search = repository.search(str(case["query"]), top_k=3)
    assert search.status == "ok"
    assert str(case["food_code"]) in [item.food_id for item in search.candidates]

    food = repository.get_by_food_id(str(case["food_code"]))
    assert food is not None
    estimate = calculate_nutrition(food, float(case["grams"]), str(case["query"]))
    now = datetime.now(timezone.utc)
    event = HealthEvent.model_validate(
        {
            "schema_version": "1.1",
            "event_id": str(uuid4()),
            "user_id": "ten-image-user",
            "event_type": "meal",
            "occurred_at": now,
            "payload": {
                "food": {"food_id": food.food_id, "name": food.name, "category": food.category},
                "portion": {"grams": float(case["grams"]), "unit": "g"},
                "nutrition": estimate.model_dump(),
                "retrieval_query": str(case["query"]),
                "candidate_source": "manual",
                "estimated": True,
            },
            "source_refs": [estimate.source_ref],
            "input_source": "image",
            "created_at": now,
            "updated_at": now,
        }
    )
    store = HealthEventStore(tmp_path / "events.jsonl")
    token = issue_confirmation_token(event)
    result = save_health_event(
        event_input=event,
        confirmation_token=token,
        idempotency_key=f"manual-{case['image']}",
        store=store,
    )

    assert result["ok"] is True
    assert len(store.read_all()) == 1
    saved = store.read_all()[0]
    assert saved.payload.candidate_source == "manual"
    assert saved.payload.nutrition.selected_food_code == case["food_code"]
    assert saved.payload.nutrition.estimated is True
