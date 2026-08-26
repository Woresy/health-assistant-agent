"""手动饮食记录纵向链路 E2E；不启动浏览器。"""

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
FOODS_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "meal.png"

EXPECTED_OUTER_FIELDS = {
    "schema_version",
    "event_id",
    "user_id",
    "event_type",
    "occurred_at",
    "payload",
    "source_refs",
    "input_source",
    "created_at",
    "updated_at",
}


def test_manual_meal_recording_vertical_slice(
    tmp_path: Path,
) -> None:
    """覆盖上传入口到 JSONL 持久化及失败分支。"""

    # 1. 固定图片是合法 PNG，测试不复制原图。
    image_result = validate_image(FIXTURE_IMAGE)
    assert image_result.ok is True
    assert image_result.image_format == "PNG"

    # 2. “西红柿”必须通过别名召回固定番茄 food_id。
    repository = FoodRepository(FOODS_PATH)
    search_result = repository.search("西红柿", top_k=3)

    assert search_result.status == "ok"
    assert search_result.candidates[0].food_id == "FOOD_001"
    assert search_result.candidates[0].name == "番茄"
    assert search_result.candidates[0].match_type == "alias"

    # 3. 200g 番茄的结果必须等于手算值。
    food = repository.get_by_food_id(
        search_result.candidates[0].food_id
    )
    estimate = calculate_nutrition(
        food=food,
        raw_grams=200,
        retrieval_query="西红柿",
    )

    assert estimate.calories_kcal == pytest.approx(30.0)
    assert estimate.protein_g == pytest.approx(1.8)
    assert estimate.fat_g == pytest.approx(0.4)
    assert estimate.carbs_g == pytest.approx(6.6)
    assert estimate.selected_food_code == "FOOD_001"
    assert estimate.retrieval_query == "西红柿"
    assert estimate.estimated is True
    assert estimate.source_ref

    # 4. 构建待确认的完整 meal HealthEvent。
    now = datetime.now(timezone.utc)
    proposed_event = HealthEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_id": str(uuid4()),
            "user_id": "e2e-user",
            "event_type": "meal",
            "occurred_at": now,
            "payload": {
                "food": {
                    "food_id": food.food_id,
                    "name": food.name,
                    "category": food.category,
                },
                "portion": {
                    "grams": 200,
                    "unit": "g",
                },
                "nutrition": estimate.model_dump(),
                "retrieval_query": "西红柿",
                "candidate_source": "manual",
                "estimated": True,
            },
            "source_refs": [estimate.source_ref],
            "input_source": "image_manual",
            "created_at": now,
            "updated_at": now,
        }
    )

    store_path = tmp_path / "health_events.jsonl"
    store = HealthEventStore(store_path)
    confirmation_token = issue_confirmation_token(proposed_event)
    idempotency_key = "e2e-manual-meal-001"

    # 5. 有效确认后恰好写入一行。
    first_result = save_health_event(
        event_input=proposed_event,
        confirmation_token=confirmation_token,
        idempotency_key=idempotency_key,
        store=store,
    )

    assert first_result["ok"] is True
    assert first_result["error"] is None
    assert first_result["data"]["idempotent"] is False

    lines = store_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    persisted_raw = json.loads(lines[0])
    assert set(persisted_raw) == EXPECTED_OUTER_FIELDS
    assert persisted_raw["event_type"] == "meal"
    assert persisted_raw["input_source"] == "image_manual"
    assert persisted_raw["payload"]["food"]["food_id"] == "FOOD_001"
    assert persisted_raw["payload"]["food"]["name"] == "番茄"
    assert persisted_raw["payload"]["portion"]["grams"] == 200
    assert persisted_raw["payload"]["portion"]["unit"] == "g"
    assert persisted_raw["payload"]["candidate_source"] == "manual"
    assert persisted_raw["payload"]["estimated"] is True
    assert persisted_raw["source_refs"] == [estimate.source_ref]

    persisted_event = HealthEvent.model_validate(persisted_raw)
    assert persisted_event.payload.nutrition.calories_kcal == pytest.approx(
        30.0
    )
    assert persisted_event.payload.nutrition.protein_g == pytest.approx(
        1.8
    )
    assert persisted_event.payload.nutrition.fat_g == pytest.approx(0.4)
    assert persisted_event.payload.nutrition.carbs_g == pytest.approx(6.6)

    # 6. 同一 idempotency_key 再次提交不新增。
    duplicate_result = save_health_event(
        event_input=proposed_event,
        confirmation_token=confirmation_token,
        idempotency_key=idempotency_key,
        store=store,
    )

    assert duplicate_result["ok"] is True
    assert duplicate_result["data"]["idempotent"] is True
    assert len(
        store_path.read_text(encoding="utf-8").splitlines()
    ) == 1

    # 7. not_found 不产生估算，也不触发写入。
    not_found_result = repository.search(
        "火星不存在食物",
        top_k=3,
    )

    assert not_found_result.status == "not_found"
    assert not_found_result.candidates == []
    assert len(
        store_path.read_text(encoding="utf-8").splitlines()
    ) == 1

    # 8. 未确认提交被稳定错误码拒绝，文件仍恰好一行。
    unconfirmed_result = save_health_event(
        event_input=proposed_event,
        confirmation_token="",
        idempotency_key="e2e-unconfirmed-001",
        store=store,
    )

    assert unconfirmed_result == {
        "ok": False,
        "data": None,
        "error": {
            "error_code": "CONFIRMATION_REQUIRED",
            "message": "必须先查看估算结果并明确确认保存",
        },
    }
    assert len(
        store_path.read_text(encoding="utf-8").splitlines()
    ) == 1