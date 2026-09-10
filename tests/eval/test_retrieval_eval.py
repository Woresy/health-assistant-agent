"""固定 20 条食物检索评测集的门槛测试。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.nutrition.evaluation import evaluate_retrieval, load_evaluation_cases
from src.nutrition.repository import FoodRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "tests" / "eval" / "nutrition_retrieval.jsonl"
SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"
CANONICAL_REPORT_PATH = PROJECT_ROOT / "docs" / "eval_report.json"


def test_fixed_evaluation_set_shape_and_thresholds() -> None:
    """评测集覆盖数量、真实示例码和门槛必须同时满足。"""

    cases = load_evaluation_cases(CASES_PATH)
    counts = Counter(case.case_type for case in cases)
    sample_codes = {
        food.food_id
        for food in FoodRepository(SAMPLE_PATH)._foods
    }

    assert len(cases) == 20
    assert counts["standard_name"] >= 2
    assert counts["alias"] >= 4
    assert counts["colloquial"] >= 3
    assert counts["compound_dish"] >= 3
    assert counts["similar_food"] >= 4
    assert counts["typo"] >= 3
    assert counts["not_found"] >= 1

    for case in cases:
        if case.case_type == "not_found":
            assert case.expected_food_codes == []
        else:
            assert case.expected_food_codes
            assert set(case.expected_food_codes) <= sample_codes

    report = evaluate_retrieval(FoodRepository(SAMPLE_PATH), cases)
    assert report["recall_at_3"] >= 0.85
    assert report["rejection_accuracy"] == 1.0
    assert report["passed"] is True


def test_only_canonical_portable_rag_report_is_committed() -> None:
    """权威报告必须唯一且不泄漏开发机绝对路径。"""

    report = json.loads(CANONICAL_REPORT_PATH.read_text(encoding="utf-8"))
    report_text = CANONICAL_REPORT_PATH.read_text(encoding="utf-8")

    assert report["passed"] is True
    assert report["index_dir"] == "data/index"
    assert "/home/" not in report_text
    assert "\\\\Users\\" not in report_text
    assert not (PROJECT_ROOT / "docs" / "eval_hybrid.json").exists()
    assert not (PROJECT_ROOT / "docs" / "eval_lexical.json").exists()
