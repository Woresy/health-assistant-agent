"""固定检索评测集加载与指标计算。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.nutrition.repository import FoodRepository


CaseType = Literal[
    "standard_name",
    "alias",
    "colloquial",
    "compound_dish",
    "similar_food",
    "typo",
    "not_found",
]


class EvaluationError(Exception):
    """评测集格式错误，包含稳定错误码。"""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class EvaluationCase(BaseModel):
    """一条固定检索评测用例。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=64)
    expected_food_codes: list[str]
    case_type: CaseType
    note: str


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """逐行加载 JSONL，坏行携带稳定错误码与行号。"""

    source_path = Path(path)
    cases: list[EvaluationCase] = []
    try:
        with source_path.open("r", encoding="utf-8") as source_file:
            for line_number, raw_line in enumerate(source_file, start=1):
                if not raw_line.strip():
                    raise EvaluationError(
                        "EVAL_CASE_INVALID",
                        f"评测集第 {line_number} 行为空",
                    )
                try:
                    case = EvaluationCase.model_validate_json(raw_line)
                except ValidationError as exc:
                    raise EvaluationError(
                        "EVAL_CASE_INVALID",
                        f"评测集第 {line_number} 行无效：{exc}",
                    ) from exc
                cases.append(case)
    except EvaluationError:
        raise
    except OSError as exc:
        raise EvaluationError(
            "EVAL_FILE_READ_FAILED",
            f"评测集无法读取：{exc}",
        ) from exc

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError("EVAL_CASE_DUPLICATE", "case_id 不得重复")
    return cases


def _safe_rate(numerator: int, denominator: int) -> float:
    """避免空分组除零，并固定保留四位小数。"""

    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate_retrieval(
    repository: FoodRepository,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    """以同一检索实现计算总体、分组指标和失败清单。"""

    expected_cases = [case for case in cases if case.expected_food_codes]
    rejection_cases = [case for case in cases if case.case_type == "not_found"]
    recall_hits = 0
    top1_hits = 0
    rejection_hits = 0
    overall_hits = 0
    group_totals: Counter[str] = Counter()
    group_passes: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for case in cases:
        result = repository.search(case.query, top_k=3)
        actual_codes = [candidate.food_id for candidate in result.candidates]
        expected = set(case.expected_food_codes)
        recall_hit = bool(expected.intersection(actual_codes[:3]))
        top1_hit = bool(actual_codes and actual_codes[0] in expected)
        rejection_hit = (
            case.case_type == "not_found"
            and result.status == "not_found"
            and not actual_codes
        )

        if case.expected_food_codes:
            recall_hits += int(recall_hit)
            top1_hits += int(top1_hit)
            case_passed = recall_hit
        else:
            rejection_hits += int(rejection_hit)
            case_passed = rejection_hit

        group_totals[case.case_type] += 1
        group_passes[case.case_type] += int(case_passed)
        overall_hits += int(case_passed)
        if not case_passed:
            failures.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "case_type": case.case_type,
                    "expected_food_codes": case.expected_food_codes,
                    "actual_food_codes": actual_codes,
                    "status": result.status,
                }
            )

    by_case_type: dict[str, dict[str, Any]] = defaultdict(dict)
    for case_type in sorted(group_totals):
        total = group_totals[case_type]
        passed = group_passes[case_type]
        by_case_type[case_type] = {
            "total": total,
            "passed": passed,
            "pass_rate": _safe_rate(passed, total),
        }

    recall_at_3 = _safe_rate(recall_hits, len(expected_cases))
    rejection_accuracy = _safe_rate(rejection_hits, len(rejection_cases))
    return {
        "dataset_id": repository.dataset_id,
        "dataset_record_count": len(repository._foods),
        "case_count": len(cases),
        "expected_case_count": len(expected_cases),
        "not_found_case_count": len(rejection_cases),
        "recall_at_3": recall_at_3,
        "top1_accuracy": _safe_rate(top1_hits, len(expected_cases)),
        "rejection_accuracy": rejection_accuracy,
        "overall_pass_rate": _safe_rate(overall_hits, len(cases)),
        "by_case_type": dict(by_case_type),
        "failures": failures,
        "thresholds": {
            "recall_at_3": 0.85,
            "rejection_accuracy": 1.0,
        },
        "passed": recall_at_3 >= 0.85 and rejection_accuracy == 1.0,
    }
