"""固定检索评测集加载与指标计算。"""

from __future__ import annotations

import json
from collections import (
    Counter,
    defaultdict,
)
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.nutrition.repository import (
    FoodRepository,
)


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
    """评测集格式错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class EvaluationCase(BaseModel):
    """一条固定检索评测用例。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: str = Field(
        min_length=1,
        max_length=64,
    )
    expected_food_codes: list[str]
    case_type: CaseType
    note: str


def load_evaluation_cases(
    path: str | Path,
) -> list[EvaluationCase]:
    """逐行加载 JSONL 评测集。"""

    source_path = Path(path)
    cases: list[EvaluationCase] = []

    try:
        with source_path.open(
            "r",
            encoding="utf-8",
        ) as source_file:
            for line_number, raw_line in enumerate(
                source_file,
                start=1,
            ):
                if not raw_line.strip():
                    raise EvaluationError(
                        "EVAL_CASE_INVALID",
                        (
                            f"评测集第 "
                            f"{line_number} 行为空"
                        ),
                    )

                try:
                    case = (
                        EvaluationCase
                        .model_validate_json(
                            raw_line
                        )
                    )
                except ValidationError as exc:
                    raise EvaluationError(
                        "EVAL_CASE_INVALID",
                        (
                            f"评测集第 "
                            f"{line_number} 行无效："
                            f"{exc}"
                        ),
                    ) from exc

                if (
                    case.case_type
                    == "not_found"
                    and case.expected_food_codes
                ):
                    raise EvaluationError(
                        "EVAL_CASE_INVALID",
                        (
                            f"{case.case_id} 是 "
                            "not_found，但期望答案非空"
                        ),
                    )

                if (
                    case.case_type
                    != "not_found"
                    and not case.expected_food_codes
                ):
                    raise EvaluationError(
                        "EVAL_CASE_UNLABELED",
                        (
                            f"{case.case_id} 尚未标注"
                            "真实 foodCode"
                        ),
                    )

                cases.append(case)

    except EvaluationError:
        raise
    except OSError as exc:
        raise EvaluationError(
            "EVAL_FILE_READ_FAILED",
            f"评测集无法读取：{exc}",
        ) from exc

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError(
            "EVAL_CASE_DUPLICATE",
            "case_id 不得重复",
        )

    if not cases:
        raise EvaluationError(
            "EVAL_CASE_EMPTY",
            "评测集不能为空",
        )

    return cases


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    return (
        round(
            numerator / denominator,
            4,
        )
        if denominator
        else 0.0
    )


def evaluate_retrieval(
    repository: FoodRepository,
    cases: list[EvaluationCase],
) -> dict[str, Any]:
    """计算 Recall@3、Top1 和拒答指标。"""

    expected_cases = [
        case
        for case in cases
        if case.expected_food_codes
    ]
    rejection_cases = [
        case
        for case in cases
        if case.case_type == "not_found"
    ]

    recall_hits = 0
    top1_hits = 0
    rejection_hits = 0
    overall_hits = 0
    degraded_case_count = 0

    group_totals: Counter[str] = Counter()
    group_passes: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for case in cases:
        result = repository.search(
            case.query,
            top_k=3,
        )

        if any(
            strategy.startswith(
                "dense_unavailable:"
            )
            for strategy
            in result.strategies_used
        ):
            degraded_case_count += 1

        actual_codes = [
            candidate.food_id
            for candidate in result.candidates
        ]
        expected = set(
            case.expected_food_codes
        )

        recall_hit = bool(
            expected.intersection(
                actual_codes[:3]
            )
        )
        top1_hit = bool(
            actual_codes
            and actual_codes[0] in expected
        )
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
            rejection_hits += int(
                rejection_hit
            )
            case_passed = rejection_hit

        group_totals[case.case_type] += 1
        group_passes[case.case_type] += int(
            case_passed
        )
        overall_hits += int(case_passed)

        if not case_passed:
            failures.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "case_type": (
                        case.case_type
                    ),
                    "expected_food_codes": (
                        case.expected_food_codes
                    ),
                    "actual_food_codes": (
                        actual_codes
                    ),
                    "status": result.status,
                    "strategies_used": (
                        result.strategies_used
                    ),
                }
            )

    by_case_type: dict[
        str,
        dict[str, Any],
    ] = defaultdict(dict)

    for case_type in sorted(
        group_totals
    ):
        total = group_totals[case_type]
        passed = group_passes[case_type]

        by_case_type[case_type] = {
            "total": total,
            "passed": passed,
            "pass_rate": _safe_rate(
                passed,
                total,
            ),
        }

    recall_at_3 = _safe_rate(
        recall_hits,
        len(expected_cases),
    )
    rejection_accuracy = _safe_rate(
        rejection_hits,
        len(rejection_cases),
    )

    hybrid_not_degraded = not (
        repository.rag_mode == "hybrid"
        and degraded_case_count > 0
    )

    return {
        "retrieval_mode": (
            repository.rag_mode
        ),
        "dataset_id": (
            repository.dataset_id
        ),
        "dataset_record_count": (
            repository.record_count
        ),
        "index_dir": str(
            repository.index_dir
        ),
        "case_count": len(cases),
        "expected_case_count": len(
            expected_cases
        ),
        "not_found_case_count": len(
            rejection_cases
        ),
        "degraded_case_count": (
            degraded_case_count
        ),
        "recall_at_3": recall_at_3,
        "top1_accuracy": _safe_rate(
            top1_hits,
            len(expected_cases),
        ),
        "rejection_accuracy": (
            rejection_accuracy
        ),
        "overall_pass_rate": _safe_rate(
            overall_hits,
            len(cases),
        ),
        "by_case_type": dict(
            by_case_type
        ),
        "failures": failures,
        "thresholds": {
            "recall_at_3": 0.85,
            "rejection_accuracy": 1.0,
            "hybrid_requires_dense": True,
        },
        "passed": (
            recall_at_3 >= 0.85
            and rejection_accuracy == 1.0
            and hybrid_not_degraded
        ),
    }