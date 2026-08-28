"""运行食物检索评测并写入报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.nutrition.evaluation import (  # noqa: E402
    EvaluationError,
    evaluate_retrieval,
    load_evaluation_cases,
)
from src.nutrition.repository import (  # noqa: E402
    FoodRepository,
    NutritionDataError,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行食物检索固定评测"
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "eval"
            / "nutrition_retrieval.jsonl"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "foods_sample.json"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "index"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "lexical",
            "hybrid",
        ],
        default="lexical",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "docs"
            / "eval_report.json"
        ),
    )

    args = parser.parse_args()

    try:
        cases = load_evaluation_cases(
            args.cases
        )
        repository = FoodRepository(
            data_path=args.data,
            rag_mode=args.mode,
            index_dir=args.index_dir,
        )
        report = evaluate_retrieval(
            repository,
            cases,
        )
    except (
        EvaluationError,
        NutritionDataError,
    ) as exc:
        print(
            (
                f"错误 [{exc.error_code}]："
                f"{exc.message}"
            ),
            file=sys.stderr,
        )
        return 2

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    return (
        0
        if report["passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())