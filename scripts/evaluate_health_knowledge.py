"""对比健康知识 Lexical 与 Hybrid RAG，并生成可提交的权威评测报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.repository import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_MIN_SCORE,
    HealthKnowledgeRepository,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evaluate_mode(
    repository: HealthKnowledgeRepository,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    positive_count = 0
    positive_hits = 0
    top1_hits = 0
    negative_count = 0
    negative_hits = 0
    semantic_count = 0
    semantic_hits = 0
    failures: list[dict[str, Any]] = []

    for case in cases:
        expected = set(case["expected_document_ids"])
        results = repository.search(str(case["question"]), top_k=3)
        actual = [item["document_id"] for item in results]
        if expected:
            positive_count += 1
            hit = bool(expected.intersection(actual))
            positive_hits += int(hit)
            top1_hits += int(bool(actual and actual[0] in expected))
            if str(case["case_id"]).startswith("semantic_"):
                semantic_count += 1
                semantic_hits += int(hit)
        else:
            negative_count += 1
            hit = not actual
            negative_hits += int(hit)
        if not hit:
            failures.append({
                "case_id": case["case_id"],
                "expected_document_ids": sorted(expected),
                "actual_document_ids": actual,
            })

    return {
        "positive_case_count": positive_count,
        "recall_at_3": round(positive_hits / positive_count, 4),
        "top1_accuracy": round(top1_hits / positive_count, 4),
        "negative_case_count": negative_count,
        "rejection_accuracy": round(negative_hits / negative_count, 4),
        "semantic_case_count": semantic_count,
        "semantic_recall_at_3": round(semantic_hits / semantic_count, 4),
        "failures": failures,
    }


def evaluate(*, cases_path: Path, safety_path: Path, index_dir: Path) -> dict[str, Any]:
    cases = _load_jsonl(cases_path)
    safety_cases = _load_jsonl(safety_path)
    lexical_repository = HealthKnowledgeRepository(index_dir=index_dir / "missing-lexical-baseline")
    hybrid_repository = HealthKnowledgeRepository(index_dir=index_dir)
    lexical = _evaluate_mode(lexical_repository, cases)
    hybrid = _evaluate_mode(hybrid_repository, cases)

    safety_hits = sum(
        HealthKnowledgeRepository.classify_safety(str(case["question"]))
        == case["expected_error_code"]
        for case in safety_cases
    )
    safety_recall = round(safety_hits / len(safety_cases), 4)
    semantic_gain = round(
        hybrid["semantic_recall_at_3"] - lexical["semantic_recall_at_3"],
        4,
    )
    passed = (
        hybrid["recall_at_3"] == 1.0
        and hybrid["top1_accuracy"] == 1.0
        and hybrid["rejection_accuracy"] == 1.0
        and hybrid["semantic_recall_at_3"] == 1.0
        and semantic_gain > 0
        and safety_recall == 1.0
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "design": {
            "pipeline": "safety_gate -> lexical+dense -> weighted_rrf -> top_k+min_score -> cited_context",
            "vector_store": "local_numpy_exact_cosine",
            "top_k": 3,
            "min_score": DEFAULT_MIN_SCORE,
            "index_id": hybrid_repository.retrieval_receipt()["index_id"],
        },
        "lexical": lexical,
        "hybrid": hybrid,
        "hybrid_semantic_gain": semantic_gain,
        "safety_case_count": len(safety_cases),
        "safety_recall": safety_recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "tests/eval/health_knowledge.jsonl")
    parser.add_argument("--safety-cases", type=Path, default=PROJECT_ROOT / "tests/eval/health_safety.jsonl")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs/health_knowledge_eval_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        cases_path=args.cases,
        safety_path=args.safety_cases,
        index_dir=args.index_dir,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
