"""健康知识 Hybrid RAG 的索引、阈值和收益证据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.knowledge.repository import HealthKnowledgeRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FixedEmbedder:
    def encode_query(self, sentences: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        del sentences, kwargs
        return np.array([1.0, 0.0], dtype=np.float32)


def test_committed_knowledge_index_matches_source() -> None:
    repository = HealthKnowledgeRepository()
    receipt = repository.retrieval_receipt()

    assert receipt["mode"] == "hybrid"
    assert receipt["vector_store"] == "local_numpy_exact_cosine"
    assert receipt["index_id"].startswith("sha256:")
    assert receipt["min_score"] == 0.44


def test_min_score_rejects_low_confidence_dense_candidate() -> None:
    low_confidence_embeddings = np.array(
        [
            [0.40, 0.0],
            [0.39, 0.0],
            [0.20, 0.0],
            [0.10, 0.0],
            [0.05, 0.0],
            [0.01, 0.0],
        ],
        dtype=np.float32,
    )
    repository = HealthKnowledgeRepository(
        embedder=FixedEmbedder(),
        embeddings=low_confidence_embeddings,
        min_score=0.50,
    )

    assert repository.search("身体怎样恢复精神？") == []


def test_authoritative_report_proves_hybrid_semantic_gain() -> None:
    report = json.loads(
        (PROJECT_ROOT / "docs" / "health_knowledge_eval_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["passed"] is True
    assert report["hybrid"]["semantic_recall_at_3"] == 1.0
    assert report["lexical"]["semantic_recall_at_3"] < 1.0
    assert report["hybrid_semantic_gain"] > 0
    assert report["safety_recall"] == 1.0
