"""Hybrid Retrieval 和拒答门控单元测试，不下载真实模型。"""

from __future__ import annotations

from pathlib import Path

from src.nutrition.dense_retriever import (
    DenseHit,
    DenseRetrievalError,
)
from src.nutrition.hybrid_retriever import (
    HybridFoodRetriever,
)
from src.nutrition.repository import (
    FoodRepository,
)
from src.nutrition.retrieval_document import (
    build_retrieval_document,
    load_retrieval_hints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "foods_sample.json"
)
HINTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval_hints.json"
)


class FakeDenseRetriever:
    """返回测试预先指定的 Dense 命中。"""

    def __init__(
        self,
        hits: list[DenseHit],
    ) -> None:
        self.hits = hits

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[DenseHit]:
        return self.hits[:top_k]


class FailingDenseRetriever:
    """模拟索引缺失或 Dense 检索不可用。"""

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[DenseHit]:
        raise DenseRetrievalError(
            "DENSE_INDEX_MISSING",
            "模拟索引缺失",
        )


def test_retrieval_document_contains_carrot_hints() -> None:
    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )
    hints = load_retrieval_hints(
        HINTS_PATH
    )
    carrot = repository.get_by_food_id(
        "FOOD_025"
    )

    document = build_retrieval_document(
        carrot,
        semantic_hints=(
            hints["FOOD_025"]
        ),
    )

    assert (
        "橙色根菜类蔬菜"
        in document.text
    )
    assert (
        "地下根部食用"
        in document.text
    )
    assert (
        "根茎类蔬菜"
        in document.text
    )
    assert document.semantic_hints


def test_dense_only_candidate_can_enter_top_k() -> None:
    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_009",
                score=0.88,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_028",
                score=0.81,
                rank=2,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "早餐喝的白色乳制品",
        top_k=3,
    )

    actual_ids = [
        candidate.food_id
        for candidate
        in result.candidates
    ]

    assert result.status == "ok"
    assert "FOOD_009" in actual_ids
    assert (
        "dense"
        in result.strategies_used
    )
    assert (
        "rrf"
        in result.strategies_used
    )
    assert any(
        strategy.startswith(
            "refusal_gate:pass:"
        )
        for strategy
        in result.strategies_used
    )


def test_exact_lexical_match_keeps_priority() -> None:
    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_019",
                score=0.99,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_001",
                score=0.80,
                rank=2,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "番茄",
        top_k=3,
    )

    assert result.status == "ok"
    assert (
        result.candidates[0].food_id
        == "FOOD_001"
    )
    assert (
        "refusal_gate:pass:"
        "exact_lexical"
        in result.strategies_used
    )


def test_orange_root_vegetable_prefers_carrot() -> None:
    """
    即使 Dense 把水果“橙”排第一，
    类别约束和弱单字 contains 过滤后，
    胡萝卜仍应排在水果橙前面。
    """

    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_015",
                score=0.91,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_025",
                score=0.89,
                rank=2,
            ),
            DenseHit(
                food_id="FOOD_026",
                score=0.75,
                rank=3,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "橙色的根茎类蔬菜",
        top_k=3,
    )

    assert result.status == "ok"
    assert (
        result.candidates[0].food_id
        == "FOOD_025"
    )

    orange = next(
        candidate
        for candidate
        in result.candidates
        if candidate.food_id
        == "FOOD_015"
    )

    assert orange.lexical_rank is None
    assert (
        "weak_contains_filter"
        in result.strategies_used
    )
    assert (
        "category_constraint"
        in result.strategies_used
    )
    assert (
        "refusal_gate:pass:"
        "dense_category_supported"
        in result.strategies_used
    )


def test_dense_failure_degrades_to_lexical() -> None:
    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=(
            FailingDenseRetriever()
        ),
    )

    result = hybrid.search(
        "西红柿",
        top_k=3,
    )

    assert result.status == "ok"
    assert (
        result.candidates[0].food_id
        == "FOOD_001"
    )
    assert any(
        strategy.startswith(
            "dense_unavailable:"
        )
        for strategy
        in result.strategies_used
    )
    assert (
        "refusal_gate:pass:"
        "exact_lexical"
        in result.strategies_used
    )


def test_out_of_domain_query_is_rejected() -> None:
    """高 Dense 分数也不能把普通物体强行解释为食物。"""

    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_001",
                score=0.92,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_002",
                score=0.60,
                rank=2,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "蓝色跑车",
        top_k=3,
    )

    assert (
        result.status
        == "not_found"
    )
    assert result.candidates == []
    assert (
        result.auto_select_allowed
        is False
    )
    assert (
        "refusal_gate:reject:"
        "out_of_domain"
        in result.strategies_used
    )


def test_low_confidence_dense_result_is_rejected() -> None:
    """饮食领域查询仍必须达到最低相似度。"""

    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_001",
                score=0.53,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_002",
                score=0.41,
                rank=2,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "一种没有具体描述的食物",
        top_k=3,
    )

    assert (
        result.status
        == "not_found"
    )
    assert result.candidates == []
    assert (
        "refusal_gate:reject:"
        "low_confidence"
        in result.strategies_used
    )


def test_ambiguous_dense_result_is_rejected() -> None:
    """第一名和第二名过于接近时拒绝猜测。"""

    repository = FoodRepository(
        SAMPLE_PATH,
        rag_mode="lexical",
    )

    dense = FakeDenseRetriever(
        [
            DenseHit(
                food_id="FOOD_001",
                score=0.70,
                rank=1,
            ),
            DenseHit(
                food_id="FOOD_002",
                score=0.69,
                rank=2,
            ),
        ]
    )

    hybrid = HybridFoodRetriever(
        repository=repository,
        index_dir=(
            PROJECT_ROOT
            / "data/index"
        ),
        dense_retriever=dense,
    )

    result = hybrid.search(
        "早餐吃的食物",
        top_k=3,
    )

    assert (
        result.status
        == "not_found"
    )
    assert result.candidates == []
    assert (
        "refusal_gate:reject:"
        "ambiguous_dense"
        in result.strategies_used
    )