"""词法召回、Dense 召回、RRF 融合、规则重排和拒答门控。"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Protocol

from src.nutrition.dense_retriever import (
    DenseFoodRetriever,
    DenseHit,
    DenseRetrievalError,
)
from src.nutrition.repository import (
    FoodCandidate,
    FoodRepository,
    SearchResult,
)
from src.nutrition.retrieval_gate import (
    evaluate_retrieval_gate,
)
from src.nutrition.text_normalize import (
    normalize_text,
)


class DenseRetrieverProtocol(Protocol):
    """便于测试时替换真实 Dense 模型。"""

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[DenseHit]:
        ...


class HybridFoodRetriever:
    """将词法和 Dense 结果用 RRF 融合，并执行拒答门控。"""

    def __init__(
        self,
        repository: FoodRepository,
        index_dir: str | Path,
        *,
        dense_retriever: (
            DenseRetrieverProtocol | None
        ) = None,
        pool_size: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.repository = repository
        self.pool_size = pool_size
        self.rrf_k = rrf_k

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseFoodRetriever(
                repository=repository,
                index_dir=index_dir,
            )
        )

    @staticmethod
    def _is_weak_contains_match(
        candidate: FoodCandidate,
    ) -> bool:
        """
        一个字的食物名不能因为出现在长句中，
        获得强 contains 词法召回。

        例如“橙色的根茎类蔬菜”不能因为含“橙”，
        就把食物“橙”作为强词法命中。
        """

        return (
            candidate.match_type == "contains"
            and len(
                normalize_text(
                    candidate.matched_term
                )
            )
            < 2
        )

    @classmethod
    def _clean_lexical_candidates(
        cls,
        candidates: list[FoodCandidate],
    ) -> list[FoodCandidate]:
        """移除弱单字包含命中，并重新计算词法排名。"""

        cleaned = [
            candidate.model_copy(deep=True)
            for candidate in candidates
            if not cls._is_weak_contains_match(
                candidate
            )
        ]

        for rank, candidate in enumerate(
            cleaned,
            start=1,
        ):
            candidate.lexical_rank = rank

        return cleaned

    @staticmethod
    def _category_priority(
        candidate: FoodCandidate,
        normalized_query: str,
    ) -> int:
        """
        查询明确出现类别时优先同类别候选。

        返回值越小，优先级越高。
        """

        category = normalize_text(
            candidate.category
        )

        category_requirements = [
            ("蔬菜", "蔬菜"),
            ("根茎", "蔬菜"),
            ("水果", "水果"),
            ("乳制品", "乳"),
            ("奶制品", "乳"),
            ("肉类", "肉"),
            ("鱼类", "鱼"),
            ("海鲜", "鱼虾蟹贝"),
            ("主食", "谷薯"),
            ("豆制品", "豆"),
            ("蛋类", "蛋"),
        ]

        requested_categories = [
            category_token
            for query_token, category_token
            in category_requirements
            if query_token in normalized_query
        ]

        if not requested_categories:
            return 0

        if any(
            requested in category
            for requested
            in requested_categories
        ):
            return 0

        return 1

    def _normalized_rrf_score(
        self,
        lexical_rank: int | None,
        dense_rank: int | None,
    ) -> float:
        """将两个检索通道的 RRF 分数归一化到 0—1。"""

        raw_score = 0.0

        if lexical_rank is not None:
            raw_score += 1.0 / (
                self.rrf_k + lexical_rank
            )

        if dense_rank is not None:
            raw_score += 1.0 / (
                self.rrf_k + dense_rank
            )

        maximum_score = (
            2.0 / (self.rrf_k + 1)
        )

        return round(
            raw_score / maximum_score,
            6,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResult:
        """
        执行 Hybrid Retrieval。

        Dense 不可用时降级到词法检索；
        所有候选在返回前都要经过拒答门控。
        """

        started_at = perf_counter()

        lexical_result = (
            self.repository.search_lexical(
                query,
                top_k=self.pool_size,
            )
        )
        normalized_query = (
            lexical_result.normalized_query
        )

        cleaned_lexical_candidates = (
            self._clean_lexical_candidates(
                lexical_result.candidates
            )
        )

        try:
            dense_hits = (
                self.dense_retriever.search(
                    query,
                    top_k=self.pool_size,
                )
            )
        except DenseRetrievalError as exc:
            gate_candidates = (
                cleaned_lexical_candidates[
                    :max(top_k, 2)
                ]
            )

            gate_decision = (
                evaluate_retrieval_gate(
                    gate_candidates,
                    normalized_query,
                    dense_available=False,
                )
            )

            fallback_candidates = (
                cleaned_lexical_candidates[
                    :top_k
                ]
                if gate_decision.allowed
                else []
            )

            return SearchResult(
                status=(
                    "ok"
                    if fallback_candidates
                    else "not_found"
                ),
                query=lexical_result.query,
                normalized_query=(
                    normalized_query
                ),
                top_k=top_k,
                candidates=(
                    fallback_candidates
                ),
                auto_select_allowed=False,
                selection_mode=(
                    "user_required"
                ),
                strategies_used=[
                    *lexical_result
                    .strategies_used,
                    "weak_contains_filter",
                    (
                        "dense_unavailable:"
                        f"{exc.error_code}"
                    ),
                    "refusal_gate",
                    gate_decision.strategy_tag,
                ],
                dataset_id=(
                    lexical_result.dataset_id
                ),
                dataset_record_count=(
                    lexical_result
                    .dataset_record_count
                ),
                elapsed_ms=round(
                    (
                        perf_counter()
                        - started_at
                    )
                    * 1000,
                    3,
                ),
            )

        lexical_by_id = {
            candidate.food_id: candidate
            for candidate
            in cleaned_lexical_candidates
        }
        dense_by_id = {
            hit.food_id: hit
            for hit in dense_hits
        }

        all_food_ids = (
            set(lexical_by_id)
            | set(dense_by_id)
        )

        fused_candidates: list[
            FoodCandidate
        ] = []

        for food_id in all_food_ids:
            lexical_candidate = (
                lexical_by_id.get(food_id)
            )
            dense_hit = dense_by_id.get(
                food_id
            )

            lexical_rank = (
                lexical_candidate.lexical_rank
                if lexical_candidate is not None
                else None
            )
            dense_rank = (
                dense_hit.rank
                if dense_hit is not None
                else None
            )

            rrf_score = (
                self._normalized_rrf_score(
                    lexical_rank,
                    dense_rank,
                )
            )

            if lexical_candidate is not None:
                data = (
                    lexical_candidate.model_dump()
                )
                data.update(
                    {
                        "score": rrf_score,
                        "match_type": (
                            "hybrid"
                            if dense_hit
                            is not None
                            else (
                                lexical_candidate
                                .match_type
                            )
                        ),
                        "lexical_match_type": (
                            lexical_candidate
                            .lexical_match_type
                        ),
                        "lexical_rank": (
                            lexical_rank
                        ),
                        "dense_rank": dense_rank,
                        "lexical_score": (
                            lexical_candidate
                            .lexical_score
                        ),
                        "dense_score": (
                            dense_hit.score
                            if dense_hit
                            is not None
                            else None
                        ),
                        "rrf_score": rrf_score,
                    }
                )

                candidate = (
                    FoodCandidate.model_validate(
                        data
                    )
                )
            else:
                assert dense_hit is not None

                food = (
                    self.repository
                    .get_by_food_id(food_id)
                )

                candidate = FoodCandidate(
                    food_id=food.food_id,
                    name=food.name,
                    category=food.category,
                    score=rrf_score,
                    stage=4,
                    match_type="dense",
                    lexical_match_type=None,
                    matched_term=food.name,
                    lexical_rank=None,
                    dense_rank=dense_hit.rank,
                    lexical_score=None,
                    dense_score=dense_hit.score,
                    rrf_score=rrf_score,
                    source=food.source,
                    source_version=(
                        food.source_version
                    ),
                    candidate_source="manual",
                )

            fused_candidates.append(
                candidate
            )

        def ranking_key(
            candidate: FoodCandidate,
        ) -> tuple[
            int,
            int,
            float,
            int,
            str,
        ]:
            exact_priority = (
                0
                if (
                    candidate
                    .lexical_match_type
                    in {
                        "standard_name",
                        "alias",
                    }
                )
                else 1
            )

            category_priority = (
                self._category_priority(
                    candidate,
                    normalized_query,
                )
            )

            return (
                exact_priority,
                category_priority,
                -(
                    candidate.rrf_score
                    or 0.0
                ),
                candidate.stage,
                candidate.food_id,
            )

        fused_candidates.sort(
            key=ranking_key
        )

        gate_candidates = (
            fused_candidates[
                :max(top_k, 2)
            ]
        )

        gate_decision = (
            evaluate_retrieval_gate(
                gate_candidates,
                normalized_query,
                dense_available=True,
            )
        )

        candidates = (
            fused_candidates[:top_k]
            if gate_decision.allowed
            else []
        )

        return SearchResult(
            status=(
                "ok"
                if candidates
                else "not_found"
            ),
            query=query.strip(),
            normalized_query=(
                normalized_query
            ),
            top_k=top_k,
            candidates=candidates,
            auto_select_allowed=False,
            selection_mode="user_required",
            strategies_used=[
                "standard_name",
                "alias",
                "contains",
                "fuzzy",
                "dense",
                "rrf",
                "rule_rerank",
                "category_constraint",
                "weak_contains_filter",
                "refusal_gate",
                gate_decision.strategy_tag,
            ],
            dataset_id=(
                self.repository.dataset_id
            ),
            dataset_record_count=(
                self.repository.record_count
            ),
            elapsed_ms=round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )