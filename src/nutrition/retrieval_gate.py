"""Hybrid Retrieval 的置信度拒答门控。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.nutrition.repository import (
    FoodCandidate,
)


GateReasonCode = Literal[
    "no_candidate",
    "exact_lexical",
    "lexical_fallback",
    "hybrid_consensus",
    "dense_category_supported",
    "dense_high_confidence",
    "out_of_domain",
    "low_confidence",
    "ambiguous_dense",
]


@dataclass(
    frozen=True,
    slots=True,
)
class RetrievalGateDecision:
    """拒答门控的一次确定性判定。"""

    allowed: bool
    reason_code: GateReasonCode
    detail: str

    @property
    def strategy_tag(self) -> str:
        """生成可以写入 strategies_used 的稳定标记。"""

        action = (
            "pass"
            if self.allowed
            else "reject"
        )

        return (
            f"refusal_gate:{action}:"
            f"{self.reason_code}"
        )


# 包含匹配至少达到该分数，
# 才能在 Dense 不可用时单独通过。
MIN_CONTAINS_LEXICAL_SCORE = 0.72

# 模糊匹配至少达到该分数，
# 才能在 Dense 不可用时单独通过。
MIN_FUZZY_LEXICAL_SCORE = 0.48

# 词法和 Dense 同时支持同一候选时，
# Dense 相似度最低要求。
MIN_HYBRID_DENSE_SCORE = 0.45

# 查询明确包含食物类别时，
# Dense 候选最低相似度要求。
MIN_CATEGORY_DENSE_SCORE = 0.55

# 只有 Dense 证据时，
# 排名第一候选最低相似度要求。
MIN_DENSE_ONLY_SCORE = 0.62

# 只有一个 Dense 候选可比较时，
# 使用更严格的分数要求。
MIN_SINGLE_DENSE_SCORE = 0.72

# 第一名与最强竞争候选的最小分差。
MIN_DENSE_MARGIN = 0.04


FOOD_DOMAIN_MARKERS = (
    "食物",
    "食品",
    "菜",
    "蔬菜",
    "水果",
    "果",
    "根茎",
    "肉",
    "鱼",
    "虾",
    "蟹",
    "贝",
    "奶",
    "乳",
    "豆",
    "蛋",
    "饭",
    "面",
    "粥",
    "汤",
    "饮品",
    "早餐",
    "午餐",
    "晚餐",
    "夜宵",
    "零食",
    "吃",
    "喝",
)


CATEGORY_RULES: tuple[
    tuple[str, tuple[str, ...]],
    ...,
] = (
    (
        "蔬菜",
        ("蔬菜",),
    ),
    (
        "根茎",
        ("蔬菜",),
    ),
    (
        "水果",
        ("水果",),
    ),
    (
        "果类",
        ("水果",),
    ),
    (
        "乳制品",
        ("乳", "奶"),
    ),
    (
        "奶制品",
        ("乳", "奶"),
    ),
    (
        "牛奶",
        ("乳", "奶"),
    ),
    (
        "肉类",
        ("肉",),
    ),
    (
        "鱼类",
        ("鱼",),
    ),
    (
        "海鲜",
        ("鱼", "虾", "蟹", "贝"),
    ),
    (
        "主食",
        ("谷", "薯", "主食"),
    ),
    (
        "豆制品",
        ("豆",),
    ),
    (
        "蛋类",
        ("蛋",),
    ),
)


def _has_food_domain_signal(
    normalized_query: str,
) -> bool:
    """查询中是否存在可解释的饮食领域信号。"""

    return any(
        marker in normalized_query
        for marker in FOOD_DOMAIN_MARKERS
    )


def _category_supported(
    candidate: FoodCandidate,
    normalized_query: str,
) -> bool:
    """查询明确指定类别，且候选属于对应类别。"""

    category = candidate.category.strip()

    requested_category_fragments: list[
        tuple[str, ...]
    ] = [
        category_fragments
        for query_marker, category_fragments
        in CATEGORY_RULES
        if query_marker in normalized_query
    ]

    if not requested_category_fragments:
        return False

    return any(
        category_fragment in category
        for fragments
        in requested_category_fragments
        for category_fragment in fragments
    )


def _dense_margin(
    top_candidate: FoodCandidate,
    candidates: Sequence[FoodCandidate],
) -> float | None:
    """
    计算当前第一名与最强 Dense 竞争者之间的分差。

    返回负数表示：规则重排后的第一名，
    Dense 原始分数低于另一名候选。
    """

    if top_candidate.dense_score is None:
        return None

    competing_scores = [
        candidate.dense_score
        for candidate in candidates
        if (
            candidate.food_id
            != top_candidate.food_id
            and candidate.dense_score
            is not None
        )
    ]

    if not competing_scores:
        return None

    return round(
        top_candidate.dense_score
        - max(competing_scores),
        6,
    )


def evaluate_retrieval_gate(
    candidates: Sequence[FoodCandidate],
    normalized_query: str,
    *,
    dense_available: bool,
) -> RetrievalGateDecision:
    """
    判断 Hybrid Retrieval 是否有足够证据返回候选。

    门控采用确定性规则，不让大模型自行决定阈值。
    """

    if not candidates:
        return RetrievalGateDecision(
            allowed=False,
            reason_code="no_candidate",
            detail="词法检索和 Dense 检索都没有返回候选",
        )

    top_candidate = candidates[0]

    if top_candidate.lexical_match_type in {
        "standard_name",
        "alias",
    }:
        return RetrievalGateDecision(
            allowed=True,
            reason_code="exact_lexical",
            detail=(
                "第一名命中了标准名称或人工维护的别名"
            ),
        )

    if not dense_available:
        lexical_score = (
            top_candidate.lexical_score
            if top_candidate.lexical_score
            is not None
            else 0.0
        )

        if (
            top_candidate.lexical_match_type
            == "contains"
            and lexical_score
            >= MIN_CONTAINS_LEXICAL_SCORE
        ):
            return RetrievalGateDecision(
                allowed=True,
                reason_code="lexical_fallback",
                detail=(
                    "Dense 不可用，但包含匹配达到"
                    "词法降级阈值"
                ),
            )

        if (
            top_candidate.lexical_match_type
            == "fuzzy"
            and lexical_score
            >= MIN_FUZZY_LEXICAL_SCORE
            and _has_food_domain_signal(
                normalized_query
            )
        ):
            return RetrievalGateDecision(
                allowed=True,
                reason_code="lexical_fallback",
                detail=(
                    "Dense 不可用，但饮食领域内的"
                    "模糊匹配达到降级阈值"
                ),
            )

        return RetrievalGateDecision(
            allowed=False,
            reason_code="low_confidence",
            detail=(
                "Dense 不可用，剩余词法证据不足，"
                "拒绝返回候选"
            ),
        )

    dense_score = top_candidate.dense_score

    if (
        top_candidate.lexical_rank is not None
        and top_candidate.dense_rank is not None
        and dense_score is not None
        and dense_score
        >= MIN_HYBRID_DENSE_SCORE
    ):
        return RetrievalGateDecision(
            allowed=True,
            reason_code="hybrid_consensus",
            detail=(
                "词法检索和 Dense 检索共同支持"
                "第一名候选"
            ),
        )

    if dense_score is None:
        return RetrievalGateDecision(
            allowed=False,
            reason_code="low_confidence",
            detail=(
                "第一名没有 Dense 相似度证据，"
                "且没有精确词法命中"
            ),
        )

    if not _has_food_domain_signal(
        normalized_query
    ):
        return RetrievalGateDecision(
            allowed=False,
            reason_code="out_of_domain",
            detail=(
                "查询中没有检测到饮食领域信号，"
                "拒绝将普通文本强行匹配为食物"
            ),
        )

    if (
        _category_supported(
            top_candidate,
            normalized_query,
        )
        and dense_score
        >= MIN_CATEGORY_DENSE_SCORE
    ):
        return RetrievalGateDecision(
            allowed=True,
            reason_code=(
                "dense_category_supported"
            ),
            detail=(
                "Dense 相似度达到阈值，且查询类别"
                "与候选食物类别一致"
            ),
        )

    if dense_score < MIN_DENSE_ONLY_SCORE:
        return RetrievalGateDecision(
            allowed=False,
            reason_code="low_confidence",
            detail=(
                "Dense 第一名相似度低于"
                f"{MIN_DENSE_ONLY_SCORE:.2f}"
            ),
        )

    margin = _dense_margin(
        top_candidate,
        candidates,
    )

    if margin is None:
        if dense_score >= MIN_SINGLE_DENSE_SCORE:
            return RetrievalGateDecision(
                allowed=True,
                reason_code=(
                    "dense_high_confidence"
                ),
                detail=(
                    "只有一个可比较的 Dense 候选，"
                    "但相似度达到严格阈值"
                ),
            )

        return RetrievalGateDecision(
            allowed=False,
            reason_code="ambiguous_dense",
            detail=(
                "缺少第二名用于比较，且第一名"
                "没有达到单候选严格阈值"
            ),
        )

    if margin < MIN_DENSE_MARGIN:
        return RetrievalGateDecision(
            allowed=False,
            reason_code="ambiguous_dense",
            detail=(
                "Dense 第一名与竞争候选分差只有"
                f"{margin:.4f}，低于"
                f"{MIN_DENSE_MARGIN:.2f}"
            ),
        )

    return RetrievalGateDecision(
        allowed=True,
        reason_code="dense_high_confidence",
        detail=(
            "Dense 第一名相似度和领先分差"
            "均达到门控要求"
        ),
    )