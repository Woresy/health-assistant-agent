"""食物数据仓库、词法检索基线与 Hybrid RAG 入口。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.nutrition.text_normalize import (
    fuzzy_score,
    normalize_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FULL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "full"
    / "foods_normalized.json"
)
DEFAULT_SAMPLE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "foods_sample.json"
)
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "index"
)

LEXICAL_STRATEGIES = [
    "standard_name",
    "alias",
    "contains",
    "fuzzy",
]

LexicalMatchType = Literal[
    "standard_name",
    "alias",
    "contains",
    "fuzzy",
]


class NutritionDataError(Exception):
    """营养数据、查询或 RAG 配置错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class FoodRecord(BaseModel):
    """清洗数据中的完整食物记录。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: list[str]
    category: str = Field(min_length=1)

    calories_per_100g: float = Field(ge=0)
    protein_per_100g: float = Field(ge=0)
    fat_per_100g: float = Field(ge=0)
    carbs_per_100g: float = Field(ge=0)

    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    updated_at: date

    quality_flags: list[str] = Field(
        default_factory=list
    )


class FoodCandidate(BaseModel):
    """词法、Dense 或 Hybrid RAG 返回的候选食物。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str
    name: str
    category: str

    score: float = Field(ge=0, le=1)
    stage: int = Field(ge=0, le=4)

    match_type: Literal[
        "standard_name",
        "alias",
        "contains",
        "fuzzy",
        "dense",
        "hybrid",
    ]
    lexical_match_type: LexicalMatchType | None = None
    matched_term: str

    lexical_rank: int | None = Field(
        default=None,
        ge=1,
    )
    dense_rank: int | None = Field(
        default=None,
        ge=1,
    )
    lexical_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    dense_score: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    rrf_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    source: str
    source_version: str
    candidate_source: Literal["manual"] = "manual"


class SearchResult(BaseModel):
    """食物检索的完整可解释返回。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "not_found"]
    query: str
    normalized_query: str
    top_k: int = Field(ge=1, le=50)
    candidates: list[FoodCandidate]

    auto_select_allowed: bool
    selection_mode: Literal[
        "auto",
        "user_required",
    ]

    strategies_used: list[str]
    dataset_id: str
    dataset_record_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)


def resolve_food_data_path() -> Path:
    """解析当前使用的食物数据文件。"""

    override = os.getenv(
        "FOOD_DATA_PATH",
        "",
    ).strip()

    if override:
        return Path(override).expanduser()

    if DEFAULT_FULL_DATA_PATH.exists():
        return DEFAULT_FULL_DATA_PATH

    return DEFAULT_SAMPLE_DATA_PATH


def resolve_index_dir() -> Path:
    """解析向量索引目录。"""

    override = os.getenv(
        "FOOD_INDEX_DIR",
        "",
    ).strip()

    if override:
        path = Path(override).expanduser()

        if not path.is_absolute():
            return PROJECT_ROOT / path

        return path

    return DEFAULT_INDEX_DIR


class FoodRepository:
    """
    加载结构化食物事实，并提供词法或 Hybrid RAG 检索。

    默认 lexical，确保没有模型和索引时仍能启动。
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        rag_mode: str | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.data_path = (
            Path(data_path)
            if data_path is not None
            else resolve_food_data_path()
        )
        self.index_dir = (
            Path(index_dir)
            if index_dir is not None
            else resolve_index_dir()
        )

        configured_mode = (
            rag_mode
            if rag_mode is not None
            else os.getenv(
                "RAG_MODE",
                "lexical",
            )
        )
        self.rag_mode = configured_mode.strip().lower()

        if self.rag_mode not in {
            "lexical",
            "hybrid",
        }:
            raise NutritionDataError(
                "RAG_MODE_INVALID",
                "RAG_MODE 只能是 lexical 或 hybrid",
            )

        (
            self._foods,
            self.dataset_sha256,
        ) = self._load()

        self._foods_by_id = {
            food.food_id: food
            for food in self._foods
        }

        self.dataset_id = (
            f"{self.data_path.stem}:"
            f"sha256:{self.dataset_sha256[:12]}"
        )

        self._hybrid_retriever: Any | None = None

    @property
    def foods(self) -> tuple[FoodRecord, ...]:
        """以只读 tuple 暴露全部食物。"""

        return tuple(self._foods)

    @property
    def record_count(self) -> int:
        return len(self._foods)

    @property
    def known_food_ids(self) -> set[str]:
        return set(self._foods_by_id)

    def _load(
        self,
    ) -> tuple[list[FoodRecord], str]:
        try:
            raw_bytes = self.data_path.read_bytes()
            raw_text = raw_bytes.decode("utf-8")
            raw_data = json.loads(raw_text)
        except FileNotFoundError as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_MISSING",
                f"找不到食物数据文件：{self.data_path}",
            ) from exc
        except UnicodeDecodeError as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                f"食物数据不是 UTF-8：{exc}",
            ) from exc
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                (
                    "食物数据无法读取或不是合法 JSON："
                    f"{exc}"
                ),
            ) from exc

        if not isinstance(raw_data, list):
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                "食物数据顶层必须是数组",
            )

        if not raw_data:
            raise NutritionDataError(
                "NUTRITION_DATA_EMPTY",
                "食物数据不能为空数组",
            )

        try:
            foods = [
                FoodRecord.model_validate(item)
                for item in raw_data
            ]
        except ValidationError as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                (
                    "食物数据字段缺失或无法解析："
                    f"{exc}"
                ),
            ) from exc

        food_ids = [
            food.food_id
            for food in foods
        ]

        if len(food_ids) != len(set(food_ids)):
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                "食物数据包含重复 food_id",
            )

        dataset_sha256 = hashlib.sha256(
            raw_bytes
        ).hexdigest()

        return foods, dataset_sha256

    @staticmethod
    def _best_term(
        normalized_query: str,
        terms: list[tuple[str, str]],
        stage: int,
    ) -> tuple[float, str] | None:
        """返回单条食物在指定阶段的最佳命中。"""

        matches: list[
            tuple[float, str, str]
        ] = []

        for original_term, normalized_term in terms:
            if not normalized_term:
                continue

            if stage == 2:
                if len(normalized_query) < 2:
                    continue

                if not (
                    normalized_query in normalized_term
                    or normalized_term
                    in normalized_query
                ):
                    continue

                shorter = min(
                    len(normalized_query),
                    len(normalized_term),
                )
                longer = max(
                    len(normalized_query),
                    len(normalized_term),
                )
                score = round(
                    0.60
                    + 0.30 * shorter / longer,
                    4,
                )
            else:
                longest = max(
                    len(normalized_query),
                    len(normalized_term),
                )
                length_gap = abs(
                    len(normalized_query)
                    - len(normalized_term)
                )

                if length_gap > max(
                    2,
                    longest // 2,
                ):
                    continue

                similarity = fuzzy_score(
                    normalized_query,
                    normalized_term,
                )

                threshold = (
                    0.62
                    if len(normalized_query) <= 4
                    else 0.68
                )

                if similarity < threshold:
                    continue

                score = round(
                    0.30
                    + 0.29 * similarity,
                    4,
                )

            matches.append(
                (
                    score,
                    normalized_term,
                    original_term,
                )
            )

        if not matches:
            return None

        score, _, matched_term = min(
            matches,
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            ),
        )

        return score, matched_term

    def _candidate_for_food(
        self,
        food: FoodRecord,
        normalized_query: str,
    ) -> FoodCandidate | None:
        normalized_name = normalize_text(food.name)
        alias_terms = [
            (
                alias,
                normalize_text(alias),
            )
            for alias in food.aliases
        ]

        if normalized_query == normalized_name:
            stage = 0
            score = 1.0
            match_type: LexicalMatchType = (
                "standard_name"
            )
            matched_term = food.name
        else:
            exact_aliases = [
                alias
                for alias, normalized_alias
                in alias_terms
                if normalized_query
                == normalized_alias
            ]

            if exact_aliases:
                stage = 1
                score = 0.95
                match_type = "alias"
                matched_term = min(
                    exact_aliases,
                    key=lambda item: (
                        normalize_text(item),
                        item,
                    ),
                )
            else:
                all_terms = [
                    (
                        food.name,
                        normalized_name,
                    ),
                    *alias_terms,
                ]

                contains_match = self._best_term(
                    normalized_query,
                    all_terms,
                    2,
                )

                if contains_match is not None:
                    stage = 2
                    score, matched_term = (
                        contains_match
                    )
                    match_type = "contains"
                else:
                    fuzzy_match = self._best_term(
                        normalized_query,
                        all_terms,
                        3,
                    )

                    if fuzzy_match is None:
                        return None

                    stage = 3
                    score, matched_term = (
                        fuzzy_match
                    )
                    match_type = "fuzzy"

        return FoodCandidate(
            food_id=food.food_id,
            name=food.name,
            category=food.category,
            score=score,
            stage=stage,
            match_type=match_type,
            lexical_match_type=match_type,
            matched_term=matched_term,
            lexical_score=score,
            source=food.source,
            source_version=food.source_version,
            candidate_source="manual",
        )

    def search_lexical(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResult:
        """只运行确定性词法检索。"""

        started_at = perf_counter()

        if not isinstance(query, str):
            raise NutritionDataError(
                "QUERY_INVALID",
                "食物检索词必须是字符串",
            )

        stripped_query = query.strip()
        normalized_query = normalize_text(
            stripped_query
        )

        if not normalized_query:
            raise NutritionDataError(
                "QUERY_REQUIRED",
                "食物检索词不能为空",
            )

        if len(stripped_query) > 64:
            raise NutritionDataError(
                "QUERY_TOO_LONG",
                "食物检索词不得超过 64 个字符",
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 50
        ):
            raise NutritionDataError(
                "TOP_K_INVALID",
                "top_k 必须是 1—50 的整数",
            )

        ranked = [
            candidate
            for food in self._foods
            if (
                candidate
                := self._candidate_for_food(
                    food,
                    normalized_query,
                )
            )
            is not None
        ]

        ranked.sort(
            key=lambda item: (
                item.stage,
                -item.score,
                item.food_id,
            )
        )

        candidates = ranked[:top_k]

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            candidate.lexical_rank = rank

        return SearchResult(
            status=(
                "ok"
                if candidates
                else "not_found"
            ),
            query=stripped_query,
            normalized_query=normalized_query,
            top_k=top_k,
            candidates=candidates,
            auto_select_allowed=False,
            selection_mode="user_required",
            strategies_used=list(
                LEXICAL_STRATEGIES
            ),
            dataset_id=self.dataset_id,
            dataset_record_count=self.record_count,
            elapsed_ms=round(
                (
                    perf_counter()
                    - started_at
                )
                * 1000,
                3,
            ),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResult:
        """根据 RAG_MODE 运行词法或 Hybrid RAG。"""

        if self.rag_mode == "lexical":
            return self.search_lexical(
                query,
                top_k=top_k,
            )

        if self._hybrid_retriever is None:
            from src.nutrition.hybrid_retriever import (
                HybridFoodRetriever,
            )

            self._hybrid_retriever = (
                HybridFoodRetriever(
                    repository=self,
                    index_dir=self.index_dir,
                )
            )

        return self._hybrid_retriever.search(
            query,
            top_k=top_k,
        )

    def get_by_food_id(
        self,
        food_id: str,
    ) -> FoodRecord:
        """按稳定 food_id 读取结构化营养事实。"""

        food = self._foods_by_id.get(food_id)

        if food is None:
            raise NutritionDataError(
                "FOOD_ID_NOT_FOUND",
                f"找不到 food_id：{food_id}",
            )

        return food