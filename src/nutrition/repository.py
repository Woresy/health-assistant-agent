"""可切换数据集的确定性、可解释四阶段食物检索。"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.nutrition.text_normalize import fuzzy_score, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_DATA_PATH = PROJECT_ROOT / "data" / "full" / "foods_normalized.json"
DEFAULT_SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"
STRATEGIES = ["standard_name", "alias", "contains", "fuzzy"]


class NutritionDataError(Exception):
    """营养数据或查询错误，包含稳定错误码。"""

    def __init__(self, error_code: str, message: str) -> None:
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
    quality_flags: list[str] = Field(default_factory=list)


class FoodCandidate(BaseModel):
    """检索候选只含溯源信息，禁止携带营养数值。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str
    name: str
    category: str
    score: float = Field(ge=0, le=1)
    stage: int = Field(ge=0, le=3)
    match_type: Literal["standard_name", "alias", "contains", "fuzzy"]
    matched_term: str
    source: str
    source_version: str
    candidate_source: Literal["manual"] = "manual"


class SearchResult(BaseModel):
    """食物检索的完整、可解释返回。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "not_found"]
    query: str
    normalized_query: str
    top_k: int = Field(ge=1, le=10)
    candidates: list[FoodCandidate]
    auto_select_allowed: bool
    selection_mode: Literal["auto", "user_required"]
    strategies_used: list[str]
    dataset_id: str
    dataset_record_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)


def resolve_food_data_path() -> Path:
    """按环境变量、完整数据、公开示例的顺序解析数据集路径。"""

    override = os.getenv("FOOD_DATA_PATH")
    if override:
        return Path(override).expanduser()
    if DEFAULT_FULL_DATA_PATH.exists():
        return DEFAULT_FULL_DATA_PATH
    return DEFAULT_SAMPLE_DATA_PATH


class FoodRepository:
    """从固定 JSON 加载食物，并执行四阶段确定性检索。"""

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = (
            Path(data_path) if data_path is not None else resolve_food_data_path()
        )
        self._foods = self._load()
        self.dataset_id = self.data_path.stem

    def _load(self) -> list[FoodRecord]:
        try:
            raw_text = self.data_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
        except FileNotFoundError as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_MISSING",
                f"找不到食物数据文件：{self.data_path}",
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                f"食物数据无法读取或不是合法 JSON：{exc}",
            ) from exc

        if not isinstance(raw_data, list):
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                "食物数据顶层必须是数组",
            )

        try:
            foods = [FoodRecord.model_validate(item) for item in raw_data]
        except ValidationError as exc:
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                f"食物数据字段缺失或无法解析：{exc}",
            ) from exc

        food_ids = [food.food_id for food in foods]
        if len(food_ids) != len(set(food_ids)):
            raise NutritionDataError(
                "NUTRITION_DATA_INVALID",
                "食物数据包含重复 food_id",
            )
        return foods

    @staticmethod
    def _best_term(
        normalized_query: str,
        terms: list[tuple[str, str]],
        stage: int,
    ) -> tuple[float, str] | None:
        """返回单条食物在指定阶段的最佳得分及命中词。"""

        matches: list[tuple[float, str, str]] = []
        for original_term, normalized_term in terms:
            if not normalized_term:
                continue
            if stage == 2:
                if not (
                    normalized_query in normalized_term
                    or normalized_term in normalized_query
                ):
                    continue
                shorter = min(len(normalized_query), len(normalized_term))
                longer = max(len(normalized_query), len(normalized_term))
                score = round(0.60 + 0.30 * shorter / longer, 4)
            else:
                similarity = fuzzy_score(normalized_query, normalized_term)
                if similarity < 0.45:
                    continue
                score = round(0.30 + 0.29 * similarity, 4)
            matches.append((score, normalized_term, original_term))

        if not matches:
            return None
        score, _, matched_term = min(
            matches,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        return score, matched_term

    def _candidate_for_food(
        self,
        food: FoodRecord,
        normalized_query: str,
    ) -> FoodCandidate | None:
        """为一条食物保留阶段最小的唯一最佳命中。"""

        normalized_name = normalize_text(food.name)
        alias_terms = [(alias, normalize_text(alias)) for alias in food.aliases]

        if normalized_query == normalized_name:
            stage = 0
            score = 1.0
            match_type = "standard_name"
            matched_term = food.name
        else:
            exact_aliases = [
                alias
                for alias, normalized_alias in alias_terms
                if normalized_query == normalized_alias
            ]
            if exact_aliases:
                stage = 1
                score = 0.95
                match_type = "alias"
                matched_term = min(
                    exact_aliases,
                    key=lambda item: (normalize_text(item), item),
                )
            else:
                all_terms = [(food.name, normalized_name), *alias_terms]
                contains_match = self._best_term(normalized_query, all_terms, 2)
                if contains_match is not None:
                    stage = 2
                    score, matched_term = contains_match
                    match_type = "contains"
                else:
                    fuzzy_match = self._best_term(normalized_query, all_terms, 3)
                    if fuzzy_match is None:
                        return None
                    stage = 3
                    score, matched_term = fuzzy_match
                    match_type = "fuzzy"

        return FoodCandidate(
            food_id=food.food_id,
            name=food.name,
            category=food.category,
            score=score,
            stage=stage,
            match_type=match_type,
            matched_term=matched_term,
            source=food.source,
            source_version=food.source_version,
            candidate_source="manual",
        )

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """严格按四阶段、固定排序和明确自动选择规则检索。"""

        started_at = perf_counter()
        if not isinstance(query, str):
            raise NutritionDataError("QUERY_INVALID", "食物检索词必须是字符串")
        stripped_query = query.strip()
        normalized_query = normalize_text(stripped_query)
        if not normalized_query:
            raise NutritionDataError("QUERY_REQUIRED", "食物检索词不能为空")
        if len(stripped_query) > 64:
            raise NutritionDataError("QUERY_TOO_LONG", "食物检索词不得超过 64 个字符")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 10
        ):
            raise NutritionDataError("TOP_K_INVALID", "top_k 必须是 1—10 的整数")

        ranked = [
            candidate
            for food in self._foods
            if (candidate := self._candidate_for_food(food, normalized_query))
            is not None
        ]
        ranked.sort(key=lambda item: (item.stage, -item.score, item.food_id))
        candidates = ranked[:top_k]

        auto_select_allowed = bool(
            candidates
            and candidates[0].stage <= 1
            and (
                len(candidates) == 1
                or candidates[1].score < candidates[0].score - 0.15
            )
        )
        return SearchResult(
            status="ok" if candidates else "not_found",
            query=stripped_query,
            normalized_query=normalized_query,
            top_k=top_k,
            candidates=candidates,
            auto_select_allowed=auto_select_allowed,
            selection_mode="auto" if auto_select_allowed else "user_required",
            strategies_used=list(STRATEGIES),
            dataset_id=self.dataset_id,
            dataset_record_count=len(self._foods),
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        )

    def get_by_food_id(self, food_id: str) -> FoodRecord:
        """只允许对真实存在的 food_id 计算营养值。"""

        for food in self._foods:
            if food.food_id == food_id:
                return food
        raise NutritionDataError(
            "FOOD_ID_NOT_FOUND",
            f"找不到 food_id：{food_id}",
        )
