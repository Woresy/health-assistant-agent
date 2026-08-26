"""固定示例食物数据的加载和确定性检索。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class NutritionDataError(Exception):
    """营养数据或查询错误，包含稳定错误码。"""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class FoodRecord(BaseModel):
    """示例数据中的完整食物记录。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    category: str = Field(min_length=1)
    calories_per_100g: float = Field(ge=0)
    protein_per_100g: float = Field(ge=0)
    fat_per_100g: float = Field(ge=0)
    carbs_per_100g: float = Field(ge=0)
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    updated_at: date


class FoodCandidate(BaseModel):
    """检索候选不包含营养数值，防止候选阶段误用。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str
    name: str
    category: str
    score: float
    match_type: Literal["standard_name", "alias", "contains"]
    source: str
    source_version: str


class SearchResult(BaseModel):
    """食物检索的统一返回。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "not_found"]
    query: str
    candidates: list[FoodCandidate]


def _normalize(value: str) -> str:
    """统一大小写并移除空白，不做模糊猜测。"""

    return "".join(value.casefold().split())


class FoodRepository:
    """从固定 JSON 加载食物，并按固定优先级检索。"""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self._foods = self._load()

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

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """标准名 → 别名 → 包含词，返回固定 Top-K。"""

        normalized_query = _normalize(query)
        if not normalized_query:
            raise NutritionDataError(
                "QUERY_REQUIRED",
                "食物检索词不能为空",
            )
        if top_k <= 0:
            raise NutritionDataError(
                "TOP_K_INVALID",
                "top_k 必须大于 0",
            )

        ranked: list[tuple[int, FoodCandidate]] = []

        for food in self._foods:
            normalized_name = _normalize(food.name)
            normalized_aliases = [_normalize(alias) for alias in food.aliases]

            if normalized_query == normalized_name:
                priority = 0
                score = 1.0
                match_type = "standard_name"
            elif normalized_query in normalized_aliases:
                priority = 1
                score = 0.95
                match_type = "alias"
            else:
                searchable_terms = [normalized_name, *normalized_aliases]
                has_contains_match = any(
                    normalized_query in term or term in normalized_query
                    for term in searchable_terms
                )
                if not has_contains_match:
                    continue
                priority = 2
                score = 0.70
                match_type = "contains"

            ranked.append(
                (
                    priority,
                    FoodCandidate(
                        food_id=food.food_id,
                        name=food.name,
                        category=food.category,
                        score=score,
                        match_type=match_type,
                        source=food.source,
                        source_version=food.source_version,
                    ),
                )
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                -item[1].score,
                item[1].food_id,
            )
        )
        candidates = [candidate for _, candidate in ranked[:top_k]]

        return SearchResult(
            status="ok" if candidates else "not_found",
            query=query.strip(),
            candidates=candidates,
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