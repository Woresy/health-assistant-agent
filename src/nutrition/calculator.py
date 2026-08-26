"""不依赖模型的确定性营养计算。"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.nutrition.repository import FoodRecord


class NutritionCalculationError(Exception):
    """计算输入错误，包含稳定错误码。"""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class NutritionEstimate(BaseModel):
    """一次食物份量的确定性估算结果。"""

    model_config = ConfigDict(extra="forbid")

    calories_kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    source_ref: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    selected_food_code: str = Field(min_length=1)
    portion_assumption: str = Field(min_length=1)
    estimated: Literal[True] = True


def parse_grams(raw_grams: Any) -> Decimal:
    """把用户输入解析成有限正数克重。"""

    if isinstance(raw_grams, bool) or raw_grams is None:
        raise NutritionCalculationError(
            "PORTION_INVALID",
            "克重必须是大于 0 的数字",
        )

    try:
        grams = Decimal(str(raw_grams).strip())
    except (InvalidOperation, AttributeError, ValueError) as exc:
        raise NutritionCalculationError(
            "PORTION_INVALID",
            "克重无法解析为数字",
        ) from exc

    if not grams.is_finite() or grams <= 0:
        raise NutritionCalculationError(
            "PORTION_INVALID",
            "克重必须是有限且大于 0 的数字",
        )

    if grams > Decimal("10000"):
        raise NutritionCalculationError(
            "PORTION_TOO_LARGE",
            "单次克重不得超过 10000g",
        )

    return grams


def _scaled(per_100g: float, grams: Decimal) -> float:
    """按每 100g 等比例计算，并保留两位小数。"""

    if not math.isfinite(per_100g) or per_100g < 0:
        raise NutritionCalculationError(
            "NUTRITION_FIELD_INVALID",
            "营养字段缺失、为负数或不是有限数字",
        )

    result = (
        Decimal(str(per_100g)) * grams / Decimal("100")
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(result)


def calculate_nutrition(
    food: FoodRecord,
    raw_grams: Any,
    retrieval_query: str,
) -> NutritionEstimate:
    """严格执行“每 100g 数值 × 克重 ÷ 100”。"""

    query = retrieval_query.strip()
    if not query:
        raise NutritionCalculationError(
            "RETRIEVAL_QUERY_REQUIRED",
            "计算结果必须保留原始检索词",
        )

    grams = parse_grams(raw_grams)

    return NutritionEstimate(
        calories_kcal=_scaled(food.calories_per_100g, grams),
        protein_g=_scaled(food.protein_per_100g, grams),
        fat_g=_scaled(food.fat_per_100g, grams),
        carbs_g=_scaled(food.carbs_per_100g, grams),
        source_ref=(
            f"{food.food_id}｜{food.source}｜{food.source_version}"
        ),
        retrieval_query=query,
        selected_food_code=food.food_id,
        portion_assumption=(
            "按用户填写的可食部分克重计算；"
            "营养值按每100g数据等比例换算"
        ),
        estimated=True,
    )