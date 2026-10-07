"""Explicitly labeled visual nutrition estimates, separate from sourced food records."""
from datetime import date
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.nutrition.repository import FoodRecord


class ModelNutritionEstimate(BaseModel):
    """Approximate cooked-food composition per 100g, provided by the vision model."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    calories_per_100g: float = Field(ge=0, le=950)
    protein_per_100g: float = Field(ge=0, le=100)
    fat_per_100g: float = Field(ge=0, le=100)
    carbs_per_100g: float = Field(ge=0, le=100)

    @model_validator(mode='after')
    def check_mass(self):
        if self.protein_per_100g + self.fat_per_100g + self.carbs_per_100g > 100:
            raise ValueError('每100克食物的营养成分合计不能超过100克')
        return self

    def to_food_record(self, name: str, model_version: str) -> FoodRecord:
        """Reuse deterministic scaling without claiming a food-table or web source."""
        return FoodRecord(
            food_id='MODEL_' + sha256((model_version + name).encode()).hexdigest()[:20],
            name=name, aliases=[], category='图片餐食估算',
            **self.model_dump(), source='模型估算（未查证营养数据库）',
            source_version=model_version[:150] or '视觉模型', updated_at=date.today(),
            quality_flags=['model_estimate', 'visual_portion', 'recipe_uncertain'],
        )
