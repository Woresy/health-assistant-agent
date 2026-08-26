"""健康事件领域模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """禁止调用方悄悄加入未定义字段。"""

    model_config = ConfigDict(extra="forbid")


class EventType(str, Enum):
    """事件类型结构预留；当天只实现 meal。"""

    MEAL = "meal"
    WATER = "water"
    WEIGHT = "weight"
    EXERCISE = "exercise"


class InputSource(str, Enum):
    """记录的输入来源。"""

    IMAGE_MANUAL = "image_manual"


class MealFood(StrictModel):
    """用户最终确认的食物快照。"""

    food_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)


class MealPortion(StrictModel):
    """当天只支持克重。"""

    grams: PositiveFloat
    unit: Literal["g"] = "g"


class MealNutrition(StrictModel):
    """确定性计算产生的营养估算结果。"""

    calories_kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    source_ref: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    selected_food_code: str = Field(min_length=1)
    portion_assumption: str = Field(min_length=1)
    estimated: Literal[True] = True


class MealPayload(StrictModel):
    """meal 事件载荷；食物和份量均为必填。"""

    food: MealFood
    portion: MealPortion
    nutrition: MealNutrition
    retrieval_query: str = Field(min_length=1)
    candidate_source: Literal["manual"] = "manual"
    estimated: Literal[True] = True


class HealthEvent(StrictModel):
    """统一健康事件外层结构。"""

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    event_type: EventType
    occurred_at: datetime
    payload: MealPayload
    source_refs: list[str] = Field(min_length=1)
    input_source: InputSource
    created_at: datetime
    updated_at: datetime

    @field_validator("occurred_at", "created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """拒绝没有时区的信息，避免今日统计产生歧义。"""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时间字段必须包含时区")
        return value

    @model_validator(mode="after")
    def validate_event_payload(self) -> "HealthEvent":
        """当前版本只开放 meal，其他类型仅在枚举中预留。"""

        if self.event_type is not EventType.MEAL:
            raise ValueError(
                f"事件类型 {self.event_type.value} 已预留，但当前版本尚未实现"
            )
        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不得早于 created_at")
        return self