"""四类健康事件领域模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.health.migrations import (
    CURRENT_HEALTH_EVENT_SCHEMA_VERSION,
    migrate_health_event_data,
)


class StrictModel(BaseModel):
    """禁止调用方加入未定义字段。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class EventType(str, Enum):
    """P0 支持的四类健康事件。"""

    MEAL = "meal"
    WATER = "water"
    WEIGHT = "weight"
    EXERCISE = "exercise"


class InputSource(str, Enum):
    """PRD 规定的健康事件输入来源。"""

    CHAT = "chat"
    IMAGE = "image"
    MODEL = "model"


class CandidateSource(str, Enum):
    """食物候选来源。"""

    MANUAL = "manual"
    MODEL = "model"


class ExerciseIntensity(str, Enum):
    """可选运动强度。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MealFood(StrictModel):
    """用户最终确认的食物快照。"""

    food_id: str = Field(
        min_length=1,
        max_length=128,
    )
    name: str = Field(
        min_length=1,
        max_length=128,
    )
    category: str = Field(
        min_length=1,
        max_length=128,
    )

    @field_validator(
        "food_id",
        "name",
        "category",
        mode="before",
    )
    @classmethod
    def strip_text(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class MealPortion(StrictModel):
    """食物可食部分克重。"""

    grams: float = Field(
        gt=0,
        le=10000,
    )
    unit: Literal["g"] = "g"


class MealNutrition(StrictModel):
    """确定性计算产生的营养估算。"""

    calories_kcal: float = Field(
        ge=0,
    )
    protein_g: float = Field(
        ge=0,
    )
    fat_g: float = Field(
        ge=0,
    )
    carbs_g: float = Field(
        ge=0,
    )

    source_ref: str = Field(
        min_length=1,
        max_length=1000,
    )
    retrieval_query: str = Field(
        min_length=1,
        max_length=128,
    )
    selected_food_code: str = Field(
        min_length=1,
        max_length=128,
    )
    portion_assumption: str = Field(
        min_length=1,
        max_length=500,
    )
    estimated: Literal[True] = True

    @field_validator(
        "source_ref",
        "retrieval_query",
        "selected_food_code",
        "portion_assumption",
        mode="before",
    )
    @classmethod
    def strip_text(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class MealPayload(StrictModel):
    """饮食事件 payload。"""

    food: MealFood
    portion: MealPortion
    nutrition: MealNutrition

    retrieval_query: str = Field(
        min_length=1,
        max_length=128,
    )
    candidate_source: CandidateSource = (
        CandidateSource.MANUAL
    )
    estimated: Literal[True] = True

    @field_validator(
        "retrieval_query",
        mode="before",
    )
    @classmethod
    def strip_query(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class WaterPayload(StrictModel):
    """饮水事件 payload。"""

    amount_ml: float = Field(
        gt=0,
        le=10000,
    )
    unit: Literal["ml"] = "ml"

    beverage: str = Field(
        default="饮用水",
        min_length=1,
        max_length=64,
    )
    note: str = Field(
        default="",
        max_length=500,
    )

    @field_validator(
        "beverage",
        "note",
        mode="before",
    )
    @classmethod
    def strip_text(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class WeightPayload(StrictModel):
    """体重事件 payload。"""

    weight_kg: float = Field(
        gt=0,
        le=500,
    )
    unit: Literal["kg"] = "kg"

    note: str = Field(
        default="",
        max_length=500,
    )

    @field_validator(
        "note",
        mode="before",
    )
    @classmethod
    def strip_note(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class ExercisePayload(StrictModel):
    """运动事件 payload。"""

    activity_type: str = Field(
        min_length=1,
        max_length=100,
    )
    duration_minutes: float = Field(
        gt=0,
        le=1440,
    )

    distance_km: float | None = Field(
        default=None,
        gt=0,
        le=1000,
    )
    intensity: (
        ExerciseIntensity
        | None
    ) = None

    note: str = Field(
        default="",
        max_length=500,
    )

    @field_validator(
        "activity_type",
        "note",
        mode="before",
    )
    @classmethod
    def strip_text(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


EventPayload = (
    MealPayload
    | WaterPayload
    | WeightPayload
    | ExercisePayload
)


class HealthEvent(StrictModel):
    """四类健康事件共用的外层结构。"""

    schema_version: Literal[
        "1.1"
    ] = (
        CURRENT_HEALTH_EVENT_SCHEMA_VERSION
    )

    event_id: UUID
    user_id: str = Field(
        min_length=1,
        max_length=128,
    )
    event_type: EventType

    occurred_at: datetime
    payload: EventPayload

    source_refs: list[str] = Field(
        default_factory=list,
    )
    input_source: InputSource

    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_data(
        cls,
        value: Any,
    ) -> Any:
        """
        在 Pydantic 字段校验前迁移旧数据。

        这样现有 JSONL 和旧 UI 构造的
        image_manual meal 仍可读取。
        """

        if isinstance(value, cls):
            return value

        if isinstance(value, dict):
            return (
                migrate_health_event_data(
                    value
                )
            )

        return value

    @field_validator(
        "user_id",
        mode="before",
    )
    @classmethod
    def strip_user_id(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value

    @field_validator(
        "occurred_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def require_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """拒绝没有时区的日期时间。"""

        if (
            value.tzinfo is None
            or value.utcoffset()
            is None
        ):
            raise ValueError(
                "时间字段必须包含时区"
            )

        return value

    @field_validator(
        "source_refs",
        mode="before",
    )
    @classmethod
    def normalize_source_refs(
        cls,
        value: Any,
    ) -> Any:
        """清理来源引用并拒绝空项。"""

        if not isinstance(value, list):
            return value

        normalized: list[Any] = []

        for item in value:
            if isinstance(item, str):
                stripped = (
                    item.strip()
                )

                if not stripped:
                    raise ValueError(
                        "source_refs "
                        "不得包含空字符串"
                    )

                normalized.append(
                    stripped
                )
            else:
                normalized.append(item)

        return normalized

    @model_validator(mode="after")
    def validate_event_payload(
        self,
    ) -> "HealthEvent":
        """确保 event_type 与 payload 类型一致。"""

        expected_payload_types = {
            EventType.MEAL: (
                MealPayload
            ),
            EventType.WATER: (
                WaterPayload
            ),
            EventType.WEIGHT: (
                WeightPayload
            ),
            EventType.EXERCISE: (
                ExercisePayload
            ),
        }

        expected_type = (
            expected_payload_types[
                self.event_type
            ]
        )

        if not isinstance(
            self.payload,
            expected_type,
        ):
            raise ValueError(
                "event_type 与 payload "
                "类型不一致："
                f"{self.event_type.value} "
                "需要 "
                f"{expected_type.__name__}"
            )

        if (
            self.event_type
            is EventType.MEAL
            and not self.source_refs
        ):
            raise ValueError(
                "meal 事件必须保留"
                "营养数据来源"
            )

        if (
            self.updated_at
            < self.created_at
        ):
            raise ValueError(
                "updated_at 不得早于 "
                "created_at"
            )

        return self