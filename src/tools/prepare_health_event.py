"""准备四类健康事件保存草稿，不直接写入 JSONL。"""

from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from typing import (
    Annotated,
    Any,
    Literal,
)
from uuid import (
    NAMESPACE_URL,
    uuid5,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from src.health.models import (
    ExerciseIntensity,
    ExercisePayload,
    HealthEvent,
    InputSource,
    MealPayload,
    WaterPayload,
    WeightPayload,
)
from src.tools.confirmation import (
    issue_confirmation_token,
)


class DraftInputModel(BaseModel):
    """四类草稿输入的公共配置。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class MealEventDraftInput(
    DraftInputModel
):
    """饮食事件草稿输入。"""

    event_type: Literal["meal"]
    payload: MealPayload
    source_refs: list[str] = Field(
        min_length=1,
    )
    input_source: InputSource = (
        InputSource.IMAGE
    )
    occurred_at: (
        datetime
        | None
    ) = None


class WaterEventDraftInput(
    DraftInputModel
):
    """饮水事件草稿输入。"""

    event_type: Literal["water"]
    amount_ml: float = Field(
        gt=0,
        le=10000,
    )
    beverage: str = Field(
        default="饮用水",
        min_length=1,
        max_length=64,
    )
    note: str = Field(
        default="",
        max_length=500,
    )
    occurred_at: (
        datetime
        | None
    ) = None


class WeightEventDraftInput(
    DraftInputModel
):
    """体重事件草稿输入。"""

    event_type: Literal["weight"]
    weight_kg: float = Field(
        gt=0,
        le=500,
    )
    note: str = Field(
        default="",
        max_length=500,
    )
    occurred_at: (
        datetime
        | None
    ) = None


class ExerciseEventDraftInput(
    DraftInputModel
):
    """运动事件草稿输入。"""

    event_type: Literal["exercise"]
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
    occurred_at: (
        datetime
        | None
    ) = None


HealthEventDraftInput = Annotated[
    (
        MealEventDraftInput
        | WaterEventDraftInput
        | WeightEventDraftInput
        | ExerciseEventDraftInput
    ),
    Field(
        discriminator="event_type"
    ),
]


_DRAFT_INPUT_ADAPTER = TypeAdapter(
    HealthEventDraftInput
)


def _error(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """构造统一工具错误。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def _normalize_idempotency_key(
    idempotency_key: str,
) -> tuple[
    str | None,
    dict[str, Any] | None,
]:
    """清理幂等键。"""

    if not isinstance(
        idempotency_key,
        str,
    ):
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_REQUIRED",
                "idempotency_key "
                "必须是字符串",
            ),
        )

    normalized = (
        idempotency_key.strip()
    )

    if not normalized:
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_REQUIRED",
                "idempotency_key 不能为空",
            ),
        )

    if len(normalized) > 128:
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_INVALID",
                "idempotency_key "
                "长度不得超过 128",
            ),
        )

    return (
        normalized,
        None,
    )


def _validate_aware_time(
    value: datetime,
    field_name: str,
) -> dict[str, Any] | None:
    """时间必须带时区。"""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        return _error(
            "TIME_INVALID",
            f"{field_name} 必须包含时区",
        )

    return None


def _build_preview(
    event_input: HealthEventDraftInput,
) -> dict[str, Any]:
    """生成适合 UI 展示的草稿摘要。"""

    if isinstance(
        event_input,
        MealEventDraftInput,
    ):
        return {
            "event_type": "meal",
            "food": (
                event_input
                .payload
                .food
                .name
            ),
            "food_id": (
                event_input
                .payload
                .food
                .food_id
            ),
            "grams": (
                event_input
                .payload
                .portion
                .grams
            ),
            "calories_kcal": (
                event_input
                .payload
                .nutrition
                .calories_kcal
            ),
            "estimated": True,
        }

    if isinstance(
        event_input,
        WaterEventDraftInput,
    ):
        return {
            "event_type": "water",
            "amount_ml": (
                event_input.amount_ml
            ),
            "beverage": (
                event_input.beverage
            ),
            "note": event_input.note,
        }

    if isinstance(
        event_input,
        WeightEventDraftInput,
    ):
        return {
            "event_type": "weight",
            "weight_kg": (
                event_input.weight_kg
            ),
            "note": event_input.note,
        }

    return {
        "event_type": "exercise",
        "activity_type": (
            event_input.activity_type
        ),
        "duration_minutes": (
            event_input
            .duration_minutes
        ),
        "distance_km": (
            event_input.distance_km
        ),
        "intensity": (
            event_input.intensity.value
            if event_input.intensity
            is not None
            else None
        ),
        "note": event_input.note,
    }


def prepare_health_event(
    *,
    event_input: (
        HealthEventDraftInput
        | dict[str, Any]
    ),
    user_id: str,
    idempotency_key: str,
    now: datetime | None = None,
    confirmation_ttl_seconds: int = 900,
) -> dict[str, Any]:
    """
    生成四类 HealthEvent 草稿和确认令牌。

    本函数不接收 store，也不会写 JSONL。
    """

    if not isinstance(
        user_id,
        str,
    ):
        return _error(
            "USER_ID_REQUIRED",
            "user_id 必须是字符串",
        )

    normalized_user_id = (
        user_id.strip()
    )

    if not normalized_user_id:
        return _error(
            "USER_ID_REQUIRED",
            "user_id 不能为空",
        )

    (
        normalized_key,
        key_error,
    ) = _normalize_idempotency_key(
        idempotency_key
    )

    if key_error is not None:
        return key_error

    assert normalized_key is not None

    try:
        validated_input = (
            _DRAFT_INPUT_ADAPTER
            .validate_python(
                event_input
            )
        )
    except ValidationError as exc:
        return _error(
            "VALIDATION_ERROR",
            "健康事件草稿"
            f"参数校验失败：{exc}",
        )

    effective_now = (
        now
        if now is not None
        else datetime.now(
            timezone.utc
        )
    )

    time_error = (
        _validate_aware_time(
            effective_now,
            "now",
        )
    )

    if time_error is not None:
        return time_error

    occurred_at = (
        validated_input.occurred_at
        if validated_input
        .occurred_at
        is not None
        else effective_now
    )

    occurred_time_error = (
        _validate_aware_time(
            occurred_at,
            "occurred_at",
        )
    )

    if occurred_time_error is not None:
        return occurred_time_error

    if isinstance(
        validated_input,
        MealEventDraftInput,
    ):
        payload = (
            validated_input.payload
        )
        source_refs = (
            validated_input
            .source_refs
        )
        input_source = (
            validated_input
            .input_source
        )

    elif isinstance(
        validated_input,
        WaterEventDraftInput,
    ):
        payload = WaterPayload(
            amount_ml=(
                validated_input
                .amount_ml
            ),
            beverage=(
                validated_input
                .beverage
            ),
            note=validated_input.note,
        )
        source_refs = []
        input_source = (
            InputSource.CHAT
        )

    elif isinstance(
        validated_input,
        WeightEventDraftInput,
    ):
        payload = WeightPayload(
            weight_kg=(
                validated_input
                .weight_kg
            ),
            note=validated_input.note,
        )
        source_refs = []
        input_source = (
            InputSource.CHAT
        )

    else:
        payload = ExercisePayload(
            activity_type=(
                validated_input
                .activity_type
            ),
            duration_minutes=(
                validated_input
                .duration_minutes
            ),
            distance_km=(
                validated_input
                .distance_km
            ),
            intensity=(
                validated_input
                .intensity
            ),
            note=validated_input.note,
        )
        source_refs = []
        input_source = (
            InputSource.CHAT
        )

    stable_event_id = uuid5(
        NAMESPACE_URL,
        "health-event:"
        f"{normalized_user_id}:"
        f"{normalized_key}",
    )

    try:
        event = HealthEvent(
            event_id=stable_event_id,
            user_id=normalized_user_id,
            event_type=(
                validated_input
                .event_type
            ),
            occurred_at=occurred_at,
            payload=payload,
            source_refs=source_refs,
            input_source=input_source,
            created_at=effective_now,
            updated_at=effective_now,
        )
    except ValidationError as exc:
        return _error(
            "VALIDATION_ERROR",
            "生成 HealthEvent "
            f"失败：{exc}",
        )

    try:
        confirmation_token = (
            issue_confirmation_token(
                event,
                ttl_seconds=(
                    confirmation_ttl_seconds
                ),
            )
        )
    except ValueError as exc:
        return _error(
            "CONFIRMATION_DRAFT_INVALID",
            str(exc),
        )

    return {
        "ok": True,
        "data": {
            "action": "save",
            "event": (
                event.model_dump(
                    mode="json"
                )
            ),
            "preview": (
                _build_preview(
                    validated_input
                )
            ),
            "confirmation_token": (
                confirmation_token
            ),
            "idempotency_key": (
                normalized_key
            ),
            "confirmation_ttl_seconds": (
                confirmation_ttl_seconds
            ),
        },
        "error": None,
    }