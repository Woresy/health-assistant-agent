"""根据已经保存的 HealthEvent 生成确定性每日汇总。"""

from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from uuid import UUID
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from src.health.models import (
    ExercisePayload,
    HealthEvent,
    MealPayload,
    WaterPayload,
    WeightPayload,
)


class SummaryModel(BaseModel):
    """每日汇总模型的公共配置。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class MealDailySummary(
    SummaryModel
):
    """饮食事件汇总。"""

    count: int = Field(ge=0)
    calories_kcal: float = Field(
        ge=0
    )
    protein_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)


class WaterDailySummary(
    SummaryModel
):
    """饮水事件汇总。"""

    count: int = Field(ge=0)
    total_ml: float = Field(ge=0)


class WeightDailySummary(
    SummaryModel
):
    """体重事件汇总。"""

    count: int = Field(ge=0)
    latest_weight_kg: (
        float
        | None
    ) = None
    latest_occurred_at: (
        datetime
        | None
    ) = None


class ExerciseDailySummary(
    SummaryModel
):
    """运动事件汇总。"""

    count: int = Field(ge=0)
    total_duration_minutes: (
        float
    ) = Field(ge=0)
    total_distance_km: float = (
        Field(ge=0)
    )


class DailyHealthSummary(
    SummaryModel
):
    """四类健康事件的每日确定性汇总。"""

    summary_date: date
    timezone: str
    event_count: int = Field(ge=0)
    event_ids: list[UUID]

    meal: MealDailySummary
    water: WaterDailySummary
    weight: WeightDailySummary
    exercise: ExerciseDailySummary


def parse_calendar_date(
    value: str | date,
) -> date:
    """把 YYYY-MM-DD 或 date 转换为日期。"""

    if isinstance(value, datetime):
        raise ValueError(
            "日期不能包含具体时间"
        )

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        raise ValueError(
            "日期必须是 YYYY-MM-DD"
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "日期不能为空"
        )

    try:
        return date.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise ValueError(
            "日期必须是有效的 "
            "YYYY-MM-DD"
        ) from exc


def resolve_day_window(
    summary_date: date,
    timezone_name: str,
) -> tuple[
    datetime,
    datetime,
    ZoneInfo,
]:
    """生成指定本地日期的开始时间和结束时间。"""

    normalized_timezone = (
        timezone_name.strip()
    )

    if not normalized_timezone:
        raise ValueError(
            "timezone_name 不能为空"
        )

    try:
        local_timezone = ZoneInfo(
            normalized_timezone
        )
    except (
        ZoneInfoNotFoundError,
        ValueError,
    ) as exc:
        raise ValueError(
            "无法加载时区："
            f"{normalized_timezone}"
        ) from exc

    day_start = datetime.combine(
        summary_date,
        time.min,
        tzinfo=local_timezone,
    )

    next_date = (
        summary_date
        + timedelta(days=1)
    )

    day_end = datetime.combine(
        next_date,
        time.min,
        tzinfo=local_timezone,
    )

    return (
        day_start,
        day_end,
        local_timezone,
    )


def _rounded(
    value: float,
) -> float:
    """减少浮点累加产生的无意义尾数。"""

    return round(
        float(value),
        6,
    )


def build_daily_summary(
    *,
    events: list[HealthEvent],
    summary_date: date,
    timezone_name: str,
) -> DailyHealthSummary:
    """
    从已保存事件生成汇总。

    本函数不调用模型，也不重新估算营养值。
    所有数值都来自 HealthEvent payload。
    """

    (
        _,
        _,
        local_timezone,
    ) = resolve_day_window(
        summary_date,
        timezone_name,
    )

    selected_events = [
        event
        for event in events
        if (
            event.occurred_at
            .astimezone(
                local_timezone
            )
            .date()
            == summary_date
        )
    ]

    selected_events.sort(
        key=lambda event: (
            event.occurred_at,
            event.created_at,
            str(event.event_id),
        )
    )

    meal_count = 0
    calories_kcal = 0.0
    protein_g = 0.0
    fat_g = 0.0
    carbs_g = 0.0

    water_count = 0
    total_water_ml = 0.0

    weight_count = 0
    latest_weight_event: (
        HealthEvent
        | None
    ) = None

    exercise_count = 0
    total_duration_minutes = 0.0
    total_distance_km = 0.0

    for event in selected_events:
        payload = event.payload

        if isinstance(
            payload,
            MealPayload,
        ):
            meal_count += 1

            calories_kcal += (
                payload
                .nutrition
                .calories_kcal
            )
            protein_g += (
                payload
                .nutrition
                .protein_g
            )
            fat_g += (
                payload
                .nutrition
                .fat_g
            )
            carbs_g += (
                payload
                .nutrition
                .carbs_g
            )

        elif isinstance(
            payload,
            WaterPayload,
        ):
            water_count += 1
            total_water_ml += (
                payload.amount_ml
            )

        elif isinstance(
            payload,
            WeightPayload,
        ):
            weight_count += 1

            if (
                latest_weight_event
                is None
                or (
                    event.occurred_at,
                    event.created_at,
                    str(event.event_id),
                )
                > (
                    latest_weight_event
                    .occurred_at,
                    latest_weight_event
                    .created_at,
                    str(
                        latest_weight_event
                        .event_id
                    ),
                )
            ):
                latest_weight_event = (
                    event
                )

        elif isinstance(
            payload,
            ExercisePayload,
        ):
            exercise_count += 1

            total_duration_minutes += (
                payload
                .duration_minutes
            )

            if (
                payload.distance_km
                is not None
            ):
                total_distance_km += (
                    payload.distance_km
                )

    latest_weight_kg: (
        float
        | None
    ) = None
    latest_weight_time: (
        datetime
        | None
    ) = None

    if (
        latest_weight_event
        is not None
        and isinstance(
            latest_weight_event.payload,
            WeightPayload,
        )
    ):
        latest_weight_kg = (
            latest_weight_event
            .payload
            .weight_kg
        )
        latest_weight_time = (
            latest_weight_event
            .occurred_at
            .astimezone(
                local_timezone
            )
        )

    return DailyHealthSummary(
        summary_date=summary_date,
        timezone=timezone_name,
        event_count=len(
            selected_events
        ),
        event_ids=[
            event.event_id
            for event in selected_events
        ],
        meal=MealDailySummary(
            count=meal_count,
            calories_kcal=_rounded(
                calories_kcal
            ),
            protein_g=_rounded(
                protein_g
            ),
            fat_g=_rounded(
                fat_g
            ),
            carbs_g=_rounded(
                carbs_g
            ),
        ),
        water=WaterDailySummary(
            count=water_count,
            total_ml=_rounded(
                total_water_ml
            ),
        ),
        weight=WeightDailySummary(
            count=weight_count,
            latest_weight_kg=(
                latest_weight_kg
            ),
            latest_occurred_at=(
                latest_weight_time
            ),
        ),
        exercise=(
            ExerciseDailySummary(
                count=exercise_count,
                total_duration_minutes=(
                    _rounded(
                        total_duration_minutes
                    )
                ),
                total_distance_km=(
                    _rounded(
                        total_distance_km
                    )
                ),
            )
        ),
    )