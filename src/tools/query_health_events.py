"""查询已保存健康事件的工具。"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.health.daily_summary import (
    parse_calendar_date,
    resolve_day_window,
)
from src.health.models import (
    EventType,
)
from src.storage.jsonl_store import (
    HealthEventStore,
    JsonlReadError,
)


def _error(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """返回统一工具错误结构。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def query_health_events(
    *,
    user_id: str,
    store: HealthEventStore,
    event_type: str | None = None,
    date: str | date | None = None,
    timezone_name: str = (
        "Asia/Shanghai"
    ),
    newest_first: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """
    查询已经保存的健康事件。

    可以按日期和事件类型过滤。
    不传 date 时查询该用户的全部事件。
    """

    normalized_user_id = (
        user_id.strip()
    )

    if not normalized_user_id:
        return _error(
            "USER_ID_REQUIRED",
            "user_id 不能为空",
        )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > 500
    ):
        return _error(
            "LIMIT_INVALID",
            "limit 必须是 1 到 500 "
            "之间的整数",
        )

    normalized_event_type: (
        EventType
        | None
    ) = None

    if event_type is not None:
        normalized_value = (
            event_type.strip()
        )

        if not normalized_value:
            return _error(
                "EVENT_TYPE_INVALID",
                "event_type 不能为空字符串",
            )

        try:
            normalized_event_type = (
                EventType(
                    normalized_value
                )
            )
        except ValueError:
            return _error(
                "EVENT_TYPE_INVALID",
                "event_type 必须是 "
                "meal、water、weight "
                "或 exercise",
            )

    selected_date: (
        date
        | None
    ) = None
    occurred_from = None
    occurred_to = None

    if date is not None:
        try:
            selected_date = (
                parse_calendar_date(
                    date
                )
            )
        except ValueError as exc:
            return _error(
                "DATE_INVALID",
                str(exc),
            )

        try:
            (
                occurred_from,
                occurred_to,
                _,
            ) = resolve_day_window(
                selected_date,
                timezone_name,
            )
        except ValueError as exc:
            return _error(
                "TIMEZONE_INVALID",
                str(exc),
            )

    try:
        matched_events = store.query(
            user_id=normalized_user_id,
            event_type=(
                normalized_event_type
            ),
            occurred_from=(
                occurred_from
            ),
            occurred_to=(
                occurred_to
            ),
            newest_first=(
                newest_first
            ),
        )
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )
    except ValueError as exc:
        return _error(
            "VALIDATION_ERROR",
            str(exc),
        )

    returned_events = (
        matched_events[:limit]
    )

    return {
        "ok": True,
        "data": {
            "events": [
                event.model_dump(
                    mode="json"
                )
                for event
                in returned_events
            ],
            "matched_count": len(
                matched_events
            ),
            "returned_count": len(
                returned_events
            ),
            "filters": {
                "user_id": (
                    normalized_user_id
                ),
                "event_type": (
                    normalized_event_type
                    .value
                    if normalized_event_type
                    is not None
                    else None
                ),
                "date": (
                    selected_date
                    .isoformat()
                    if selected_date
                    is not None
                    else None
                ),
                "timezone": (
                    timezone_name
                    if selected_date
                    is not None
                    else None
                ),
                "newest_first": (
                    newest_first
                ),
                "limit": limit,
            },
        },
        "error": None,
    }