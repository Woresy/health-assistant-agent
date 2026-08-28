"""读取 committed events 并生成每日健康汇总。"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.health.daily_summary import (
    build_daily_summary,
    parse_calendar_date,
    resolve_day_window,
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


def get_daily_health_summary(
    *,
    user_id: str,
    date: str | date,
    store: HealthEventStore,
    timezone_name: str = (
        "Asia/Shanghai"
    ),
) -> dict[str, Any]:
    """
    从 JSONL 中重新读取指定日期的事件并汇总。

    没有记录时仍返回成功，events 为空，
    所有合计为 0。
    """

    normalized_user_id = (
        user_id.strip()
    )

    if not normalized_user_id:
        return _error(
            "USER_ID_REQUIRED",
            "user_id 不能为空",
        )

    try:
        selected_date = (
            parse_calendar_date(date)
        )
    except ValueError as exc:
        return _error(
            "DATE_INVALID",
            str(exc),
        )

    try:
        (
            day_start,
            day_end,
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
        events = store.query(
            user_id=normalized_user_id,
            occurred_from=day_start,
            occurred_to=day_end,
            newest_first=False,
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

    summary = build_daily_summary(
        events=events,
        summary_date=selected_date,
        timezone_name=timezone_name,
    )

    return {
        "ok": True,
        "data": {
            "events": [
                event.model_dump(
                    mode="json"
                )
                for event in events
            ],
            "summary": (
                summary.model_dump(
                    mode="json"
                )
            ),
        },
        "error": None,
    }