"""食物候选检索工具协议封装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.nutrition.repository import FoodRepository, NutritionDataError
from src.nutrition.retrieval_trace import build_retrieval_trace
from src.storage.trace_store import TraceStore, TraceWriteError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = PROJECT_ROOT / "data" / "traces.jsonl"


def _failure(error_code: str, message: str) -> dict[str, Any]:
    """构造稳定的失败协议。"""

    return {
        "ok": False,
        "data": None,
        "error": {"error_code": error_code, "message": message},
    }


def retrieve_nutrition_candidates(
    query: Any,
    top_k: Any = 5,
    repository: FoodRepository | None = None,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """校验参数、检索候选并尽力追加 Trace。"""

    if not isinstance(query, str):
        return _failure("QUERY_INVALID", "query 必须是字符串")
    if not query.strip():
        return _failure("QUERY_REQUIRED", "query 不能为空")
    if len(query.strip()) > 64:
        return _failure("QUERY_TOO_LONG", "query 不得超过 64 个字符")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        return _failure("TOP_K_INVALID", "top_k 必须是 1—10 的整数")

    try:
        active_repository = repository or FoodRepository()
        result = active_repository.search(query, top_k=top_k)
    except NutritionDataError as exc:
        return _failure(exc.error_code, exc.message)

    trace = build_retrieval_trace(result)
    trace_warning: dict[str, str] | None = None
    try:
        (trace_store or TraceStore(DEFAULT_TRACE_PATH)).append(trace)
    except TraceWriteError as exc:
        trace_warning = {
            "error_code": exc.error_code,
            "message": exc.message,
        }

    return {
        "ok": True,
        "data": {
            **result.model_dump(mode="json"),
            "trace": trace.model_dump(mode="json"),
            "trace_warning": trace_warning,
        },
        "error": None,
    }
