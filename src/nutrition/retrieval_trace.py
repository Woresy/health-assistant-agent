"""检索证据 Trace 模型与构建函数。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.nutrition.repository import FoodCandidate, SearchResult


class RetrievalTrace(BaseModel):
    """一次确定性检索的可序列化证据。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: UUID
    created_at: datetime
    status: Literal["ok", "not_found"]
    query: str
    normalized_query: str
    top_k: int = Field(ge=1, le=10)
    candidates: list[FoodCandidate]
    auto_select_allowed: bool
    selection_mode: Literal["auto", "user_required"]
    strategies_used: list[str]
    dataset_id: str
    dataset_record_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)


def build_retrieval_trace(result: SearchResult) -> RetrievalTrace:
    """从完整检索返回构建独立 Trace，不写入 HealthEvent。"""

    return RetrievalTrace(
        trace_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        **result.model_dump(),
    )
