"""本地、可审计的健康知识检索与安全门控。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "samples" / "health_knowledge.json"

URGENT_TERMS = (
    "胸痛", "呼吸困难", "喘不上气", "晕厥", "昏倒", "大量出血", "抽搐",
    "自杀", "自伤", "不想活", "急救", "emergency",
)
MEDICAL_TERMS = (
    "诊断", "是不是得了", "什么病", "药", "用药", "剂量", "处方", "停药",
    "怀孕", "孕妇", "哺乳", "未成年人", "儿童", "小孩", "进食障碍",
    "催吐", "暴食", "厌食", "严重低体重", "过敏", "胰岛素",
)
INJECTION_TERMS = (
    "忽略之前", "忽略系统", "system prompt", "developer message", "越过规则",
    "调用删除工具", "泄露密钥", "执行外部动作",
)


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    topics: tuple[str, ...] = Field(min_length=1, max_length=30)
    content: str = Field(min_length=1, max_length=1200)
    source_url: HttpUrl
    source_organization: str = Field(min_length=1, max_length=120)
    updated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    scope: str = Field(min_length=1, max_length=300)


class HealthKnowledgeRepository:
    """对小型权威知识集执行确定性 Top-K 检索。"""

    def __init__(self, path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> None:
        self.path = Path(path)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.documents = tuple(KnowledgeDocument.model_validate(item) for item in payload)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ValueError(f"健康知识库无法读取：{exc}") from exc
        ids = [item.document_id for item in self.documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("健康知识库必须包含唯一 document_id")

    @staticmethod
    def classify_safety(question: str) -> str | None:
        normalized = question.strip().casefold()
        if any(term in normalized for term in INJECTION_TERMS):
            return "PROMPT_INJECTION_DETECTED"
        if any(term in normalized for term in URGENT_TERMS):
            return "URGENT_HELP_REQUIRED"
        if any(term in normalized for term in MEDICAL_TERMS):
            return "MEDICAL_BOUNDARY"
        return None

    @staticmethod
    def _document_is_untrusted(document: KnowledgeDocument) -> bool:
        normalized = f"{document.title} {document.content}".casefold()
        return any(term in normalized for term in INJECTION_TERMS)

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        normalized = question.strip().casefold()
        scored: list[tuple[int, KnowledgeDocument]] = []
        for document in self.documents:
            if self._document_is_untrusted(document):
                continue
            score = sum(1 for topic in document.topics if topic.casefold() in normalized)
            if score:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return [
            {
                **document.model_dump(mode="json"),
                "score": score,
                "candidate_source": "trusted_local_knowledge",
            }
            for score, document in scored[:top_k]
        ]
