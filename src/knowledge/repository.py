"""本地、可审计的健康知识 Hybrid RAG 与安全门控。

健康知识先经过确定性安全分类，再执行词法召回和本地 Dense 语义召回。
两路结果使用 RRF 融合，并通过显式 min_score、领域信号和候选分差门控。
向量只负责检索可信文档；最终回答仍由 Agent 基于带来源的上下文生成。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "samples" / "health_knowledge.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "knowledge_index"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 面向 Agent 的最终候选数默认是 3；词法与 Dense 各自先扩大召回池，再融合重排。
DEFAULT_TOP_K = 3
DEFAULT_POOL_SIZE = 8
DEFAULT_MIN_SCORE = 0.44
DEFAULT_MIN_MARGIN = 0.025
DEFAULT_RRF_K = 10
LEXICAL_RRF_WEIGHT = 0.15
DENSE_RRF_WEIGHT = 0.85

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
HEALTH_DOMAIN_MARKERS = (
    "健康", "身体", "运动", "活动", "动起来", "坐着", "锻炼", "训练", "久坐", "有氧", "肌肉", "力量", "抗阻",
    "饮食", "吃", "搭配", "偏科", "蔬菜", "水果", "全谷", "营养", "食物", "外卖", "口味", "少盐", "盐",
    "钠", "喝水", "饮水", "补水", "口渴", "饮料", "睡", "熬夜", "休息",
)


class QueryEmbedder(Protocol):
    """Dense 查询编码器协议，便于测试注入确定性向量。"""

    def encode_query(self, sentences: Any, **kwargs: Any) -> Any:
        ...


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

    def retrieval_text(self) -> str:
        """构建不含指令的、可重建的向量检索文档。"""

        return (
            f"标题：{self.title}。主题：{'、'.join(self.topics)}。"
            f"内容：{self.content}。适用范围：{self.scope}。"
        )


class KnowledgeIndexManifest(BaseModel):
    """健康知识向量索引的可验证版本清单。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    index_id: str
    created_at: str
    model_name: str
    model_revision: str | None
    query_instruction: str
    dataset_sha256: str
    document_count: int = Field(ge=1)
    embedding_dimension: int = Field(ge=1)
    embedding_dtype: str
    document_ids: tuple[str, ...]
    embeddings_file: str
    embeddings_sha256: str


class HealthKnowledgeRepository:
    """对可信健康文档执行安全优先的本地 Hybrid RAG。"""

    _model_cache: dict[tuple[str, str | None], QueryEmbedder] = {}

    def __init__(
        self,
        path: str | Path = DEFAULT_KNOWLEDGE_PATH,
        *,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        embedder: QueryEmbedder | None = None,
        embeddings: np.ndarray[Any, Any] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        pool_size: int = DEFAULT_POOL_SIZE,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.path = Path(path)
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self._embeddings = embeddings
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.pool_size = int(pool_size)
        self.rrf_k = int(rrf_k)
        self._manifest: KnowledgeIndexManifest | None = None

        if not 0 <= self.min_score <= 1:
            raise ValueError("健康知识 min_score 必须在 0—1 之间")
        if not 0 <= self.min_margin <= 1:
            raise ValueError("健康知识 min_margin 必须在 0—1 之间")
        if not 1 <= self.pool_size <= 50:
            raise ValueError("健康知识 pool_size 必须在 1—50 之间")
        if self.rrf_k <= 0:
            raise ValueError("健康知识 rrf_k 必须大于 0")

        try:
            source_bytes = self.path.read_bytes()
            payload = json.loads(source_bytes.decode("utf-8"))
            self.documents = tuple(KnowledgeDocument.model_validate(item) for item in payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ValueError(f"健康知识库无法读取：{exc}") from exc

        ids = [item.document_id for item in self.documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("健康知识库必须包含唯一 document_id")
        self.dataset_sha256 = hashlib.sha256(source_bytes).hexdigest()

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

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _trusted_documents(self) -> tuple[KnowledgeDocument, ...]:
        return tuple(
            document for document in self.documents
            if not self._document_is_untrusted(document)
        )

    def _lexical_ranking(self, question: str) -> list[tuple[str, float]]:
        normalized = question.strip().casefold()
        scored: list[tuple[str, float]] = []
        for document in self._trusted_documents():
            topic_hits = sum(1 for topic in document.topics if topic.casefold() in normalized)
            if topic_hits:
                score = min(1.0, 0.68 + 0.08 * (topic_hits - 1))
                scored.append((document.document_id, round(score, 6)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: self.pool_size]

    def _load_dense_index(self) -> bool:
        if self._embeddings is not None:
            return self.embedder is not None or self._manifest is not None

        manifest_path = self.index_dir / "knowledge_index_manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = KnowledgeIndexManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            embeddings_path = self.index_dir / manifest.embeddings_file
            if manifest.schema_version != "1.0":
                return False
            if manifest.dataset_sha256 != self.dataset_sha256:
                return False
            if not embeddings_path.exists():
                return False
            if self._sha256_file(embeddings_path) != manifest.embeddings_sha256:
                return False
            trusted_ids = tuple(document.document_id for document in self._trusted_documents())
            if manifest.document_ids != trusted_ids:
                return False
            embeddings = np.load(embeddings_path, allow_pickle=False).astype(np.float32, copy=False)
            if embeddings.shape != (manifest.document_count, manifest.embedding_dimension):
                return False
        except (OSError, ValueError, ValidationError, json.JSONDecodeError):
            return False

        self._manifest = manifest
        self._embeddings = embeddings
        return True

    def _get_embedder(self) -> QueryEmbedder | None:
        if self.embedder is not None:
            return self.embedder
        if self._manifest is None:
            return None
        key = (self._manifest.model_name, self._manifest.model_revision)
        if key in self._model_cache:
            return self._model_cache[key]
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"local_files_only": True}
            if self._manifest.model_revision:
                kwargs["revision"] = self._manifest.model_revision
            model = SentenceTransformer(self._manifest.model_name, **kwargs)
        except Exception:
            return None
        self._model_cache[key] = model
        return model

    def _dense_ranking(self, question: str) -> list[tuple[str, float]]:
        if not self._load_dense_index():
            return []
        embedder = self._get_embedder()
        if embedder is None or self._embeddings is None:
            return []
        instruction = (
            self._manifest.query_instruction
            if self._manifest is not None
            else DEFAULT_QUERY_INSTRUCTION
        )
        try:
            encoded = embedder.encode_query(
                instruction + question.strip(),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            return []
        query_vector = np.asarray(encoded, dtype=np.float32)
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        if query_vector.ndim != 1 or query_vector.shape[0] != self._embeddings.shape[1]:
            return []
        scores = self._embeddings @ query_vector
        document_ids = (
            self._manifest.document_ids
            if self._manifest is not None
            else tuple(document.document_id for document in self._trusted_documents())
        )
        indices = np.argsort(-scores, kind="stable")[: min(self.pool_size, len(scores))]
        return [(document_ids[int(index)], float(scores[int(index)])) for index in indices]

    def retrieval_receipt(self) -> dict[str, Any]:
        """返回不含问题和文档正文的检索配置凭据。"""

        dense_ready = self._load_dense_index()
        return {
            "mode": "hybrid" if dense_ready else "lexical_fallback",
            "vector_store": "local_numpy_exact_cosine" if dense_ready else None,
            "embedding_model": self._manifest.model_name if self._manifest else None,
            "index_id": self._manifest.index_id if self._manifest else None,
            "pool_size": self.pool_size,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "rrf_k": self.rrf_k,
        }

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """执行 Top-K Hybrid Search；证据不足时返回空列表。"""

        if not 1 <= top_k <= 10:
            raise ValueError("健康知识 top_k 必须在 1—10 之间")
        active_min_score = self.min_score if min_score is None else float(min_score)
        if not 0 <= active_min_score <= 1:
            raise ValueError("健康知识 min_score 必须在 0—1 之间")

        normalized = question.strip().casefold()
        lexical = self._lexical_ranking(normalized)
        dense = self._dense_ranking(normalized)
        lexical_by_id = {document_id: (rank, score) for rank, (document_id, score) in enumerate(lexical, 1)}
        dense_by_id = {document_id: (rank, score) for rank, (document_id, score) in enumerate(dense, 1)}
        documents_by_id = {document.document_id: document for document in self._trusted_documents()}

        fused: list[tuple[float, str, int | None, int | None, float | None, float | None]] = []
        for document_id in set(lexical_by_id) | set(dense_by_id):
            lexical_item = lexical_by_id.get(document_id)
            dense_item = dense_by_id.get(document_id)
            lexical_rank = lexical_item[0] if lexical_item else None
            dense_rank = dense_item[0] if dense_item else None
            dense_available = bool(dense)
            lexical_weight = LEXICAL_RRF_WEIGHT if dense_available else 1.0
            dense_weight = DENSE_RRF_WEIGHT if dense_available else 0.0
            raw_rrf = (
                (lexical_weight / (self.rrf_k + lexical_rank) if lexical_rank else 0.0)
                + (dense_weight / (self.rrf_k + dense_rank) if dense_rank else 0.0)
            )
            maximum = (lexical_weight + dense_weight) / (self.rrf_k + 1)
            fused.append((
                raw_rrf / maximum,
                document_id,
                lexical_rank,
                dense_rank,
                lexical_item[1] if lexical_item else None,
                dense_item[1] if dense_item else None,
            ))
        fused.sort(key=lambda item: (-item[0], item[1]))
        if not fused:
            return []

        has_domain_signal = any(marker in normalized for marker in HEALTH_DOMAIN_MARKERS)
        top = fused[0]
        top_dense_score = top[5]
        exact_topic = top[2] is not None
        dense_allowed = top_dense_score is not None and top_dense_score >= active_min_score
        competing_dense_scores = [item[5] for item in fused[1:] if item[5] is not None]
        dense_margin = (
            top_dense_score - max(competing_dense_scores)
            if top_dense_score is not None and competing_dense_scores
            else None
        )
        if not exact_topic and (
            not has_domain_signal
            or not dense_allowed
            or (dense_margin is not None and dense_margin < self.min_margin)
        ):
            return []

        strategies = ["lexical_topics"]
        if dense:
            strategies.extend(["dense_cosine", "rrf", "min_score_gate"])
        else:
            strategies.append("lexical_fallback")

        candidate_dense_floor = (
            max(active_min_score, top_dense_score - 0.08)
            if top_dense_score is not None
            else active_min_score
        )
        eligible = [
            item for item in fused
            if (
                item[5] is not None and item[5] >= candidate_dense_floor
                if dense
                else item[2] is not None
            )
        ]

        results: list[dict[str, Any]] = []
        for rrf_score, document_id, lexical_rank, dense_rank, lexical_score, dense_score in eligible[:top_k]:
            document = documents_by_id[document_id]
            results.append({
                **document.model_dump(mode="json"),
                "score": round(rrf_score, 6),
                "lexical_score": lexical_score,
                "dense_score": round(dense_score, 6) if dense_score is not None else None,
                "lexical_rank": lexical_rank,
                "dense_rank": dense_rank,
                "strategies_used": strategies,
                "min_score": active_min_score,
                "candidate_source": "trusted_local_knowledge",
            })
        return results
