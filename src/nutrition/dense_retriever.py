"""本地 NumPy 向量索引和中文 Dense Retrieval。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.nutrition.repository import (
    FoodRepository,
)
from src.nutrition.retrieval_document import (
    RetrievalDocument,
)


class DenseRetrievalError(Exception):
    """Dense Retrieval 的稳定错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class IndexManifest(BaseModel):
    """向量索引的版本清单。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    index_id: str
    created_at: str

    model_name: str
    model_revision: str | None
    query_instruction: str

    dataset_id: str
    dataset_sha256: str
    document_count: int = Field(ge=1)
    embedding_dimension: int = Field(ge=1)
    embedding_dtype: str

    documents_file: str
    documents_sha256: str
    embeddings_file: str
    embeddings_sha256: str


class DenseHit(BaseModel):
    """Dense Retrieval 返回的一条命中。"""

    model_config = ConfigDict(extra="forbid")

    food_id: str = Field(min_length=1)
    score: float = Field(ge=-1, le=1)
    rank: int = Field(ge=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        while chunk := source_file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


class DenseFoodRetriever:
    """加载本地索引，并用 BGE 模型检索食物。"""

    def __init__(
        self,
        repository: FoodRepository,
        index_dir: str | Path,
    ) -> None:
        self.repository = repository
        self.index_dir = Path(index_dir)

        self._manifest: IndexManifest | None = None
        self._documents: list[
            RetrievalDocument
        ] | None = None
        self._embeddings: np.ndarray[Any, Any] | None = None
        self._model: Any | None = None

    @property
    def manifest(self) -> IndexManifest:
        self._load_index()

        assert self._manifest is not None
        return self._manifest

    def _load_index(self) -> None:
        if (
            self._manifest is not None
            and self._documents is not None
            and self._embeddings is not None
        ):
            return

        manifest_path = (
            self.index_dir
            / "index_manifest.json"
        )

        if not manifest_path.exists():
            raise DenseRetrievalError(
                "DENSE_INDEX_MISSING",
                (
                    "找不到向量索引清单："
                    f"{manifest_path}"
                ),
            )

        try:
            manifest = IndexManifest.model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            ValidationError,
        ) as exc:
            raise DenseRetrievalError(
                "DENSE_INDEX_INVALID",
                f"索引清单无效：{exc}",
            ) from exc

        if (
            manifest.dataset_sha256
            != self.repository.dataset_sha256
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_DATASET_MISMATCH",
                (
                    "向量索引对应的食物数据已经变化；"
                    "请重新运行 build_food_index.py"
                ),
            )

        documents_path = (
            self.index_dir
            / manifest.documents_file
        )
        embeddings_path = (
            self.index_dir
            / manifest.embeddings_file
        )

        if (
            not documents_path.exists()
            or not embeddings_path.exists()
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_INCOMPLETE",
                "索引文档或向量文件缺失",
            )

        if (
            _sha256_file(documents_path)
            != manifest.documents_sha256
        ):
            raise DenseRetrievalError(
                "DENSE_DOCUMENTS_HASH_MISMATCH",
                "检索文档哈希不匹配",
            )

        if (
            _sha256_file(embeddings_path)
            != manifest.embeddings_sha256
        ):
            raise DenseRetrievalError(
                "DENSE_EMBEDDINGS_HASH_MISMATCH",
                "向量文件哈希不匹配",
            )

        try:
            raw_documents = json.loads(
                documents_path.read_text(
                    encoding="utf-8"
                )
            )
            documents = [
                RetrievalDocument.model_validate(
                    item
                )
                for item in raw_documents
            ]
            embeddings = np.load(
                embeddings_path,
                allow_pickle=False,
            )
        except (
            OSError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            raise DenseRetrievalError(
                "DENSE_INDEX_INVALID",
                f"索引文件无法加载：{exc}",
            ) from exc

        if embeddings.ndim != 2:
            raise DenseRetrievalError(
                "DENSE_INDEX_SHAPE_INVALID",
                "向量矩阵必须是二维数组",
            )

        if (
            len(documents)
            != manifest.document_count
            or embeddings.shape[0]
            != manifest.document_count
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_COUNT_MISMATCH",
                "文档数量和向量数量不一致",
            )

        if (
            embeddings.shape[1]
            != manifest.embedding_dimension
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_DIMENSION_MISMATCH",
                "向量维度与索引清单不一致",
            )

        document_food_ids = {
            document.food_id
            for document in documents
        }

        if (
            document_food_ids
            != self.repository.known_food_ids
        ):
            raise DenseRetrievalError(
                "DENSE_INDEX_FOOD_IDS_MISMATCH",
                "索引 food_id 与当前数据集不一致",
            )

        self._manifest = manifest
        self._documents = documents
        self._embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

    def _get_model(self) -> Any:
        self._load_index()

        if self._model is not None:
            return self._model

        assert self._manifest is not None

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )

            kwargs: dict[str, Any] = {
                "local_files_only": True,
            }

            if self._manifest.model_revision:
                kwargs["revision"] = (
                    self._manifest.model_revision
                )

            self._model = SentenceTransformer(
                self._manifest.model_name,
                **kwargs,
            )
        except Exception as exc:
            raise DenseRetrievalError(
                "EMBEDDING_MODEL_UNAVAILABLE",
                (
                    "本地无法加载 embedding 模型；"
                    "请先运行 build_food_index.py："
                    f"{exc}"
                ),
            ) from exc

        return self._model

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[DenseHit]:
        """将查询编码为向量并返回 cosine Top-K。"""

        if not query.strip():
            raise DenseRetrievalError(
                "DENSE_QUERY_REQUIRED",
                "Dense 查询不能为空",
            )

        if not 1 <= top_k <= 50:
            raise DenseRetrievalError(
                "DENSE_TOP_K_INVALID",
                "Dense top_k 必须是 1—50",
            )

        self._load_index()
        model = self._get_model()

        assert self._manifest is not None
        assert self._documents is not None
        assert self._embeddings is not None

        query_text = (
            self._manifest.query_instruction
            + query.strip()
        )

        try:
            query_embedding = model.encode_query(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise DenseRetrievalError(
                "QUERY_EMBEDDING_FAILED",
                f"查询向量生成失败：{exc}",
            ) from exc

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        if query_vector.ndim != 1:
            raise DenseRetrievalError(
                "QUERY_EMBEDDING_SHAPE_INVALID",
                "查询向量必须是一维",
            )

        if (
            query_vector.shape[0]
            != self._embeddings.shape[1]
        ):
            raise DenseRetrievalError(
                "QUERY_EMBEDDING_DIMENSION_MISMATCH",
                "查询向量维度与文档索引不同",
            )

        scores = self._embeddings @ query_vector

        count = min(
            top_k,
            len(self._documents),
        )

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:count]

        return [
            DenseHit(
                food_id=(
                    self._documents[
                        int(index)
                    ].food_id
                ),
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(
                ranked_indices,
                start=1,
            )
        ]