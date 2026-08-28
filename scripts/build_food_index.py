"""为食物检索文档构建本地 NumPy 向量索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.nutrition.repository import (  # noqa: E402
    FoodRepository,
)
from src.nutrition.retrieval_document import (  # noqa: E402
    build_retrieval_documents,
    load_retrieval_hints,
)


DEFAULT_MODEL_NAME = (
    "BAAI/bge-small-zh-v1.5"
)
DEFAULT_QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        while chunk := source_file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _write_text_atomic(
    path: Path,
    content: str,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        content,
        encoding="utf-8",
    )
    os.replace(
        temporary_path,
        path,
    )


def _write_numpy_atomic(
    path: Path,
    array: np.ndarray[Any, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as target:
        np.save(
            target,
            array,
            allow_pickle=False,
        )

    os.replace(
        temporary_path,
        path,
    )


def build_index(
    *,
    data_path: Path,
    index_dir: Path,
    hints_path: Path | None,
    model_name: str,
    model_revision: str | None,
    query_instruction: str,
) -> dict[str, Any]:
    """加载食物、生成语义文档并构建索引。"""

    repository = FoodRepository(
        data_path=data_path,
        rag_mode="lexical",
    )

    hints_by_food_id: dict[
        str,
        list[str],
    ] = {}

    if hints_path is not None:
        hints_by_food_id = (
            load_retrieval_hints(
                hints_path
            )
        )

    documents = build_retrieval_documents(
        repository.foods,
        hints_by_food_id=hints_by_food_id,
    )

    if not documents:
        raise ValueError(
            "没有可用于构建索引的文档"
        )

    from sentence_transformers import (
        SentenceTransformer,
    )

    model_kwargs: dict[str, Any] = {}

    if model_revision:
        model_kwargs["revision"] = (
            model_revision
        )

    model = SentenceTransformer(
        model_name,
        **model_kwargs,
    )

    document_texts = [
        document.text
        for document in documents
    ]

    embeddings = model.encode_document(
        document_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embedding_matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embedding_matrix.ndim != 2:
        raise ValueError(
            "文档 embedding 必须是二维矩阵"
        )

    if (
        embedding_matrix.shape[0]
        != len(documents)
    ):
        raise ValueError(
            "文档数量与 embedding 数量不一致"
        )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents_path = (
        index_dir
        / "food_documents.json"
    )
    embeddings_path = (
        index_dir
        / "food_embeddings.npy"
    )
    manifest_path = (
        index_dir
        / "index_manifest.json"
    )

    documents_text = (
        json.dumps(
            [
                document.model_dump(
                    mode="json"
                )
                for document in documents
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    _write_text_atomic(
        documents_path,
        documents_text,
    )
    _write_numpy_atomic(
        embeddings_path,
        embedding_matrix,
    )

    documents_sha256 = _sha256_file(
        documents_path
    )
    embeddings_sha256 = _sha256_file(
        embeddings_path
    )

    index_identity = (
        f"{repository.dataset_sha256}:"
        f"{model_name}:"
        f"{model_revision or 'default'}:"
        f"{documents_sha256}:"
        f"{embeddings_sha256}"
    )
    index_id = hashlib.sha256(
        index_identity.encode("utf-8")
    ).hexdigest()

    matched_hint_food_count = sum(
        int(
            bool(
                hints_by_food_id.get(
                    document.food_id
                )
            )
        )
        for document in documents
    )

    manifest = {
        "schema_version": "1.0",
        "index_id": (
            f"sha256:{index_id}"
        ),
        "created_at": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "model_name": model_name,
        "model_revision": model_revision,
        "query_instruction": (
            query_instruction
        ),
        "dataset_id": (
            repository.dataset_id
        ),
        "dataset_sha256": (
            repository.dataset_sha256
        ),
        "document_count": len(documents),
        "embedding_dimension": int(
            embedding_matrix.shape[1]
        ),
        "embedding_dtype": str(
            embedding_matrix.dtype
        ),
        "documents_file": (
            documents_path.name
        ),
        "documents_sha256": (
            documents_sha256
        ),
        "embeddings_file": (
            embeddings_path.name
        ),
        "embeddings_sha256": (
            embeddings_sha256
        )
    }

    _write_text_atomic(
        manifest_path,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    print(
        (
            "已为 "
            f"{matched_hint_food_count} "
            "条食物应用人工语义提示。"
        )
    )

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "为食物数据构建中文 Dense "
            "Retrieval 索引"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "foods_sample.json"
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "index"
        ),
    )
    parser.add_argument(
        "--hints",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "retrieval_hints.json"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--model-revision",
        default=None,
    )
    parser.add_argument(
        "--query-instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
    )

    args = parser.parse_args()

    manifest = build_index(
        data_path=args.data,
        index_dir=args.index_dir,
        hints_path=args.hints,
        model_name=args.model_name,
        model_revision=args.model_revision,
        query_instruction=(
            args.query_instruction
        ),
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())