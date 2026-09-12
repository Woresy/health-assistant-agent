"""为可信健康知识构建可校验的本地 NumPy 向量索引。"""

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

from src.knowledge.repository import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_KNOWLEDGE_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_INSTRUCTION,
    HealthKnowledgeRepository,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_numpy_atomic(path: Path, array: np.ndarray[Any, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, array, allow_pickle=False)
    os.replace(temporary, path)


def build_index(
    *,
    data_path: Path,
    index_dir: Path,
    model_name: str,
    model_revision: str | None,
    query_instruction: str,
    local_files_only: bool = False,
) -> dict[str, Any]:
    repository = HealthKnowledgeRepository(data_path, index_dir=index_dir)
    documents = repository._trusted_documents()  # 索引构建与运行时使用同一信任过滤器。
    if not documents:
        raise ValueError("没有可信健康知识文档可供索引")

    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if model_revision:
        kwargs["revision"] = model_revision
    model = SentenceTransformer(model_name, **kwargs)
    embeddings = model.encode_document(
        [document.retrieval_text() for document in documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(documents):
        raise ValueError("健康知识文档数与 embedding 矩阵不一致")

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "knowledge_embeddings.npy"
    manifest_path = index_dir / "knowledge_index_manifest.json"
    _write_numpy_atomic(embeddings_path, matrix)
    embeddings_sha256 = _sha256_file(embeddings_path)
    identity = (
        f"{repository.dataset_sha256}:{model_name}:{model_revision or 'default'}:"
        f"{embeddings_sha256}"
    )
    manifest = {
        "schema_version": "1.0",
        "index_id": "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_revision": model_revision,
        "query_instruction": query_instruction,
        "dataset_sha256": repository.dataset_sha256,
        "document_count": len(documents),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_dtype": str(matrix.dtype),
        "document_ids": [document.document_id for document in documents],
        "embeddings_file": embeddings_path.name,
        "embeddings_sha256": embeddings_sha256,
    }
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_KNOWLEDGE_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_index(
        data_path=args.data,
        index_dir=args.index_dir,
        model_name=args.model,
        model_revision=args.revision or None,
        query_instruction=args.query_instruction,
        local_files_only=args.offline,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
