"""准备健康知识 Hybrid RAG 所需的查询编码器。

文档向量随仓库提交，查询编码器不提交；缺少它时 Dense 召回会静默降级为词法检索，
语义改写问题会直接拿到 KNOWLEDGE_NOT_FOUND。本脚本按索引 manifest 固定的模型和
revision 把编码器准备到本地缓存，并校验它与已提交向量维度一致。

用法：

    python scripts/prepare_health_knowledge_model.py            # 下载并校验
    python scripts/prepare_health_knowledge_model.py --emit-pin # 只打印固定值

`--emit-pin` 只依赖标准库，可以在安装项目依赖之前运行，用于 CI 缓存键。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "knowledge_index" / "knowledge_index_manifest.json"
)
REPOSITORY_SOURCE_PATH = PROJECT_ROOT / "src" / "knowledge" / "repository.py"
PROBE_QUESTION = "久坐之后怎么活动"


def _constant_from_source(name: str) -> str:
    """在不导入第三方依赖的前提下读取 repository.py 中的字符串常量。"""

    source = REPOSITORY_SOURCE_PATH.read_text(encoding="utf-8")
    match = re.search(rf'^{name} = "([^"]*)"', source, re.MULTILINE)
    if match is None:
        raise ValueError(f"在 {REPOSITORY_SOURCE_PATH.name} 中找不到常量 {name}")
    return match.group(1)


def load_pin(manifest_path: Path) -> dict[str, Any]:
    """返回查询编码器固定值；manifest 不可读时回退到运行时默认常量。

    运行时按 manifest 的 model_name / model_revision 加载编码器，因此 manifest 是
    固定值的第一来源。
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pin = {
            "model_name": str(manifest["model_name"]),
            "model_revision": str(manifest.get("model_revision") or ""),
            "query_instruction": str(manifest.get("query_instruction") or ""),
            "embedding_dimension": manifest.get("embedding_dimension"),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pin = {
            "model_name": _constant_from_source("DEFAULT_MODEL_NAME"),
            "model_revision": _constant_from_source("DEFAULT_MODEL_REVISION"),
            "query_instruction": _constant_from_source("DEFAULT_QUERY_INSTRUCTION"),
            "embedding_dimension": None,
        }

    if not pin["model_name"]:
        raise ValueError("健康知识查询模型名称不能为空")
    return pin


def cache_key(pin: dict[str, Any]) -> str:
    """构造只含缓存键安全字符的固定值标识。"""

    identity = f"{pin['model_name']}-{pin['model_revision'] or 'default'}"
    return re.sub(r"[^A-Za-z0-9._-]", "-", identity)


def emit_pin(pin: dict[str, Any]) -> None:
    """以 GitHub Actions 的 key=value 形式打印固定值。"""

    sys.stdout.write(
        f"model_name={pin['model_name']}\n"
        f"model_revision={pin['model_revision']}\n"
        f"cache_key={cache_key(pin)}\n"
    )


def prepare_model(pin: dict[str, Any], *, offline: bool) -> int:
    """加载查询编码器并校验它能产生与已提交向量一致的查询向量。"""

    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"local_files_only": offline}
    if pin["model_revision"]:
        kwargs["revision"] = pin["model_revision"]
    model = SentenceTransformer(pin["model_name"], **kwargs)

    vector = model.encode_query(
        pin["query_instruction"] + PROBE_QUESTION,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dimension = int(vector.shape[-1])
    expected = pin["embedding_dimension"]
    if expected is not None and dimension != int(expected):
        raise ValueError(
            f"查询向量维度 {dimension} 与已提交索引的 {expected} 不一致："
            "请用同一模型重建 data/knowledge_index/"
        )

    print(
        f"health knowledge query encoder ready: {pin['model_name']}"
        f"@{pin['model_revision'] or 'default'} dim={dimension}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--emit-pin", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pin = load_pin(args.manifest)
    if args.emit_pin:
        emit_pin(pin)
        return 0
    return prepare_model(pin, offline=args.offline)


if __name__ == "__main__":
    raise SystemExit(main())
