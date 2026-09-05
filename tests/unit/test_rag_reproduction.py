"""RAG 一键复现入口的回归测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.build_food_index import (
    DEFAULT_MODEL_REVISION,
)
from scripts.reproduce_rag import (
    RagReproductionError,
    download_embedding_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_embedding_revision_is_pinned() -> None:
    """默认模型必须固定到不可变 commit，而不是浮动 main。"""

    assert len(DEFAULT_MODEL_REVISION) == 40
    assert all(
        character in "0123456789abcdef"
        for character in DEFAULT_MODEL_REVISION
    )


def test_offline_download_uses_exact_cached_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """离线复跑只能读取指定 revision 的本地缓存。"""

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(
        **kwargs: object,
    ) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setenv(
        "HF_HUB_OFFLINE",
        "0",
    )
    monkeypatch.setenv(
        "TRANSFORMERS_OFFLINE",
        "0",
    )

    result = download_embedding_model(
        model_name="BAAI/bge-small-zh-v1.5",
        model_revision=DEFAULT_MODEL_REVISION,
        offline=True,
    )

    assert result == tmp_path
    assert calls == [
        {
            "repo_id": "BAAI/bge-small-zh-v1.5",
            "revision": DEFAULT_MODEL_REVISION,
            "local_files_only": True,
        }
    ]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_model_download_failure_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型缺失时应返回稳定错误码和恢复提示。"""

    def fail_snapshot_download(
        **_: object,
    ) -> str:
        raise OSError("network unavailable")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fail_snapshot_download,
    )

    with pytest.raises(
        RagReproductionError
    ) as error:
        download_embedding_model(
            model_name=(
                "BAAI/bge-small-zh-v1.5"
            ),
            model_revision=(
                DEFAULT_MODEL_REVISION
            ),
            offline=False,
        )

    assert (
        error.value.error_code
        == "EMBEDDING_MODEL_DOWNLOAD_FAILED"
    )
    assert error.value.exit_code == 3
    assert "重新运行同一命令" in (
        error.value.message
    )


def test_shell_entrypoint_is_executable() -> None:
    """新环境应能直接执行单命令包装器。"""

    entrypoint = (
        PROJECT_ROOT
        / "scripts"
        / "reproduce_rag.sh"
    )
    assert entrypoint.is_file()
    assert os.access(entrypoint, os.X_OK)
