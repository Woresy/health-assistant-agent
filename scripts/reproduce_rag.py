"""下载固定 embedding 模型、重建索引并运行 RAG 评测。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# 固定评测默认使用 CPU，避免 CUDA 驱动和硬件差异影响新环境复现。
# 显式设置 CUDA_VISIBLE_DEVICES 的使用者仍可覆盖此默认值。
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    "",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.build_food_index import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_INSTRUCTION,
    build_index,
)
from src.nutrition.evaluation import (  # noqa: E402
    EvaluationError,
    evaluate_retrieval,
    load_evaluation_cases,
)
from src.nutrition.repository import (  # noqa: E402
    FoodRepository,
    NutritionDataError,
)


class RagReproductionError(Exception):
    """RAG 复现步骤失败，并携带稳定错误码。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        exit_code: int = 2,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def download_embedding_model(
    *,
    model_name: str,
    model_revision: str,
    offline: bool,
) -> Path:
    """下载固定 revision，或在离线模式验证本地缓存。"""

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from huggingface_hub import (
            snapshot_download,
        )

        snapshot_path = snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            local_files_only=offline,
        )
    except Exception as exc:
        mode_hint = (
            "离线缓存中没有该模型。请联网后重新运行同一命令。"
            if offline
            else (
                "请检查网络或 Hugging Face 访问配置，"
                "然后重新运行同一命令。"
            )
        )
        raise RagReproductionError(
            "EMBEDDING_MODEL_DOWNLOAD_FAILED",
            f"无法取得 {model_name}@{model_revision}。{mode_hint}原始错误：{exc}",
            exit_code=3,
        ) from exc

    return Path(snapshot_path)


def run_evaluation(
    *,
    mode: str,
    data_path: Path,
    index_dir: Path,
    cases_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """运行一种检索模式并写出机器可读报告。"""

    cases = load_evaluation_cases(
        cases_path
    )
    repository = FoodRepository(
        data_path=data_path,
        rag_mode=mode,
        index_dir=index_dir,
    )
    report = evaluate_retrieval(
        repository,
        cases,
    )
    _write_json(
        report_path,
        report,
    )
    return report


def reproduce_rag(
    *,
    data_path: Path,
    hints_path: Path | None,
    index_dir: Path,
    cases_path: Path,
    report_dir: Path,
    model_name: str,
    model_revision: str,
    query_instruction: str,
    offline: bool,
) -> dict[str, Any]:
    """依次完成模型、索引和评测复现。"""

    print(
        "[1/4] 检查 embedding 模型："
        f"{model_name}@{model_revision}",
        flush=True,
    )
    snapshot_path = download_embedding_model(
        model_name=model_name,
        model_revision=model_revision,
        offline=offline,
    )
    print(
        f"      模型可用：{snapshot_path}",
        flush=True,
    )

    print(
        f"[2/4] 重建向量索引：{index_dir}",
        flush=True,
    )
    try:
        manifest = build_index(
            data_path=data_path,
            index_dir=index_dir,
            hints_path=hints_path,
            model_name=model_name,
            model_revision=model_revision,
            query_instruction=query_instruction,
        )
    except Exception as exc:
        raise RagReproductionError(
            "RAG_INDEX_BUILD_FAILED",
            f"向量索引构建失败：{exc}",
            exit_code=4,
        ) from exc

    reports: dict[str, dict[str, Any]] = {}

    for step, mode in (
        ("3/4", "lexical"),
        ("4/4", "hybrid"),
    ):
        report_path = (
            report_dir
            / f"eval_{mode}.json"
        )
        print(
            f"[{step}] 运行 {mode} 评测：{report_path}",
            flush=True,
        )
        try:
            reports[mode] = run_evaluation(
                mode=mode,
                data_path=data_path,
                index_dir=index_dir,
                cases_path=cases_path,
                report_path=report_path,
            )
        except (
            EvaluationError,
            NutritionDataError,
        ) as exc:
            raise RagReproductionError(
                exc.error_code,
                exc.message,
                exit_code=5,
            ) from exc
        except Exception as exc:
            raise RagReproductionError(
                "RAG_EVALUATION_FAILED",
                f"{mode} 评测无法完成：{exc}",
                exit_code=5,
            ) from exc

        report = reports[mode]
        print(
            (
                "      "
                f"Recall@3={report['recall_at_3']:.4f}，"
                f"拒答准确率={report['rejection_accuracy']:.4f}，"
                f"Dense 降级={report['degraded_case_count']}，"
                f"passed={report['passed']}"
            ),
            flush=True,
        )

    summary = {
        "passed": all(
            report["passed"]
            for report in reports.values()
        ),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "snapshot_path": str(
                snapshot_path
            ),
        },
        "index": {
            "directory": str(index_dir),
            "index_id": manifest["index_id"],
            "dataset_id": manifest["dataset_id"],
            "document_count": manifest[
                "document_count"
            ],
        },
        "evaluations": {
            mode: {
                "report": str(
                    report_dir
                    / f"eval_{mode}.json"
                ),
                "recall_at_3": report[
                    "recall_at_3"
                ],
                "rejection_accuracy": report[
                    "rejection_accuracy"
                ],
                "degraded_case_count": report[
                    "degraded_case_count"
                ],
                "passed": report["passed"],
            }
            for mode, report in reports.items()
        },
    }
    summary_path = (
        report_dir
        / "reproduction_summary.json"
    )
    _write_json(
        summary_path,
        summary,
    )

    if not summary["passed"]:
        failed_modes = [
            mode
            for mode, report in reports.items()
            if not report["passed"]
        ]
        raise RagReproductionError(
            "RAG_EVALUATION_THRESHOLD_FAILED",
            (
                "以下评测未达到门槛："
                f"{', '.join(failed_modes)}。"
                f"查看 {report_dir} 中的报告。"
            ),
            exit_code=1,
        )

    print(
        f"复现通过。汇总报告：{summary_path}",
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "下载固定 embedding 模型、"
            "重建索引并运行 Lexical/Hybrid 评测"
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
        "--hints",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "retrieval_hints.json"
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
        "--cases",
        type=Path,
        default=(
            PROJECT_ROOT
            / "tests"
            / "eval"
            / "nutrition_retrieval.jsonl"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "rag-reproduction"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
    )
    parser.add_argument(
        "--query-instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="禁止联网，只使用已经下载的模型缓存。",
    )
    args = parser.parse_args()

    try:
        reproduce_rag(
            data_path=args.data.resolve(),
            hints_path=(
                args.hints.resolve()
                if args.hints
                else None
            ),
            index_dir=args.index_dir.resolve(),
            cases_path=args.cases.resolve(),
            report_dir=args.report_dir.resolve(),
            model_name=args.model_name,
            model_revision=(
                args.model_revision
            ),
            query_instruction=(
                args.query_instruction
            ),
            offline=args.offline,
        )
    except RagReproductionError as exc:
        print(
            f"错误 [{exc.error_code}]：{exc.message}",
            file=sys.stderr,
        )
        return exc.exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
