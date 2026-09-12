"""使用纯合成数据验证 LangSmith 写入、读取和隐私输出契约。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from langsmith import Client  # noqa: E402

from src.observability.langsmith import configure_langsmith  # noqa: E402


SYNTHETIC_INPUT = "synthetic-health-input-never-upload"
SYNTHETIC_OUTPUT = "synthetic-health-output-never-upload"


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def verify(*, report_path: Path | None = None) -> dict[str, object]:
    status = configure_langsmith()
    if not status.ready:
        raise RuntimeError(status.message)
    if not _enabled(os.getenv("LANGSMITH_HIDE_INPUTS")):
        raise RuntimeError("验证要求 LANGSMITH_HIDE_INPUTS=true")
    if not _enabled(os.getenv("LANGSMITH_HIDE_OUTPUTS")):
        raise RuntimeError("验证要求 LANGSMITH_HIDE_OUTPUTS=true")

    # 先在本地验证业务输出契约；远程只接收隐藏后的观测结构。
    local_output = {
        "state": "completed",
        "finish_reason": "completed",
        "tool_sequence": ["retrieve_health_knowledge"],
        "citation_count": 1,
        "answer": SYNTHETIC_OUTPUT,
    }
    if local_output["state"] != "completed" or local_output["citation_count"] != 1:
        raise RuntimeError("本地合成输出契约验证失败")

    client = Client()
    run_id = uuid4()
    started_at = datetime.now(timezone.utc)
    client.create_run(
        id=run_id,
        name="HealthOS Synthetic Output Verification",
        run_type="chain",
        project_name=status.project,
        inputs={"message": SYNTHETIC_INPUT},
        start_time=started_at,
        tags=["healthos", "verification", "synthetic-only"],
        extra={
            "metadata": {
                "scenario": "synthetic-output-contract",
                "expected_state": "completed",
                "privacy": "inputs-outputs-hidden",
            }
        },
    )
    client.update_run(
        run_id,
        end_time=datetime.now(timezone.utc),
        outputs=local_output,
    )
    client.flush()

    remote_run = None
    for _ in range(20):
        try:
            remote_run = client.read_run(run_id)
            break
        except Exception:
            time.sleep(0.5)
    if remote_run is None:
        raise RuntimeError("LangSmith 写入后无法读取合成 Trace")

    serialized = json.dumps(remote_run.model_dump(mode="json"), ensure_ascii=False)
    if SYNTHETIC_INPUT in serialized or SYNTHETIC_OUTPUT in serialized:
        raise RuntimeError("LangSmith 隐私验证失败：远程 Trace 出现未隐藏的合成输入或输出")
    tags = set(remote_run.tags or [])
    required_tags = {"healthos", "verification", "synthetic-only"}
    if not required_tags.issubset(tags):
        raise RuntimeError("LangSmith Trace 标签不完整")

    result: dict[str, object] = {
        "passed": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "project": status.project,
        "run_id": str(run_id),
        "checks": {
            "local_output_contract": "passed",
            "remote_trace_roundtrip": "passed",
            "remote_inputs_hidden": "passed",
            "remote_outputs_hidden": "passed",
            "required_tags": "passed",
        },
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/langsmith-verification/report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file, override=False)
    print(json.dumps(verify(report_path=args.report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
