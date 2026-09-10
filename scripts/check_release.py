#!/usr/bin/env python3
"""Fail fast when a GitHub delivery contains secrets or machine-local artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    ".env.example",
    ".github/workflows/ci.yml",
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "app.py",
    "docs/ARCHITECTURE.md",
    "docs/EVALUATION.md",
    "docs/DEPLOY_TENCENT.md",
    "docs/PERFORMANCE.md",
    "docs/RAG.md",
    "docs/TOOLS.md",
    "package-lock.json",
    "deploy/tencent/Caddyfile",
    "deploy/tencent/compose.yaml",
    "deploy/tencent/production.env.example",
    "requirements.txt",
    "scripts/benchmark_performance.py",
    "scripts/reproduce_rag.sh",
    "tests/browser/healthos.test.cjs",
)
FORBIDDEN_REPOSITORY_PATHS = (
    ".env",
    "data/healthos.db",
    "data/health_events.jsonl",
    "data/agent_traces.jsonl",
)
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "LangSmith API key": re.compile(r"\blsv2_[A-Za-z0-9_-]{20,}\b"),
    "Feishu webhook token": re.compile(r"/bot/v2/hook/[A-Za-z0-9_-]{20,}"),
}
ABSOLUTE_HOME = re.compile(r"/(?:home|Users)/[^/\s]+/")


def _repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [PROJECT_ROOT / item for item in result.stdout.splitlines() if item]


def _read_text(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > 5_000_000:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def check_release() -> dict[str, Any]:
    files = _repository_files()
    relative_files = {str(path.relative_to(PROJECT_ROOT)) for path in files}
    failures: list[str] = []

    for required in REQUIRED_PATHS:
        if required not in relative_files:
            failures.append(f"缺少交付文件：{required}")

    for forbidden in FORBIDDEN_REPOSITORY_PATHS:
        if forbidden in relative_files:
            failures.append(f"运行时或敏感文件不应进入仓库：{forbidden}")

    ignored_env = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    if not ignored_env:
        failures.append(".env 没有被 .gitignore 排除")

    scanned_files = 0
    for path in files:
        relative = str(path.relative_to(PROJECT_ROOT))
        text = _read_text(path)
        if text is None:
            continue
        scanned_files += 1
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"疑似 {label}：{relative}")
        if relative != "scripts/check_release.py" and ABSOLUTE_HOME.search(text):
            failures.append(f"包含开发机绝对路径：{relative}")

    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    unpinned = [
        line for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" not in line
    ]
    if unpinned:
        failures.append("requirements.txt 存在未固定版本的依赖")

    return {
        "schema_version": 1,
        "checked_files": len(files),
        "scanned_text_files": scanned_files,
        "checks": {
            "required_delivery_files": all(path in relative_files for path in REQUIRED_PATHS),
            "env_is_ignored": ignored_env,
            "dependencies_are_pinned": not unpinned,
            "no_secrets_or_machine_paths": not any(
                "疑似" in failure or "绝对路径" in failure for failure in failures
            ),
            "no_runtime_data": not any("运行时" in failure for failure in failures),
        },
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check_release()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
