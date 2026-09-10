#!/usr/bin/env python3
"""Measure the local performance contract without calling an external model."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THRESHOLDS = {
    "lexical_cold_start_seconds": 15.0,
    "warm_page_response_seconds": 3.0,
    "deterministic_draft_seconds": 2.0,
    "model_target_response_seconds": 15.0,
    "model_hard_timeout_seconds": 60.0,
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _isolated_environment(temp_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "AGENT_PROVIDER_MODE": "disabled",
            "APP_HOST": "127.0.0.1",
            "LANGSMITH_TRACING": "false",
            "RAG_MODE": "lexical",
            "SQLITE_DATABASE_PATH": str(temp_dir / "healthos.db"),
            "STORAGE_BACKEND": "sqlite",
        }
    )
    return environment


def _reserve_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _measure_server(environment: dict[str, str]) -> tuple[float, float]:
    port = _reserve_port()
    environment = {**environment, "APP_PORT": str(port)}
    process = subprocess.Popen(
        [str(PYTHON), "app.py"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    started_at = time.perf_counter()
    ready_at: float | None = None
    try:
        deadline = started_at + 30
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Gradio exited during startup:\n{output[-4000:]}")
            try:
                with urllib.request.urlopen(url, timeout=0.5) as response:
                    if response.status == 200:
                        ready_at = time.perf_counter()
                        break
            except OSError:
                time.sleep(0.05)
        if ready_at is None:
            raise TimeoutError("Gradio was not ready within 30 seconds")

        warm_samples: list[float] = []
        for _ in range(10):
            request_started_at = time.perf_counter()
            with urllib.request.urlopen(url, timeout=3) as response:
                response.read(512)
            warm_samples.append(time.perf_counter() - request_started_at)
        return ready_at - started_at, _percentile(warm_samples, 0.95)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _measure_deterministic_draft(temp_dir: Path) -> float:
    from src.agent.models import AgentModelReply
    from src.agent.runner import AgentRunner
    from src.agent.tool_router import HealthToolRouter
    from src.nutrition.repository import FoodRepository
    from src.storage.healthos_store import HealthOSStore
    from src.storage.jsonl_store import HealthEventStore

    class ModelMustNotRun:
        def complete(self, messages: Any, tool_definitions: Any) -> AgentModelReply:
            raise AssertionError("deterministic reminder unexpectedly called the model")

    router = HealthToolRouter(
        HealthEventStore(temp_dir / "events.jsonl"),
        healthos_store=HealthOSStore(temp_dir / "healthos.json"),
        nutrition_repository=FoodRepository(rag_mode="lexical"),
    )
    runner = AgentRunner(model=ModelMustNotRun(), router=router)
    samples: list[float] = []
    for index in range(20):
        state = runner.create_session_state(
            session_id=f"performance-{index}",
            user_id="performance-user",
        )
        started_at = time.perf_counter()
        outcome = runner.run_turn(
            session_state=state,
            user_text="两分钟后通过飞书提醒我喝水",
        )
        samples.append(time.perf_counter() - started_at)
        if outcome.result.model_rounds != 0:
            raise AssertionError("deterministic draft used a model round")
    return _percentile(samples, 0.95)


def benchmark() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="healthos-performance-") as raw_dir:
        temp_dir = Path(raw_dir)
        environment = _isolated_environment(temp_dir)
        cold_start, warm_response = _measure_server(environment)
        deterministic_draft = _measure_deterministic_draft(temp_dir)

    measurements = {
        "lexical_cold_start_seconds": round(cold_start, 3),
        "warm_page_response_seconds_p95": round(warm_response, 4),
        "deterministic_draft_seconds_p95": round(deterministic_draft, 4),
    }
    checks = {
        "lexical_cold_start": cold_start <= THRESHOLDS["lexical_cold_start_seconds"],
        "warm_page_response": warm_response <= THRESHOLDS["warm_page_response_seconds"],
        "deterministic_draft": deterministic_draft <= THRESHOLDS["deterministic_draft_seconds"],
    }
    return {
        "schema_version": 1,
        "mode": "lexical-offline",
        "measurements": measurements,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "passed": all(checks.values()),
        "notes": [
            "External provider latency is not fabricated by this offline benchmark.",
            "The 15s model value is an experience target; the enforced per-request timeout is 60s.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--assert-thresholds", action="store_true")
    args = parser.parse_args()
    report = benchmark()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    if args.assert_thresholds and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
