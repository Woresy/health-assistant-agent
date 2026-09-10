"""Tencent Lighthouse single-instance deployment safety contract."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_tencent_compose_keeps_gradio_private_and_data_persistent() -> None:
    manifest = yaml.safe_load(
        (PROJECT_ROOT / "deploy" / "tencent" / "compose.yaml").read_text(
            encoding="utf-8"
        )
    )
    healthos = manifest["services"]["healthos"]
    caddy = manifest["services"]["caddy"]

    assert "ports" not in healthos
    assert healthos["environment"]["APP_HOST"] == "0.0.0.0"
    assert healthos["environment"]["SQLITE_DATABASE_PATH"] == (
        "/app/runtime/healthos.db"
    )
    assert healthos["environment"]["AGENT_MAX_RETRIES"] == 0
    assert "./runtime-data:/app/runtime" in healthos["volumes"]
    assert "80:80" in caddy["ports"]
    assert "443:443" in caddy["ports"]


def test_caddy_requires_authentication_before_reverse_proxy() -> None:
    caddyfile = (PROJECT_ROOT / "deploy" / "tencent" / "Caddyfile").read_text(
        encoding="utf-8"
    )

    assert caddyfile.index("basic_auth") < caddyfile.index("reverse_proxy")
    assert "{$BASIC_AUTH_HASH}" in caddyfile
    assert "healthos:7860" in caddyfile


def test_container_runs_as_unprivileged_user() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--uid 10001" in dockerfile
    assert "USER healthos" in dockerfile
    assert "HEALTHCHECK" in dockerfile
