"""对话优先入口的 Gradio 配置回归测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_conversation_is_default_entry_with_working_starters() -> None:
    """首屏应直接开聊，并提供真实快捷语句。"""

    script = r'''
from src.ui.app import build_demo

config = build_demo().get_config_file()
components = config["components"]
main_tabs = next(
    component
    for component in components
    if component.get("props", {}).get("elem_id") == "main-tabs"
)
assert main_tabs["props"]["selected"] == "chat"

starter_layout = next(
    component
    for component in components
    if "conversation-starters"
    in component.get("props", {}).get("elem_classes", [])
)
assert starter_layout["props"]["scale"] == 0
assert starter_layout["props"]["min_width"] == 0

chatbot = next(
    component
    for component in components
    if component.get("props", {}).get("elem_id") == "health-chat"
)
assert chatbot["props"]["buttons"] == ["copy_all"]

starters = {
    "我刚喝了水",
    "我今天吃了什么",
    "我刚刚运动了",
    "我刚称重了",
}
button_values = {
    component.get("props", {}).get("value")
    for component in components
    if component.get("type") == "button"
}
assert starters | {"查看今天"} <= button_values

starter_button_ids = {
    component["id"]
    for component in components
    if component.get("type") == "button"
    and component.get("props", {}).get("value") in starters
}
triggered_ids = {
    target[0]
    for dependency in config["dependencies"]
    for target in dependency.get("targets", [])
    if isinstance(target, (list, tuple)) and target
}
assert starter_button_ids <= triggered_ids
'''
    environment = os.environ.copy()
    environment["RAG_MODE"] = "lexical"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
