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
from src.ui.app import begin_agent_activity, build_demo, finish_agent_activity

active_process = begin_agent_activity("我今天喝了多少水", {})
assert active_process.value.count("<li") == 3
assert "小满正在处理" in active_process.value
assert "完成后会告诉你结果" in active_process.value

finished_process = finish_agent_activity(
    "本轮操作已完成。",
    [{"tool": "query_health_events", "status": "成功", "source": "本地 SQLite 业务数据"}],
)
assert "本次处理" in finished_process.value
assert "读取已确认的健康记录" in finished_process.value
assert "成功" in finished_process.value
assert "SQLite" not in finished_process.value

config = build_demo().get_config_file()
components = config["components"]
main_tabs = next(
    component
    for component in components
    if component.get("props", {}).get("elem_id") == "main-tabs"
)
assert main_tabs["props"]["selected"] == "chat"

developer_tab = next(
    component
    for component in components
    if component.get("props", {}).get("elem_id") == "healthos-evidence"
)
assert developer_tab["props"]["visible"] is False

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
assert starters | {"查看完整汇总", "＋  创建新对话"} <= button_values

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
    environment["HEALTHOS_SHOW_DEVELOPER_UI"] = "false"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
