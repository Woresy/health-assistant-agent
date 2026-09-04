"""HealthOS P1 档案、目标与提醒的 Agent 端到端流程。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.models import AgentMessage, AgentModelReply, ModelToolCall
from src.agent.runner import AgentRunner, ConversationSession
from src.agent.tool_router import HealthToolRouter
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore


class OneShotModel:
    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments

    def complete(
        self,
        messages: Sequence[AgentMessage],
        tool_definitions: Sequence[dict[str, Any]],
    ) -> AgentModelReply:
        del messages
        assert len(tool_definitions) == 15
        return AgentModelReply(
            tool_calls=(
                ModelToolCall(
                    call_id="p1-call",
                    name=self.tool_name,
                    arguments=self.arguments,
                ),
            )
        )


def build_session(
    tmp_path: Path, tool_name: str, arguments: dict[str, Any]
) -> tuple[ConversationSession, HealthOSStore, HealthToolRouter]:
    healthos_store = HealthOSStore(tmp_path / "healthos.json")
    router = HealthToolRouter(
        HealthEventStore(tmp_path / "events.jsonl"),
        healthos_store=healthos_store,
    )
    runner = AgentRunner(
        model=OneShotModel(tool_name, arguments),
        router=router,
    )
    return (
        ConversationSession(
            runner=runner,
            session_id="p1-session",
            user_id="p1-user",
            timezone_name="Asia/Shanghai",
        ),
        healthos_store,
        router,
    )


def dispatch_read(router: HealthToolRouter, name: str, arguments: dict[str, Any]):
    return router.dispatch(
        tool_name=name,
        arguments=arguments,
        user_id="p1-user",
        timezone_name="Asia/Shanghai",
        session_id="read-session",
        call_id=f"read-{name}",
    )


def test_agent_profile_draft_confirm_and_read(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_profile_update",
        {"patch": {"coach_style": "concise"}},
    )
    prepared = session.send("把教练风格改成简洁提醒")
    assert prepared.state.value == "awaiting_confirmation"
    assert dispatch_read(router, "get_user_profile", {}).result["data"]["profile"]["coach_style"] == "gentle"
    confirmed = session.confirm()
    assert confirmed.answer == "个人档案已确认更新。"
    assert dispatch_read(router, "get_user_profile", {}).result["data"]["profile"]["coach_style"] == "concise"


def test_agent_goal_draft_confirm_preserves_versioned_goal(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_goal_change",
        {
            "operation": "create",
            "title": "每周运动 150 分钟",
            "goal_type": "exercise",
            "target_value": 150,
            "unit": "分钟",
            "period": "weekly",
            "reason": "保持规律运动",
        },
    )
    prepared = session.send("创建每周运动 150 分钟的目标")
    assert "确认" in prepared.answer
    session.confirm()
    goals = dispatch_read(router, "get_health_goals", {}).result["data"]["goals"]
    assert goals[0]["versions"][0]["title"] == "每周运动 150 分钟"
    assert goals[0]["versions"][0]["version"] == 1


def test_agent_reminder_draft_confirm_and_recall(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "create_reminder_draft",
        {
            "content": "称体重",
            "scheduled_for": "2099-09-05T21:00:00+08:00",
        },
    )
    prepared = session.send("今晚九点提醒我称体重")
    assert prepared.pending_confirmation is not None
    before = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"})
    assert before.result["data"]["count"] == 0
    confirmed = session.confirm()
    assert confirmed.answer == "提醒已确认安排。"
    after = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"})
    assert after.result["data"]["reminders"][0]["content"] == "称体重"
