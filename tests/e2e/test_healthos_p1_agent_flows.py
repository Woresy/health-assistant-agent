"""HealthOS P1 档案、目标与提醒的 Agent 端到端流程。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.models import AgentMessage, AgentModelReply, ModelToolCall, PendingConfirmation
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
        assert self.tool_name in {
            item["function"]["name"]
            for item in tool_definitions
        }
        return AgentModelReply(
            tool_calls=(
                ModelToolCall(
                    call_id="p1-call",
                    name=self.tool_name,
                    arguments=self.arguments,
                ),
            )
        )


class SequenceModel:
    """按顺序返回工具调用和最终文本，用于只读工具 E2E。"""

    def __init__(self, replies: list[AgentModelReply]) -> None:
        self.replies = list(replies)

    def complete(
        self,
        messages: Sequence[AgentMessage],
        tool_definitions: Sequence[dict[str, Any]],
    ) -> AgentModelReply:
        del messages, tool_definitions
        if not self.replies:
            raise AssertionError("SequenceModel 没有剩余响应")
        return self.replies.pop(0)


def tool_reply(name: str, arguments: dict[str, Any], call_id: str) -> AgentModelReply:
    return AgentModelReply(
        tool_calls=(ModelToolCall(call_id=call_id, name=name, arguments=arguments),)
    )


def sequence_session(
    router: HealthToolRouter,
    replies: list[AgentModelReply],
    session_id: str,
) -> ConversationSession:
    return ConversationSession(
        runner=AgentRunner(model=SequenceModel(replies), router=router),
        session_id=session_id,
        user_id="p1-user",
        timezone_name="Asia/Shanghai",
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


def dispatch_read(
    router: HealthToolRouter,
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str | None = None,
):
    return router.dispatch(
        tool_name=name,
        arguments=arguments,
        user_id="p1-user",
        timezone_name="Asia/Shanghai",
        session_id="read-session",
        call_id=call_id or f"read-{name}",
    )


def pending_from(result: Any, tool_name: str) -> PendingConfirmation:
    data = result.result["data"]
    return PendingConfirmation(action=data["action"], tool_name=tool_name, draft_data=data)


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


def test_profile_update_then_profile_read_preserves_confirmed_fields(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_profile_update",
        {"patch": {"dietary_preferences": ["少油"], "exclusions": ["花生"]}},
    )
    assert session.send("记住我偏好少油并且不吃花生").pending_confirmation is not None
    session.confirm()
    profile = dispatch_read(router, "get_user_profile", {}).result["data"]["profile"]
    assert profile["dietary_preferences"] == ["少油"]
    assert profile["exclusions"] == ["花生"]


def test_goal_create_then_update_appends_second_version(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_goal_change",
        {
            "operation": "create",
            "title": "每天饮水 1800 毫升",
            "goal_type": "water",
            "target_value": 1800,
            "unit": "ml",
            "period": "daily",
            "reason": "建立饮水习惯",
        },
    )
    created = session.send("创建饮水目标")
    goal_id = created.pending_confirmation.draft_data["payload"]["goal_id"]
    session.confirm()
    changed = dispatch_read(
        router,
        "prepare_goal_change",
        {"operation": "update", "goal_id": goal_id, "target_value": 2000, "reason": "逐步增加"},
    )
    router.confirm(pending_from(changed, "prepare_goal_change"))
    versions = dispatch_read(router, "get_health_goals", {}).result["data"]["goals"][0]["versions"]
    assert [item["target_value"] for item in versions] == [1800.0, 2000.0]


def test_goal_pause_then_resume_keeps_full_history(tmp_path: Path) -> None:
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
            "reason": "规律运动",
        },
    )
    created = session.send("创建运动目标")
    goal_id = created.pending_confirmation.draft_data["payload"]["goal_id"]
    session.confirm()
    for index, operation in enumerate(("pause", "resume"), start=1):
        draft = dispatch_read(
            router,
            "prepare_goal_change",
            {"operation": operation, "goal_id": goal_id, "reason": f"状态变更 {index}"},
            call_id=f"goal-{operation}",
        )
        router.confirm(pending_from(draft, "prepare_goal_change"))
    versions = dispatch_read(router, "get_health_goals", {}).result["data"]["goals"][0]["versions"]
    assert [item["status"] for item in versions] == ["active", "paused", "active"]


def test_committed_water_event_updates_daily_summary(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_health_event",
        {
            "event_type": "water",
            "amount_ml": 600,
            "occurred_at": "2099-09-05T09:00:00+08:00",
        },
    )
    session.send("记录喝水 600 毫升")
    session.confirm()
    summary = dispatch_read(
        router,
        "get_daily_summary",
        {"date": "2099-09-05", "timezone_name": "Asia/Shanghai"},
    )
    assert summary.result["data"]["summary"]["water"]["total_ml"] == 600


def test_committed_exercise_event_updates_period_summary(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "prepare_health_event",
        {
            "event_type": "exercise",
            "activity_type": "快走",
            "duration_minutes": 35,
            "occurred_at": "2099-09-05T18:00:00+08:00",
        },
    )
    session.send("记录快走 35 分钟")
    session.confirm()
    period = dispatch_read(
        router,
        "get_period_summary",
        {"days": 7, "end_date": "2099-09-05", "timezone_name": "Asia/Shanghai"},
    )
    assert period.result["data"]["exercise"]["total_minutes"] == 35
    assert period.result["data"]["event_count"] == 1


def test_agent_health_knowledge_returns_cited_result(tmp_path: Path) -> None:
    _, _, router = build_session(tmp_path, "get_user_profile", {})
    session = sequence_session(
        router,
        [
            tool_reply("retrieve_health_knowledge", {"question": "成年人每周运动多久？"}, "knowledge"),
            AgentModelReply(content="一般参考是每周至少 150 分钟，并请查看页面中的 WHO 来源。"),
        ],
        "knowledge-session",
    )
    result = session.send("成年人每周运动多久？")
    assert result.state.value == "completed"
    assert result.tool_steps[0].result["data"]["citations"][0]["source_url"].startswith("https://www.who.int/")


def test_agent_health_knowledge_dangerous_request_stops_safely(tmp_path: Path) -> None:
    _, _, router = build_session(tmp_path, "get_user_profile", {})
    session = sequence_session(
        router,
        [tool_reply("retrieve_health_knowledge", {"question": "我胸痛并且呼吸困难"}, "urgent")],
        "urgent-session",
    )
    result = session.send("我胸痛并且呼吸困难")
    assert result.state.value == "failed"
    assert result.tool_steps[0].result["error"]["error_code"] == "URGENT_HELP_REQUIRED"


def test_reminder_create_then_snooze_is_confirmed_and_traceable(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "create_reminder_draft",
        {"content": "站起来活动", "scheduled_for": "2099-09-05T20:00:00+08:00"},
    )
    session.send("晚上八点提醒我活动")
    session.confirm()
    reminder = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"}).result["data"]["reminders"][0]
    draft = dispatch_read(
        router,
        "list_or_cancel_reminders",
        {
            "action": "snooze",
            "reminder_id": reminder["reminder_id"],
            "scheduled_for": "2099-09-05T20:30:00+08:00",
            "reason": "延后半小时",
        },
    )
    router.confirm(pending_from(draft, "list_or_cancel_reminders"))
    updated = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"}).result["data"]["reminders"][0]
    assert updated["status"] == "snoozed"
    assert len(updated["transitions"]) == 2


def test_reminder_create_then_cancel_requires_second_confirmation(tmp_path: Path) -> None:
    session, _, router = build_session(
        tmp_path,
        "create_reminder_draft",
        {"content": "记录体重", "scheduled_for": "2099-09-05T21:00:00+08:00"},
    )
    session.send("晚上九点提醒我记录体重")
    session.confirm()
    reminder = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"}).result["data"]["reminders"][0]
    draft = dispatch_read(
        router,
        "list_or_cancel_reminders",
        {"action": "cancel", "reminder_id": reminder["reminder_id"], "reason": "计划取消"},
    )
    before = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"}).result["data"]["reminders"][0]
    assert before["status"] == "scheduled"
    router.confirm(pending_from(draft, "list_or_cancel_reminders"))
    after = dispatch_read(router, "list_or_cancel_reminders", {"action": "list"}).result["data"]["reminders"][0]
    assert after["status"] == "cancelled"
