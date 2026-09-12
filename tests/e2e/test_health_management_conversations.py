"""面向真实使用场合的连续健康管理对话 E2E。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.models import AgentMessage, AgentModelReply, ModelToolCall
from src.agent.runner import AgentRunner, ConversationSession
from src.agent.tool_router import HealthToolRouter
from src.storage.healthos_store import HealthOSStore
from src.storage.jsonl_store import HealthEventStore


class ScenarioSequenceModel:
    """按真实多轮节奏返回工具调用，并保留每次场景化系统上下文。"""

    def __init__(self, replies: Sequence[AgentModelReply]) -> None:
        self.replies = list(replies)
        self.system_prompts: list[str] = []

    def complete(
        self,
        messages: Sequence[AgentMessage],
        tool_definitions: Sequence[dict[str, Any]],
    ) -> AgentModelReply:
        self.system_prompts.append(messages[0].content)
        if not self.replies:
            raise AssertionError("场景模型没有剩余响应")
        reply = self.replies.pop(0)
        available = {item["function"]["name"] for item in tool_definitions}
        assert all(call.name in available for call in reply.tool_calls)
        return reply


def _tool(name: str, arguments: dict[str, Any], call_id: str) -> AgentModelReply:
    return AgentModelReply(
        tool_calls=(
            ModelToolCall(call_id=call_id, name=name, arguments=arguments),
        )
    )


def _text(content: str) -> AgentModelReply:
    return AgentModelReply(content=content)


def _session(
    tmp_path: Path,
    replies: Sequence[AgentModelReply],
    session_id: str,
) -> tuple[ConversationSession, ScenarioSequenceModel, HealthOSStore, HealthEventStore]:
    event_store = HealthEventStore(tmp_path / f"{session_id}-events.jsonl")
    healthos_store = HealthOSStore(tmp_path / f"{session_id}-healthos.json")
    router = HealthToolRouter(event_store, healthos_store=healthos_store)
    model = ScenarioSequenceModel(replies)
    session = ConversationSession(
        runner=AgentRunner(model=model, router=router),
        session_id=session_id,
        user_id="scenario-user",
        timezone_name="Asia/Shanghai",
    )
    return session, model, healthos_store, event_store


def test_period_review_continues_into_subjective_context_without_inventing_cause(
    tmp_path: Path,
) -> None:
    session, model, _, _ = _session(
        tmp_path,
        [
            _tool("get_period_summary", {"days": 7}, "review-summary"),
            _text("已汇总最近 7 天。记录本身不能说明原因，这周有什么变化吗？"),
            _text("我会把“加班”作为你补充的感受，不把它写成数据已经证明的原因。"),
        ],
        "period-review",
    )

    first = session.send("帮我复盘最近一周的健康情况")
    second = session.send("我感觉主要是这周一直加班")

    assert first.tool_steps[0].tool_name == "get_period_summary"
    assert "不能说明原因" in first.answer
    assert "你补充的感受" in second.answer
    assert "period_review" in model.system_prompts[-1]


def test_habit_conversation_turns_user_choice_into_confirmed_reminder(
    tmp_path: Path,
) -> None:
    session, model, healthos_store, _ = _session(
        tmp_path,
        [
            _text("你通常在什么时候最容易忘记？我们可以先选一个容易做到的时点。"),
            _tool(
                "create_reminder_draft",
                {
                    "content": "喝水",
                    "scheduled_for": "2099-09-12T10:00:00+08:00",
                    "timezone_name": "Asia/Shanghai",
                    "delivery_channel": "feishu",
                    "recurrence": "daily",
                },
                "habit-reminder",
            ),
        ],
        "habit-followup",
    )

    question = session.send("我总是忘记喝水，想养成习惯")
    draft = session.send("那就每天上午十点提醒我喝水")

    assert "什么时候" in question.answer
    assert draft.pending_confirmation is not None
    assert healthos_store.read().reminders == []
    session.confirm()
    assert healthos_store.read().reminders[0].recurrence == "daily"
    assert "habit_followup" in model.system_prompts[-1]


def test_sleep_conversation_uses_knowledge_then_creates_opt_in_routine_reminder(
    tmp_path: Path,
) -> None:
    session, model, healthos_store, _ = _session(
        tmp_path,
        [
            _tool(
                "retrieve_health_knowledge",
                {"question": "成年人睡眠不足的一般改善建议", "top_k": 3},
                "sleep-knowledge",
            ),
            _text("这种情况持续多久、白天是否受影响？我不会把它当作诊断。"),
            _tool(
                "create_reminder_draft",
                {
                    "content": "准备睡觉",
                    "scheduled_for": "2099-09-12T23:00:00+08:00",
                    "timezone_name": "Asia/Shanghai",
                    "delivery_channel": "feishu",
                    "recurrence": "daily",
                },
                "sleep-reminder",
            ),
        ],
        "sleep-routine",
    )

    first = session.send("最近总是睡不着，白天也很困")
    draft = session.send("持续两周了，每晚十一点提醒我准备睡觉")

    assert first.tool_steps[0].tool_name == "retrieve_health_knowledge"
    assert "不会把它当作诊断" in first.answer
    assert draft.pending_confirmation is not None
    session.confirm()
    assert healthos_store.read().reminders[0].content == "准备睡觉"
    assert "sleep_wellbeing" in model.system_prompts[-1]


def test_confirmed_food_preference_is_available_to_later_meal_guidance(
    tmp_path: Path,
) -> None:
    session, model, healthos_store, _ = _session(
        tmp_path,
        [
            _tool(
                "prepare_profile_update",
                {"patch": {"exclusions": ["花生"]}},
                "preference-draft",
            ),
            _tool("get_user_profile", {}, "preference-read"),
            _tool(
                "retrieve_health_knowledge",
                {"question": "晚餐如何均衡搭配", "top_k": 3},
                "meal-guidance",
            ),
            _text("会避开花生，并结合今天已确认的记录给你几个晚餐选择。"),
        ],
        "preference-continuity",
    )

    draft = session.send("请记住我不吃花生")
    assert draft.pending_confirmation is not None
    session.confirm()
    answer = session.send("那以后晚餐怎么搭配？")

    profile = healthos_store.get_profile("scenario-user", "Asia/Shanghai")
    assert profile.exclusions == ["花生"]
    assert [step.tool_name for step in answer.tool_steps] == [
        "get_user_profile",
        "retrieve_health_knowledge",
    ]
    assert "花生" in answer.answer
    assert "忌口：花生" in model.system_prompts[-1]


def test_conflicting_goals_are_clarified_before_a_versioned_goal_is_created(
    tmp_path: Path,
) -> None:
    session, model, healthos_store, _ = _session(
        tmp_path,
        [
            _tool("get_health_goals", {}, "goal-read"),
            _text("你更希望先改善哪一个？也请告诉我时间范围。"),
            _tool(
                "prepare_goal_change",
                {
                    "operation": "create",
                    "title": "8 周减重 4 公斤",
                    "goal_type": "weight",
                    "target_value": 4,
                    "unit": "kg",
                    "period": "8_weeks",
                    "reason": "用户选择先减重",
                },
                "goal-draft",
            ),
        ],
        "goal-tradeoff",
    )

    clarification = session.send("我既想减重又想增肌，哪个目标优先？")
    draft = session.send("先减重，希望八周减四公斤")

    assert clarification.tool_steps[0].tool_name == "get_health_goals"
    assert "先改善哪一个" in clarification.answer
    assert healthos_store.read().goals == []
    session.confirm()
    assert healthos_store.read().goals[0].current.title == "8 周减重 4 公斤"
    assert "goal_tradeoff" in model.system_prompts[0]


def test_urgent_symptom_interrupts_ordinary_coaching_without_writing_data(
    tmp_path: Path,
) -> None:
    session, model, _, event_store = _session(
        tmp_path,
        [
            _tool(
                "retrieve_health_knowledge",
                {"question": "运动后胸痛并呼吸困难", "top_k": 3},
                "urgent-safety",
            ),
            _text("这可能需要立即处理。请马上联系当地急救服务或尽快就医，不要继续运动。"),
        ],
        "urgent-safety",
    )

    result = session.send("我运动后胸痛，还有点呼吸困难")

    assert result.tool_steps[0].tool_name == "retrieve_health_knowledge"
    assert result.tool_steps[0].result["ok"] is False
    assert result.tool_steps[0].result["error"]["error_code"] == "URGENT_HELP_REQUIRED"
    assert "立即" in result.answer
    assert event_store.read_all() == []
    assert "safety_escalation" in model.system_prompts[0]


def test_weight_plateau_reads_period_facts_and_goals_before_discussion(
    tmp_path: Path,
) -> None:
    session, model, _, _ = _session(
        tmp_path,
        [
            _tool("get_period_summary", {"days": 14}, "weight-period"),
            _tool("get_health_goals", {}, "weight-goals"),
            _text("现有记录不足以判断平台期原因。你最近有哪些自己注意到的变化？"),
        ],
        "weight-journey",
    )

    result = session.send("减重进入平台期了，体重一直没变化")

    assert [step.tool_name for step in result.tool_steps] == [
        "get_period_summary",
        "get_health_goals",
    ]
    assert "不足以判断" in result.answer
    assert "weight_journey" in model.system_prompts[0]


def test_exercise_recovery_separates_recorded_activity_from_general_guidance(
    tmp_path: Path,
) -> None:
    session, model, _, _ = _session(
        tmp_path,
        [
            _tool(
                "get_health_events",
                {"event_type": "exercise", "limit": 20},
                "exercise-events",
            ),
            _tool(
                "retrieve_health_knowledge",
                {"question": "运动后普通肌肉酸痛如何恢复", "top_k": 3},
                "recovery-knowledge",
            ),
            _text("我会把已记录的运动和你描述的酸痛分开看；如果疼痛异常或加重，请停止训练并寻求专业帮助。"),
        ],
        "exercise-recovery",
    )

    result = session.send("训练后一直酸痛，下一次运动怎么安排？")

    assert [step.tool_name for step in result.tool_steps] == [
        "get_health_events",
        "retrieve_health_knowledge",
    ]
    assert "分开看" in result.answer
    assert "exercise_recovery" in model.system_prompts[0]
