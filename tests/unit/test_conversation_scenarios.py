"""真实健康管理多轮场景的识别与上下文注入。"""

from __future__ import annotations

import pytest

from src.agent.context_pipeline import build_prompt_context
from src.agent.conversation_scenarios import SCENARIOS, detect_conversation_scenario
from src.agent.models import AgentMessage
from src.agent.tool_router import TOOL_DEFINITIONS


@pytest.mark.parametrize(
    ("user_text", "expected"),
    [
        ("我胸痛而且有点呼吸困难", "safety_escalation"),
        ("我既想减重又想增肌，哪个目标优先", "goal_tradeoff"),
        ("今天已经吃了午餐，晚餐吃什么", "meal_guidance"),
        ("帮我复盘最近一周的健康情况", "period_review"),
        ("训练后一直酸痛，下一次运动怎么安排", "exercise_recovery"),
        ("最近总是睡不着，白天很困", "sleep_wellbeing"),
        ("减重进入平台期，体重没变化", "weight_journey"),
        ("我总是忘记喝水，想养成习惯", "habit_followup"),
        ("请记住我不吃花生", "preference_continuity"),
        ("早餐吃了米饭，午餐还没记录", "daily_meal_tracking"),
    ],
)
def test_detects_supported_health_management_scenarios(
    user_text: str,
    expected: str,
) -> None:
    scenario = detect_conversation_scenario(user_text)
    assert scenario is not None
    assert scenario.scenario_id == expected


def test_urgent_signal_takes_priority_over_other_scenarios() -> None:
    scenario = detect_conversation_scenario("运动后胸痛，晚上也睡不着")
    assert scenario is not None
    assert scenario.scenario_id == "safety_escalation"


def test_elliptical_follow_up_inherits_recent_scenario() -> None:
    messages = (
        AgentMessage(role="user", content="最近总是睡不着，想改善睡眠"),
        AgentMessage(role="assistant", content="这种情况持续多久了？"),
        AgentMessage(role="user", content="差不多两周"),
    )
    scenario = detect_conversation_scenario("那今晚先做什么？", messages)
    assert scenario is not None
    assert scenario.scenario_id == "sleep_wellbeing"


def test_scenario_playbook_is_injected_without_repeating_user_text() -> None:
    context = build_prompt_context(
        system_rules="SYSTEM RULES",
        user_input="帮我复盘最近一周的健康情况",
        profile_context={"profile": {}, "goals": []},
        pending_task=None,
        messages=[
            AgentMessage(role="user", content="帮我复盘最近一周的健康情况"),
        ],
    )
    rendered = context.render_system_message()
    assert "7/14/30 天健康复盘" in rendered
    assert "先确认复盘周期" in rendered
    assert "get_period_summary" in rendered
    assert "帮我复盘最近一周的健康情况" not in rendered


def test_every_playbook_only_recommends_real_tools() -> None:
    available = {item["function"]["name"] for item in TOOL_DEFINITIONS}
    assert len(SCENARIOS) == 10
    for scenario in SCENARIOS:
        assert set(scenario.recommended_tools) <= available
