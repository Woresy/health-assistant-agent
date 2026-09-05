"""五层 Prompt Pipeline 的内容和隐私边界。"""

from src.agent.context_pipeline import build_prompt_context
from src.agent.models import AgentMessage, PendingTask


def test_prompt_pipeline_orders_five_layers_without_hidden_reasoning() -> None:
    context = build_prompt_context(
        system_rules="SYSTEM RULES",
        user_input="我喝了 350 ml 水",
        profile_context={
            "profile": {
                "timezone_name": "Asia/Shanghai",
                "unit_system": "metric",
                "coach_style": "gentle",
                "dietary_preferences": [],
                "exclusions": [],
            },
            "goals": [
                {
                    "title": "每日饮水",
                    "target_value": 1800.0,
                    "unit": "ml",
                    "period": "daily",
                }
            ],
        },
        pending_task=PendingTask(
            tool_name="prepare_health_event",
            arguments={"event_type": "water"},
            missing_parameters=["amount_ml"],
            question="喝了多少？",
        ),
        messages=[
            AgentMessage(role="user", content="旧请求"),
            AgentMessage(role="tool", content="{}", tool_name="old_tool"),
            AgentMessage(role="user", content="我喝了 350 ml 水"),
            AgentMessage(role="tool", content="{}", tool_name="get_health_events"),
        ],
    )

    rendered = context.render_system_message()
    positions = [rendered.index(f"第 {number} 层") for number in range(1, 6)]
    assert positions == sorted(positions)
    assert "我喝了 350 ml 水" not in rendered
    assert "每日饮水：1800 ml / daily" in rendered
    assert "get_health_events" in rendered
    assert "old_tool" not in rendered
    assert len(context.layer_receipt) == 5
