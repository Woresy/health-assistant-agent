"""健康记录面向用户的展示文本回归测试。"""

from src.agent.runner import (
    _preview_answer,
    format_health_event_summary,
)


def test_weight_preview_uses_plain_language() -> None:
    """体重草稿不应把模型参数直接展示给用户。"""

    answer = _preview_answer(
        {
            "action": "save",
            "preview": {
                "event_type": "weight",
                "weight_kg": 68.0,
                "note": "",
            },
        }
    )

    assert "体重 68 kg" in answer
    assert "event_type" not in answer
    assert "weight_kg" not in answer
    assert "{" not in answer


def test_full_exercise_event_has_readable_summary() -> None:
    """完整事件也应压缩成一行自然语言。"""

    summary = format_health_event_summary(
        {
            "event_type": "exercise",
            "payload": {
                "activity_type": "慢跑",
                "duration_minutes": 30,
                "distance_km": 4.2,
                "intensity": "medium",
                "note": "晚饭后",
            },
        }
    )

    assert summary == (
        "慢跑，30 分钟，4.2 km，"
        "中等强度，晚饭后"
    )


def test_update_preview_compares_before_and_after() -> None:
    """修改草稿保留前后对比，但不暴露 JSON 字段名。"""

    answer = _preview_answer(
        {
            "action": "update",
            "current_event": {
                "event_type": "water",
                "payload": {
                    "amount_ml": 300,
                    "beverage": "饮用水",
                },
            },
            "proposed_event": {
                "event_type": "water",
                "payload": {
                    "amount_ml": 500,
                    "beverage": "饮用水",
                },
            },
        }
    )

    assert "修改前" in answer
    assert "饮用水 300 ml" in answer
    assert "修改后" in answer
    assert "饮用水 500 ml" in answer
    assert "amount_ml" not in answer


def test_english_water_value_is_localized_for_confirmation() -> None:
    """模型返回英文饮品值时，确认卡仍应使用用户可读中文。"""

    summary = format_health_event_summary(
        {
            "event_type": "water",
            "amount_ml": 600,
            "beverage": "water",
        }
    )

    assert summary == "饮用水 600 ml"
