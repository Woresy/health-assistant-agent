"""五层 Prompt Context Pipeline。

这里组织的是可审计上下文，不保存也不展示模型的隐藏思维过程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.agent.models import AgentMessage, PendingTask


@dataclass(frozen=True)
class PromptContext:
    """一次模型调用使用的五类上下文。"""

    system_rules: str
    user_input: str
    user_profile: dict[str, Any]
    goals: tuple[dict[str, Any], ...]
    pending_task: PendingTask | None
    verified_tool_names: tuple[str, ...]

    @property
    def layer_receipt(self) -> tuple[dict[str, str], ...]:
        """只返回来源与状态，不泄露健康参数或内部 Tool Result。"""

        return (
            {"layer": "系统规则", "source": "版本化应用规则", "status": "已装载"},
            {"layer": "本轮输入", "source": "用户当前消息", "status": "已装载"},
            {
                "layer": "用户档案",
                "source": "用户已确认的最小档案",
                "status": "已装载" if self.user_profile else "无数据",
            },
            {
                "layer": "目标与待办",
                "source": "活动目标与当前待补充任务",
                "status": "已装载" if self.goals or self.pending_task else "无数据",
            },
            {
                "layer": "可信结果",
                "source": "本轮已执行工具的结构化结果",
                "status": "已装载" if self.verified_tool_names else "等待工具",
            },
        )

    def render_system_message(self) -> str:
        profile = self.user_profile
        goal_lines = [
            f"- {goal['title']}：{goal['target_value']:g} {goal['unit']} / {goal['period']}"
            for goal in self.goals[:5]
        ]
        pending_text = "暂无"
        if self.pending_task is not None:
            missing = "、".join(self.pending_task.missing_parameters)
            pending_text = f"等待补充：{missing}；目标工具：{self.pending_task.tool_name}"
        verified = "、".join(self.verified_tool_names) or "暂无；不得把模型猜测当作事实"
        return (
            self.system_rules
            + "\n\n[Prompt Context Pipeline]\n"
            + "第 1 层｜系统规则：以上规则具有最高优先级。\n"
            + "第 2 层｜本轮用户输入：作为独立 user message 提供，不在此重复。\n"
            + "第 3 层｜用户已确认档案：\n"
            + f"- 时区：{profile.get('timezone_name', '未设置')}\n"
            + f"- 单位：{profile.get('unit_system', '未设置')}\n"
            + f"- 教练风格：{profile.get('coach_style', '未设置')}\n"
            + f"- 饮食偏好：{'、'.join(profile.get('dietary_preferences', [])) or '未设置'}\n"
            + f"- 忌口：{'、'.join(profile.get('exclusions', [])) or '未设置'}\n"
            + "第 4 层｜活动目标与待办：\n"
            + ("\n".join(goal_lines) if goal_lines else "- 暂无活动目标")
            + f"\n- {pending_text}\n"
            + "第 5 层｜可信上下文：\n"
            + f"- 本轮已验证工具结果：{verified}\n"
            + "仅使用与当前请求相关的最小信息；工具结果优先于模型记忆，"
            + "用户最新明确确认优先于旧会话内容。"
        )


def build_prompt_context(
    *,
    system_rules: str,
    user_input: str,
    profile_context: dict[str, Any],
    pending_task: PendingTask | None,
    messages: Sequence[AgentMessage],
) -> PromptContext:
    """从结构化状态构建稳定、可单测的五层上下文。"""

    latest_user_index = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=-1,
    )
    current_turn_messages = messages[latest_user_index + 1 :]
    return PromptContext(
        system_rules=system_rules,
        user_input=user_input.strip(),
        user_profile=dict(profile_context.get("profile", {})),
        goals=tuple(profile_context.get("goals", ())),
        pending_task=pending_task,
        verified_tool_names=tuple(
            dict.fromkeys(
                message.tool_name
                for message in current_turn_messages
                if message.role == "tool" and message.tool_name
            )
        ),
    )
