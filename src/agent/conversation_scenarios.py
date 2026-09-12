"""面向真实健康管理过程的多轮对话场景。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.agent.models import AgentMessage


@dataclass(frozen=True)
class ConversationScenario:
    """一次连续健康对话采用的业务场景策略。"""

    scenario_id: str
    title: str
    triggers: tuple[str, ...]
    objective: str
    guidance: tuple[str, ...]
    recommended_tools: tuple[str, ...]

    def render(self) -> str:
        """生成可注入系统上下文的简洁场景协议。"""

        steps = "\n".join(f"- {item}" for item in self.guidance)
        tools = "、".join(self.recommended_tools)
        return (
            f"当前业务场景：{self.title}（{self.scenario_id}）\n"
            f"本场景目标：{self.objective}\n"
            f"建议工具：{tools}\n"
            "对话策略：\n"
            f"{steps}"
        )


SCENARIOS: tuple[ConversationScenario, ...] = (
    ConversationScenario(
        scenario_id="safety_escalation",
        title="异常症状与安全分流",
        triggers=(
            "胸痛", "胸闷", "呼吸困难", "喘不上气", "晕厥", "昏厥",
            "意识不清", "剧烈头痛", "大出血", "急症", "自杀", "伤害自己",
            "严重过敏", "过敏性休克",
        ),
        objective="优先保护用户安全，不让普通健康陪伴延误紧急处理。",
        guidance=(
            "一旦出现紧急信号，立即停止饮食、运动、目标或习惯建议。",
            "不要等待用户补齐普通记录信息；清楚建议联系当地急救或立即就医。",
            "不得诊断、评估严重程度或承诺安全；只提供必要且可执行的安全行动。",
            "非紧急的一般健康问题也必须先检索可信知识并展示来源。",
        ),
        recommended_tools=("retrieve_health_knowledge",),
    ),
    ConversationScenario(
        scenario_id="goal_tradeoff",
        title="健康目标冲突与优先级协商",
        triggers=(
            "兼顾减脂和增肌", "减脂又增肌", "减重又增肌", "目标冲突",
            "两个目标", "哪个目标优先", "既想减重", "同时增肌",
        ),
        objective="帮助用户澄清优先级，形成不互相冲突且可确认的目标。",
        guidance=(
            "先读取现有目标，再询问当前最重要的结果、时间范围和可投入程度。",
            "不要替用户决定优先级，也不要承诺特定体重或体成分结果。",
            "只有用户明确选择后才准备目标变更；每项变更仍需单独确认。",
            "后续复盘以已确认事实和目标差距为依据，不把推测当作原因。",
        ),
        recommended_tools=("get_health_goals", "prepare_goal_change"),
    ),
    ConversationScenario(
        scenario_id="meal_guidance",
        title="当日餐食衔接与下一餐建议",
        triggers=(
            "下一餐", "晚餐吃什么", "午餐吃什么", "怎么搭配", "营养搭配",
            "今天还可以吃", "已经吃了", "接下来怎么吃", "下一顿",
        ),
        objective="把已确认的当日饮食与可信一般知识连接成可执行的下一餐选择。",
        guidance=(
            "先查询今天已确认的饮食事实；草稿和未记录内容不得计入。",
            "读取并遵守已确认的饮食偏好和忌口，信息不足时先询问。",
            "一般营养建议必须检索可信知识并展示来源，不根据单餐做诊断。",
            "给出少量可替换选择，并说明记录不完整时建议只能作为一般参考。",
        ),
        recommended_tools=(
            "get_health_events", "get_daily_summary", "get_user_profile",
            "retrieve_health_knowledge",
        ),
    ),
    ConversationScenario(
        scenario_id="period_review",
        title="7/14/30 天健康复盘",
        triggers=(
            "复盘", "最近一周", "过去一周", "这周情况", "7天", "14天",
            "30天", "最近一个月", "周期总结", "近期趋势",
        ),
        objective="先还原周期事实，再和用户共同选择下一周期的一项行动。",
        guidance=(
            "先确认复盘周期，再调用周期汇总；不得用聊天印象替代已保存事实。",
            "清楚区分观察到的变化、数据缺口和用户随后补充的主观感受。",
            "不得推断变化原因；可以询问哪些生活变化可能与用户体验有关。",
            "收尾时只聚焦一项可执行行动；涉及目标或提醒变更必须再次确认。",
        ),
        recommended_tools=(
            "get_period_summary", "get_health_goals", "prepare_goal_change",
        ),
    ),
    ConversationScenario(
        scenario_id="exercise_recovery",
        title="运动计划、完成反馈与恢复",
        triggers=(
            "运动计划", "训练计划", "训练后", "运动后", "恢复", "酸痛",
            "跑步计划", "力量训练", "练完", "训练疲劳", "运动目标",
        ),
        objective="连接运动事实、主观恢复感受和下一次可调整的行动。",
        guidance=(
            "先区分已完成的运动事实、用户主观感受和尚未执行的计划。",
            "普通恢复问题先检索可信知识；出现异常症状时立即切换安全分流。",
            "不要根据一次疲劳自动降低目标，先询问持续时间和用户意愿。",
            "记录、目标或提醒的任何改变都必须生成草稿并等待确认。",
        ),
        recommended_tools=(
            "get_health_events", "get_health_goals", "retrieve_health_knowledge",
            "prepare_goal_change",
        ),
    ),
    ConversationScenario(
        scenario_id="sleep_wellbeing",
        title="睡眠与疲劳改善",
        triggers=(
            "睡眠", "睡不着", "失眠", "早醒", "熬夜", "睡得不好",
            "睡了多久", "白天很困", "最近很累", "作息",
        ),
        objective="通过可信知识、用户观察和小步行动支持睡眠改善。",
        guidance=(
            "先了解持续时间、作息和白天影响；不要根据描述诊断睡眠障碍。",
            "一般建议必须检索可信知识；持续或严重影响生活时建议咨询专业人员。",
            "当前系统没有结构化睡眠事件，不得声称已保存睡眠时长或睡眠质量。",
            "可以在用户同意后创建作息提醒，后续询问体验并共同调整。",
        ),
        recommended_tools=("retrieve_health_knowledge", "create_reminder_draft"),
    ),
    ConversationScenario(
        scenario_id="weight_journey",
        title="体重目标陪伴与平台期复盘",
        triggers=(
            "体重趋势", "减重", "减脂", "平台期", "体重没变化", "体重不变",
            "控制体重", "体重目标", "瘦下来", "增重",
        ),
        objective="用周期事实帮助用户理解进展，并共同决定是否调整目标。",
        guidance=(
            "先读取周期体重事实和活动目标，说明记录数量与数据缺口。",
            "不得把短期波动解释为脂肪增减，也不得猜测平台期原因。",
            "询问饮食、活动、睡眠等主观变化时，明确它们是用户补充而非已证实原因。",
            "只有用户明确决定后才准备目标调整，并保留旧版本。",
        ),
        recommended_tools=(
            "get_period_summary", "get_health_goals", "prepare_goal_change",
        ),
    ),
    ConversationScenario(
        scenario_id="habit_followup",
        title="健康习惯建立与未完成跟进",
        triggers=(
            "养成习惯", "健康习惯", "坚持", "打卡", "总是忘", "没做到",
            "没有完成", "提醒时间", "每天提醒", "老是忘记",
        ),
        objective="不评判地了解阻碍，并把习惯缩小为用户愿意尝试的行动。",
        guidance=(
            "先核对已确认记录；缺少记录不等于用户没有完成。",
            "询问阻碍、合适时机和用户愿意尝试的最小行动，不使用羞耻或施压语言。",
            "提醒和主动 check-in 只能在用户选择时间和频率后创建，统一通过飞书发送。",
            "后续未完成时优先调整行动或提醒，而不是自动提高强度。",
        ),
        recommended_tools=(
            "get_daily_summary", "get_period_summary", "create_reminder_draft",
            "list_or_cancel_reminders",
        ),
    ),
    ConversationScenario(
        scenario_id="preference_continuity",
        title="饮食偏好与长期个性化",
        triggers=(
            "忌口", "我不吃", "不喜欢吃", "素食", "清真", "低盐偏好",
            "饮食偏好", "记住我", "以后不要推荐", "忘掉这个偏好",
        ),
        objective="让后续建议遵守用户明确确认的偏好，并保持可查看、可纠正。",
        guidance=(
            "先区分临时要求和希望长期记住的偏好，不自动写入档案。",
            "长期偏好必须生成档案变更草稿并等待确认；用户可再次修改或遗忘。",
            "后续餐食建议先读取档案，只使用已确认偏好，不根据对话自行推断。",
            "涉及过敏或医疗饮食时不扩大解释，必要时建议咨询专业人员。",
        ),
        recommended_tools=("get_user_profile", "prepare_profile_update"),
    ),
    ConversationScenario(
        scenario_id="daily_meal_tracking",
        title="一日饮食连续记录",
        triggers=(
            "早餐", "午餐", "晚餐", "加餐", "餐食", "今天吃了什么",
            "记录吃的", "一日饮食", "三餐",
        ),
        objective="连续记录当天餐食，并在每次确认后保持清楚的当日上下文。",
        guidance=(
            "每顿餐食独立完成候选匹配、份量估算、草稿和确认，不能沿用上一餐参数。",
            "用户询问当天饮食时只查询已确认记录，并明确未记录或未确认的部分。",
            "后续出现“这一顿”“刚才那餐”等指代时结合历史定位；有歧义就追问。",
            "只有用户请求建议时才进入餐食指导，不在记录完成后强行评价饮食。",
        ),
        recommended_tools=(
            "retrieve_nutrition_candidates", "calculate_nutrition",
            "prepare_health_event", "get_health_events",
        ),
    ),
)


def _match_score(text: str, scenario: ConversationScenario) -> int:
    """根据明确短语计算场景命中强度。"""

    return sum(1 for trigger in scenario.triggers if trigger in text)


def detect_conversation_scenario(
    user_input: str,
    messages: Sequence[AgentMessage] = (),
) -> ConversationScenario | None:
    """识别当前场景；省略式追问可继承最近六条用户消息的场景。"""

    normalized = user_input.strip().casefold()
    direct_scores = [(_match_score(normalized, scenario), scenario) for scenario in SCENARIOS]
    best_score, best = max(direct_scores, key=lambda item: item[0])
    if best_score > 0:
        return best

    recent_user_messages = [
        message.content.strip().casefold()
        for message in messages
        if message.role == "user" and message.content.strip()
    ][-6:]
    for previous in reversed(recent_user_messages):
        inherited_scores = [
            (_match_score(previous, scenario), scenario) for scenario in SCENARIOS
        ]
        inherited_score, inherited = max(inherited_scores, key=lambda item: item[0])
        if inherited_score > 0:
            return inherited
    return None
