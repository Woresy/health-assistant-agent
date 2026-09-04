"""Agent 静态工具白名单、参数校验和执行路由。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from src.agent.models import (
    PendingConfirmation,
)
from src.health.models import (
    EventType,
    ExerciseIntensity,
    InputSource,
)
from src.storage.jsonl_store import (
    HealthEventStore,
)
from src.storage.healthos_store import HealthOSStore, HealthOSStoreError
from src.nutrition.repository import FoodRepository
from src.tools.delete_health_event import (
    delete_health_event,
)
from src.tools.get_daily_health_summary import (
    get_daily_health_summary,
)
from src.tools.prepare_health_event import (
    prepare_health_event,
)
from src.tools.prepare_health_event_mutation import (
    prepare_delete_health_event,
    prepare_update_health_event,
)
from src.tools.query_health_events import (
    query_health_events,
)
from src.tools.save_health_event import (
    save_health_event,
)
from src.tools.update_health_event import (
    update_health_event,
)
from src.tools.healthos import (
    calculate_nutrition,
    create_reminder_draft,
    execute_healthos_confirmation,
    execute_reminder,
    get_health_events,
    get_health_goals,
    get_daily_summary,
    get_period_summary,
    get_user_profile,
    list_or_cancel_reminders,
    prepare_event_change,
    prepare_goal_change,
    prepare_profile_update,
    retrieve_health_knowledge,
    retrieve_nutrition_candidates,
)


ToolDispatchStatus = Literal[
    "executed",
    "needs_clarification",
    "invalid",
]


@dataclass(frozen=True)
class ToolDispatchResult:
    """工具路由的一次返回结果。"""

    status: ToolDispatchStatus
    tool_name: str
    arguments: dict[str, Any]
    result: (
        dict[str, Any]
        | None
    ) = None
    missing_parameters: tuple[
        str,
        ...,
    ] = ()
    question: str | None = None


class ToolInputModel(BaseModel):
    """工具输入公共配置。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class PrepareHealthEventArguments(
    ToolInputModel
):
    """模型调用 prepare_health_event 时使用的扁平参数。"""

    event_type: EventType
    occurred_at: str | None = None

    meal_payload: (
        dict[str, Any]
        | None
    ) = None
    source_refs: (
        list[str]
        | None
    ) = None
    input_source: (
        InputSource
        | None
    ) = Field(
        default=None,
        description=(
            "仅饮食事件使用。"
            "饮水、体重和运动事件"
            "不要传入该字段，"
            "来源由应用固定为 chat。"
        ),
    )

    amount_ml: float | None = Field(
        default=None,
        gt=0,
        le=10000,
    )
    beverage: str | None = None

    weight_kg: float | None = Field(
        default=None,
        gt=0,
        le=500,
    )

    activity_type: str | None = None
    duration_minutes: (
        float
        | None
    ) = Field(
        default=None,
        gt=0,
        le=1440,
    )
    distance_km: (
        float
        | None
    ) = Field(
        default=None,
        gt=0,
        le=1000,
    )
    intensity: (
        ExerciseIntensity
        | None
    ) = None

    note: str | None = None

    @model_validator(mode="after")
    def reject_irrelevant_fields(
        self,
    ) -> "PrepareHealthEventArguments":
        common_fields = {
            "event_type",
            "occurred_at",
        }

        allowed_by_type = {
            EventType.MEAL: {
                "meal_payload",
                "source_refs",
                "input_source",
            },
            EventType.WATER: {
                "amount_ml",
                "beverage",
                "note",
            },
            EventType.WEIGHT: {
                "weight_kg",
                "note",
            },
            EventType.EXERCISE: {
                "activity_type",
                "duration_minutes",
                "distance_km",
                "intensity",
                "note",
            },
        }

        allowed = (
            common_fields
            | allowed_by_type[
                self.event_type
            ]
        )

        unexpected = (
            self.model_fields_set
            - allowed
        )

        if unexpected:
            names = ", ".join(
                sorted(unexpected)
            )
            raise ValueError(
                f"{self.event_type.value} "
                "事件包含无关字段："
                f"{names}"
            )

        return self


class QueryEventsArguments(
    ToolInputModel
):
    """查询健康事件参数。"""

    event_type: (
        EventType
        | None
    ) = None
    date: str | None = None
    timezone_name: (
        str
        | None
    ) = None
    newest_first: bool = False
    limit: int = Field(
        default=100,
        ge=1,
        le=500,
    )


class DailySummaryArguments(
    ToolInputModel
):
    """每日汇总参数。"""

    date: str
    timezone_name: (
        str
        | None
    ) = None


class PrepareUpdateArguments(
    ToolInputModel
):
    """更新草稿参数。"""

    event_id: str
    patch: dict[str, Any] = Field(
        description=(
            "健康事件更新内容。"
            "occurred_at 和 source_refs "
            "位于 patch 顶层；"
            "健康数值必须放在 payload 中。"
            "例如修改体重应使用 "
            "{'payload': {'weight_kg': 64.8}}。"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_payload_fields(
        cls,
        data: Any,
    ) -> Any:
        """
        将模型常见的扁平健康字段归入 payload。

        领域工具仍然只接收标准嵌套结构。
        """

        if not isinstance(
            data,
            dict,
        ):
            return data

        raw_patch = data.get(
            "patch"
        )

        if not isinstance(
            raw_patch,
            dict,
        ):
            return data

        payload_field_names = {
            "amount_ml",
            "beverage",
            "weight_kg",
            "activity_type",
            "duration_minutes",
            "distance_km",
            "intensity",
            "note",
        }

        flat_fields = (
            payload_field_names
            & set(raw_patch)
        )

        if not flat_fields:
            return data

        normalized_data = dict(
            data
        )
        normalized_patch = dict(
            raw_patch
        )

        raw_payload = (
            normalized_patch.get(
                "payload"
            )
        )

        if raw_payload is None:
            normalized_payload: dict[
                str,
                Any,
            ] = {}
        elif isinstance(
            raw_payload,
            dict,
        ):
            normalized_payload = dict(
                raw_payload
            )
        else:
            raise ValueError(
                "patch.payload 必须是对象"
            )

        duplicate_fields = (
            flat_fields
            & set(normalized_payload)
        )

        if duplicate_fields:
            names = ", ".join(
                sorted(
                    duplicate_fields
                )
            )
            raise ValueError(
                "更新字段不能同时出现在 "
                "patch 顶层和 patch.payload："
                f"{names}"
            )

        for field_name in sorted(
            flat_fields
        ):
            normalized_payload[
                field_name
            ] = normalized_patch.pop(
                field_name
            )

        normalized_patch[
            "payload"
        ] = normalized_payload
        normalized_data[
            "patch"
        ] = normalized_patch

        return normalized_data


class PrepareDeleteArguments(
    ToolInputModel
):
    """删除草稿参数。"""

    event_id: str


class EmptyArguments(ToolInputModel):
    """无需模型参数的读取工具。"""


class ProfilePatchArguments(ToolInputModel):
    timezone_name: str | None = Field(default=None, min_length=1, max_length=100)
    coach_style: Literal["gentle", "rational", "concise", "goal_focused"] | None = None
    dietary_preferences: list[str] | None = Field(default=None, max_length=20)
    exclusions: list[str] | None = Field(default=None, max_length=20)
    reminders_enabled: bool | None = None
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None

    @field_validator("coach_style", mode="before")
    @classmethod
    def normalize_style_alias(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        return {
            "encouraging": "gentle",
            "warm": "gentle",
            "supportive": "gentle",
            "温和": "gentle",
            "温暖": "gentle",
            "鼓励": "gentle",
            "温和陪伴": "gentle",
            "analytical": "rational",
            "logical": "rational",
            "理性": "rational",
            "理性复盘": "rational",
            "brief": "concise",
            "simple": "concise",
            "简洁": "concise",
            "简洁提醒": "concise",
            "accountability": "goal_focused",
            "strict": "goal_focused",
            "目标督促": "goal_focused",
        }.get(normalized, normalized)

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "ProfilePatchArguments":
        if not self.model_fields_set:
            raise ValueError("请至少提供一项要修改的设置")
        return self


class ProfileUpdateArguments(ToolInputModel):
    patch: ProfilePatchArguments


class GoalChangeArguments(ToolInputModel):
    operation: Literal["create", "update", "pause", "resume"]
    goal_id: str | None = None
    title: str | None = None
    goal_type: Literal["water", "exercise", "weight", "nutrition", "custom"] | None = None
    target_value: float | None = Field(default=None, gt=0, le=1_000_000)
    unit: str | None = Field(default=None, min_length=1, max_length=30)
    period: Literal["daily", "weekly", "monthly", "8_weeks"] | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=500)


class PrepareEventChangeArguments(ToolInputModel):
    operation: Literal["update", "delete"]
    event_id: str
    patch: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_payload_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("operation") != "update":
            return data
        normalized = PrepareUpdateArguments.normalize_flat_payload_fields(
            {"event_id": data.get("event_id"), "patch": data.get("patch")}
        )
        return {**data, "patch": normalized.get("patch")}


class NutritionCandidatesArguments(ToolInputModel):
    query: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=10)


class NutritionCalculationArguments(ToolInputModel):
    food_code: str = Field(min_length=1, max_length=128)
    grams: float = Field(gt=0, le=10000)
    retrieval_query: str = Field(min_length=1, max_length=128)


class HealthKnowledgeArguments(ToolInputModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class PeriodSummaryArguments(ToolInputModel):
    days: Literal[7, 14, 30] = 7
    end_date: str | None = None
    timezone_name: str | None = None


class ReminderDraftArguments(ToolInputModel):
    content: str = Field(min_length=1, max_length=300)
    scheduled_for: str
    timezone_name: str | None = None


class ExecuteReminderArguments(ToolInputModel):
    draft: dict[str, Any]
    confirmation_token: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ReminderListOrChangeArguments(ToolInputModel):
    action: Literal["list", "cancel", "snooze", "pause", "resume"] = "list"
    reminder_id: str | None = None
    scheduled_for: str | None = None
    reason: str | None = Field(default=None, max_length=500)


def _definition(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _definition("get_user_profile", "读取最小必要用户档案、单位、时区和教练风格。", EmptyArguments),
    _definition(
        "prepare_profile_update",
        "生成档案或提醒偏好变更草稿；确认前不写入。教练风格：gentle=温和陪伴、rational=理性复盘、concise=简洁提醒、goal_focused=目标督促。",
        ProfileUpdateArguments,
    ),
    _definition("get_health_goals", "读取健康目标当前状态和完整版本历史。", EmptyArguments),
    _definition("prepare_goal_change", "创建、调整、暂停或恢复目标草稿；不覆盖旧版本。", GoalChangeArguments),
    _definition("get_health_events", "查询已经确认保存的饮食、饮水、体重和运动事实。", QueryEventsArguments),
    _definition("prepare_health_event", "生成健康记录保存草稿；缺参时追问，确认前不保存。", PrepareHealthEventArguments),
    _definition("prepare_event_change", "生成已有记录的修改前后对比或删除草稿。", PrepareEventChangeArguments),
    _definition("retrieve_nutrition_candidates", "检索 Top-K 标准食物候选并返回来源和分数。", NutritionCandidatesArguments),
    _definition("calculate_nutrition", "只使用选中食物数据行和克重确定性计算营养。", NutritionCalculationArguments),
    _definition("retrieve_health_knowledge", "检索带引用的一般健康知识；医疗或紧急问题拒答。", HealthKnowledgeArguments),
    _definition("get_daily_summary", "从已保存事实汇总指定日期并展示数据完整度。", DailySummaryArguments),
    _definition("get_period_summary", "汇总 7、14 或 30 天趋势事实；不推断原因。", PeriodSummaryArguments),
    _definition("create_reminder_draft", "生成本地提醒草稿并展示时间、时区和影响范围。", ReminderDraftArguments),
    _definition("execute_reminder", "仅凭有效确认令牌和幂等键执行提醒草稿。", ExecuteReminderArguments),
    _definition("list_or_cancel_reminders", "查看提醒；取消、延后、暂停或恢复时生成待确认草稿。", ReminderListOrChangeArguments),
)

TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "get_user_profile": {"risk_level": "read", "timeout_seconds": 5, "confirmation": False},
    "prepare_profile_update": {"risk_level": "draft", "timeout_seconds": 5, "confirmation": True},
    "get_health_goals": {"risk_level": "read", "timeout_seconds": 5, "confirmation": False},
    "prepare_goal_change": {"risk_level": "draft", "timeout_seconds": 5, "confirmation": True},
    "get_health_events": {"risk_level": "read", "timeout_seconds": 5, "confirmation": False},
    "prepare_health_event": {"risk_level": "draft", "timeout_seconds": 5, "confirmation": True},
    "prepare_event_change": {"risk_level": "draft", "timeout_seconds": 5, "confirmation": True},
    "retrieve_nutrition_candidates": {"risk_level": "retrieval", "timeout_seconds": 15, "confirmation": False},
    "calculate_nutrition": {"risk_level": "calculation", "timeout_seconds": 5, "confirmation": False},
    "retrieve_health_knowledge": {"risk_level": "retrieval", "timeout_seconds": 5, "confirmation": False},
    "get_daily_summary": {"risk_level": "read", "timeout_seconds": 5, "confirmation": False},
    "get_period_summary": {"risk_level": "read", "timeout_seconds": 5, "confirmation": False},
    "create_reminder_draft": {"risk_level": "draft", "timeout_seconds": 5, "confirmation": True},
    "execute_reminder": {"risk_level": "write", "timeout_seconds": 5, "confirmation": True},
    "list_or_cancel_reminders": {"risk_level": "read_or_draft", "timeout_seconds": 5, "confirmation": "write_only"},
}


_TOOL_ALIASES = {
    "query_health_events": "get_health_events",
    "get_daily_health_summary": "get_daily_summary",
    "prepare_update_health_event": "prepare_event_change",
    "prepare_delete_health_event": "prepare_event_change",
}


_FOLLOW_UP_QUESTIONS = {
    "event_type": (
        "你想记录饮食、饮水、"
        "体重还是运动？"
    ),
    "meal_payload": (
        "请先完成食物候选选择、"
        "份量填写和营养计算。"
    ),
    "source_refs": (
        "饮食记录还缺少营养数据来源。"
    ),
    "amount_ml": (
        "这次喝了多少毫升？"
    ),
    "weight_kg": (
        "这次体重是多少公斤？"
    ),
    "activity_type": (
        "这次做了什么运动？"
    ),
    "duration_minutes": (
        "这次运动持续了多少分钟？"
    ),
    "date": (
        "你想查询哪一天？"
        "请使用 YYYY-MM-DD。"
    ),
    "event_id": (
        "请先选择要修改或删除的"
        "具体记录。"
    ),
    "patch": (
        "你想修改这条记录的"
        "哪些内容？"
    ),
    "goal_id": "请先选择要调整的健康目标。",
    "profile_patch": "你想修改哪项档案或提醒偏好？",
    "goal_fields": "请补充目标名称、数值、单位和周期。",
    "reminder_content": "你希望我提醒什么？",
    "scheduled_for": "请告诉我提醒的具体日期和时间。",
}


def _tool_error(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """生成统一工具错误。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def _is_missing(
    arguments: dict[str, Any],
    field_name: str,
) -> bool:
    """判断参数是否缺失或为空。"""

    if field_name not in arguments:
        return True

    value = arguments[
        field_name
    ]

    if value is None:
        return True

    if (
        isinstance(value, str)
        and not value.strip()
    ):
        return True

    if (
        isinstance(
            value,
            (dict, list),
        )
        and not value
    ):
        return True

    return False


def _find_missing_parameters(
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, ...]:
    """按照工具和事件类型确定必填参数。"""

    canonical_name = _TOOL_ALIASES.get(tool_name, tool_name)

    if canonical_name == (
        "prepare_health_event"
    ):
        if _is_missing(
            arguments,
            "event_type",
        ):
            return (
                "event_type",
            )

        try:
            event_type = EventType(
                arguments[
                    "event_type"
                ]
            )
        except (
            ValueError,
            TypeError,
        ):
            return ()

        required_by_type = {
            EventType.MEAL: (
                "meal_payload",
                "source_refs",
            ),
            EventType.WATER: (
                "amount_ml",
            ),
            EventType.WEIGHT: (
                "weight_kg",
            ),
            EventType.EXERCISE: (
                "activity_type",
                "duration_minutes",
            ),
        }

        return tuple(
            field_name
            for field_name
            in required_by_type[
                event_type
            ]
            if _is_missing(
                arguments,
                field_name,
            )
        )

    if canonical_name == (
        "get_daily_summary"
    ):
        return (
            ("date",)
            if _is_missing(
                arguments,
                "date",
            )
            else ()
        )

    if tool_name == (
        "prepare_update_health_event"
    ):
        return tuple(
            field_name
            for field_name in (
                "event_id",
                "patch",
            )
            if _is_missing(
                arguments,
                field_name,
            )
        )

    if tool_name == (
        "prepare_delete_health_event"
    ):
        return (
            ("event_id",)
            if _is_missing(
                arguments,
                "event_id",
            )
            else ()
        )

    if canonical_name == "prepare_profile_update" and _is_missing(arguments, "patch"):
        return ("profile_patch",)

    if canonical_name == "prepare_goal_change":
        operation = str(arguments.get("operation", "")).strip().lower()
        if operation == "create" and any(
            _is_missing(arguments, field)
            for field in ("title", "goal_type", "target_value", "unit", "period")
        ):
            return ("goal_fields",)
        if operation in {"update", "pause", "resume"} and _is_missing(arguments, "goal_id"):
            return ("goal_id",)

    if canonical_name == "prepare_event_change":
        if _is_missing(arguments, "event_id"):
            return ("event_id",)
        if arguments.get("operation") == "update" and _is_missing(arguments, "patch"):
            return ("patch",)

    if canonical_name == "create_reminder_draft":
        missing = []
        if _is_missing(arguments, "content"):
            missing.append("reminder_content")
        if _is_missing(arguments, "scheduled_for"):
            missing.append("scheduled_for")
        return tuple(missing)

    return ()


def _build_question(
    missing_parameters: tuple[
        str,
        ...,
    ],
) -> str:
    """按固定顺序生成缺参追问。"""

    questions = [
        _FOLLOW_UP_QUESTIONS[
            parameter
        ]
        for parameter
        in missing_parameters
    ]

    return "\n".join(
        questions
    )


def _idempotency_key(
    *,
    session_id: str,
    call_id: str,
) -> str:
    """为一个已经就绪的草稿生成稳定幂等键。"""

    digest = hashlib.sha256(
        (
            f"{session_id}:"
            f"{call_id}"
        ).encode("utf-8")
    ).hexdigest()

    return (
        f"agent-{digest}"
    )


class HealthToolRouter:
    """只执行静态白名单中的健康工具。"""

    def __init__(
        self,
        store: HealthEventStore,
        *,
        healthos_store: HealthOSStore | None = None,
        nutrition_repository: FoodRepository | None = None,
    ) -> None:
        self._store = store
        self._healthos_store = healthos_store or HealthOSStore(
            Path(__file__).resolve().parents[2] / "data" / "healthos_state.json"
        )
        self._nutrition_repository = nutrition_repository or FoodRepository()
        self._available_tools = tuple(
            item["function"]["name"] for item in TOOL_DEFINITIONS
        )
        self._accepted_tools = (*self._available_tools, *_TOOL_ALIASES)

    @property
    def available_tools(
        self,
    ) -> tuple[str, ...]:
        """返回模型允许提出的工具名。"""

        return self._available_tools

    @property
    def tool_definitions(
        self,
    ) -> tuple[
        dict[str, Any],
        ...,
    ]:
        """返回 Provider 可使用的工具 Schema。"""

        return TOOL_DEFINITIONS

    @property
    def tool_contracts(self) -> dict[str, dict[str, Any]]:
        """返回风险、超时预算与确认策略，供验收和 UI 展示。"""

        return {name: dict(contract) for name, contract in TOOL_CONTRACTS.items()}

    def minimal_user_context(self, *, user_id: str, timezone_name: str) -> str:
        """只组装当前任务可用的最小档案与活动目标。"""

        try:
            profile = self._healthos_store.get_profile(user_id, timezone_name)
            goals = [
                goal.current
                for goal in self._healthos_store.read().goals
                if goal.user_id == user_id and goal.current.status.value == "active"
            ]
        except (HealthOSStoreError, ValueError):
            return ""
        goal_lines = [
            f"- {goal.title}：{goal.target_value:g} {goal.unit} / {goal.period}"
            for goal in goals[:5]
        ]
        return (
            "\n\n当前已确认的最小用户上下文：\n"
            f"- 时区：{profile.timezone_name}\n"
            f"- 单位：{profile.unit_system}\n"
            f"- 教练风格：{profile.coach_style.value}\n"
            f"- 饮食偏好：{'、'.join(profile.dietary_preferences) or '未设置'}\n"
            f"- 忌口：{'、'.join(profile.exclusions) or '未设置'}\n"
            "- 活动目标：\n"
            + ("\n".join(goal_lines) if goal_lines else "  暂无")
            + "\n只在当前请求相关时使用这些信息，不得推断未确认属性。"
        )

    def dispatch(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str,
        timezone_name: str,
        session_id: str,
        call_id: str,
    ) -> ToolDispatchResult:
        """校验并执行一个白名单工具。"""

        if (
            tool_name
            not in self._accepted_tools
        ):
            return ToolDispatchResult(
                status="invalid",
                tool_name=tool_name,
                arguments=arguments,
                result=_tool_error(
                    "UNKNOWN_TOOL",
                    f"工具不在白名单中："
                    f"{tool_name}",
                ),
            )

        canonical_name = _TOOL_ALIASES.get(tool_name, tool_name)
        canonical_arguments = dict(arguments)
        if tool_name == "prepare_update_health_event":
            canonical_arguments = {**canonical_arguments, "operation": "update"}
        elif tool_name == "prepare_delete_health_event":
            canonical_arguments = {**canonical_arguments, "operation": "delete"}

        missing_parameters = (
            _find_missing_parameters(
                tool_name,
                canonical_arguments,
            )
        )

        if missing_parameters:
            return ToolDispatchResult(
                status=(
                    "needs_clarification"
                ),
                tool_name=tool_name,
                arguments=arguments,
                missing_parameters=(
                    missing_parameters
                ),
                question=_build_question(
                    missing_parameters
                ),
            )

        try:
            if canonical_name == (
                "prepare_health_event"
            ):
                validation_arguments = dict(
                    canonical_arguments
                )

                if (
                    validation_arguments.get(
                        "event_type"
                    )
                    != EventType.MEAL
                ):
                    validation_arguments.pop(
                        "input_source",
                        None,
                    )

                validated = (
                    PrepareHealthEventArguments
                    .model_validate(
                        validation_arguments
                    )
                )

                flat_data = (
                    validated.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                )

                event_type = (
                    flat_data.pop(
                        "event_type"
                    )
                )

                event_input: dict[
                    str,
                    Any,
                ] = {
                    "event_type": (
                        event_type
                    ),
                }

                if (
                    "occurred_at"
                    in flat_data
                ):
                    event_input[
                        "occurred_at"
                    ] = flat_data.pop(
                        "occurred_at"
                    )

                if (
                    event_type
                    == "meal"
                ):
                    event_input[
                        "payload"
                    ] = flat_data.pop(
                        "meal_payload"
                    )
                    event_input[
                        "source_refs"
                    ] = flat_data.pop(
                        "source_refs"
                    )
                    event_input[
                        "input_source"
                    ] = flat_data.pop(
                        "input_source",
                        "image",
                    )
                else:
                    event_input.update(
                        flat_data
                    )

                result = (
                    prepare_health_event(
                        event_input=(
                            event_input
                        ),
                        user_id=user_id,
                        idempotency_key=(
                            _idempotency_key(
                                session_id=(
                                    session_id
                                ),
                                call_id=call_id,
                            )
                        ),
                    )
                )

            elif canonical_name == "get_user_profile":
                EmptyArguments.model_validate(canonical_arguments)
                result = get_user_profile(
                    user_id=user_id,
                    timezone_name=timezone_name,
                    store=self._healthos_store,
                )

            elif canonical_name == "prepare_profile_update":
                validated = ProfileUpdateArguments.model_validate(canonical_arguments)
                result = prepare_profile_update(
                    user_id=user_id,
                    timezone_name=timezone_name,
                    patch=validated.patch.model_dump(mode="json", exclude_unset=True),
                    idempotency_key=_idempotency_key(session_id=session_id, call_id=call_id),
                    store=self._healthos_store,
                )

            elif canonical_name == "get_health_goals":
                EmptyArguments.model_validate(canonical_arguments)
                result = get_health_goals(user_id=user_id, store=self._healthos_store)

            elif canonical_name == "prepare_goal_change":
                validated = GoalChangeArguments.model_validate(canonical_arguments)
                result = prepare_goal_change(
                    user_id=user_id,
                    idempotency_key=_idempotency_key(session_id=session_id, call_id=call_id),
                    store=self._healthos_store,
                    **validated.model_dump(mode="json", exclude_none=True),
                )

            elif canonical_name == (
                "get_health_events"
            ):
                validated = (
                    QueryEventsArguments
                    .model_validate(
                        canonical_arguments
                    )
                )

                result = (
                    get_health_events(
                        user_id=user_id,
                        event_type=(
                            validated
                            .event_type
                            .value
                            if validated
                            .event_type
                            is not None
                            else None
                        ),
                        date=validated.date,
                        timezone_name=(
                            validated
                            .timezone_name
                            or timezone_name
                        ),
                        newest_first=(
                            validated
                            .newest_first
                        ),
                        limit=(
                            validated.limit
                        ),
                        store=self._store,
                    )
                )

            elif canonical_name == (
                "get_daily_summary"
            ):
                validated = (
                    DailySummaryArguments
                    .model_validate(
                        canonical_arguments
                    )
                )

                result = (
                    get_daily_summary(
                        user_id=user_id,
                        date=validated.date,
                        timezone_name=(
                            validated
                            .timezone_name
                            or timezone_name
                        ),
                        store=self._store,
                        healthos_store=self._healthos_store,
                    )
                )

            elif canonical_name == "prepare_event_change":
                validated = PrepareEventChangeArguments.model_validate(canonical_arguments)
                result = prepare_event_change(
                    operation=validated.operation,
                    event_id=validated.event_id,
                    user_id=user_id,
                    patch=validated.patch,
                    idempotency_key=_idempotency_key(session_id=session_id, call_id=call_id),
                    store=self._store,
                )

            elif canonical_name == "retrieve_nutrition_candidates":
                validated = NutritionCandidatesArguments.model_validate(canonical_arguments)
                result = retrieve_nutrition_candidates(
                    query=validated.query,
                    top_k=validated.top_k,
                    repository=self._nutrition_repository,
                )

            elif canonical_name == "calculate_nutrition":
                validated = NutritionCalculationArguments.model_validate(canonical_arguments)
                result = calculate_nutrition(
                    food_code=validated.food_code,
                    grams=validated.grams,
                    retrieval_query=validated.retrieval_query,
                    repository=self._nutrition_repository,
                )

            elif canonical_name == "retrieve_health_knowledge":
                validated = HealthKnowledgeArguments.model_validate(canonical_arguments)
                result = retrieve_health_knowledge(
                    question=validated.question,
                    top_k=validated.top_k,
                )

            elif canonical_name == "get_period_summary":
                validated = PeriodSummaryArguments.model_validate(canonical_arguments)
                result = get_period_summary(
                    user_id=user_id,
                    days=validated.days,
                    end_date=validated.end_date,
                    timezone_name=validated.timezone_name or timezone_name,
                    store=self._store,
                    healthos_store=self._healthos_store,
                )

            elif canonical_name == "create_reminder_draft":
                validated = ReminderDraftArguments.model_validate(canonical_arguments)
                result = create_reminder_draft(
                    user_id=user_id,
                    content=validated.content,
                    scheduled_for=validated.scheduled_for,
                    timezone_name=validated.timezone_name or timezone_name,
                    idempotency_key=_idempotency_key(session_id=session_id, call_id=call_id),
                    store=self._healthos_store,
                )

            elif canonical_name == "execute_reminder":
                validated = ExecuteReminderArguments.model_validate(canonical_arguments)
                result = execute_reminder(
                    user_id=user_id,
                    draft=validated.draft,
                    confirmation_token=validated.confirmation_token,
                    idempotency_key=validated.idempotency_key,
                    store=self._healthos_store,
                )

            else:
                validated = ReminderListOrChangeArguments.model_validate(canonical_arguments)
                result = list_or_cancel_reminders(
                    user_id=user_id,
                    idempotency_key=_idempotency_key(session_id=session_id, call_id=call_id),
                    store=self._healthos_store,
                    **validated.model_dump(mode="json", exclude_none=True),
                )

        except ValidationError as exc:
            if canonical_name == "prepare_profile_update":
                validation_message = (
                    "这项个人设置无法识别。教练风格可以选择："
                    "温和陪伴、理性复盘、简洁提醒或目标督促；"
                    "免打扰时间请使用 HH:MM。"
                )
            else:
                validation_message = f"工具参数校验失败：{exc}"
            return ToolDispatchResult(
                status="invalid",
                tool_name=tool_name,
                arguments=arguments,
                result=_tool_error(
                    "VALIDATION_ERROR",
                    validation_message,
                ),
            )
        except Exception as exc:
            return ToolDispatchResult(
                status="invalid",
                tool_name=tool_name,
                arguments=arguments,
                result=_tool_error(
                    "TOOL_EXECUTION_ERROR",
                    f"工具执行失败：{type(exc).__name__}",
                ),
            )

        return ToolDispatchResult(
            status="executed",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    def confirm(
        self,
        pending: PendingConfirmation,
    ) -> dict[str, Any]:
        """用户明确确认后执行真正的写工具。"""

        data = pending.draft_data

        if pending.action == "save":
            return save_health_event(
                event_input=data["event"],
                confirmation_token=(
                    data[
                        "confirmation_token"
                    ]
                ),
                idempotency_key=(
                    data[
                        "idempotency_key"
                    ]
                ),
                store=self._store,
            )

        if pending.action == "update":
            return update_health_event(
                event_id=data["event_id"],
                user_id=(
                    data[
                        "proposed_event"
                    ]["user_id"]
                ),
                replacement_event_input=(
                    data[
                        "proposed_event"
                    ]
                ),
                confirmation_token=(
                    data[
                        "confirmation_token"
                    ]
                ),
                idempotency_key=(
                    data[
                        "idempotency_key"
                    ]
                ),
                store=self._store,
            )

        if pending.action == "delete":
            target_event = data[
                "target_event"
            ]

            return delete_health_event(
                event_id=data["event_id"],
                user_id=(
                    target_event[
                        "user_id"
                    ]
                ),
                confirmation_token=(
                    data[
                        "confirmation_token"
                    ]
                ),
                idempotency_key=(
                    data[
                        "idempotency_key"
                    ]
                ),
                store=self._store,
            )

        if pending.action in {
            "profile_update",
            "goal_change",
            "reminder_create",
            "reminder_change",
        }:
            return execute_healthos_confirmation(
                draft_data=data,
                store=self._healthos_store,
            )

        return _tool_error(
            "UNSUPPORTED_CONFIRMATION",
            "不支持的确认操作",
        )
