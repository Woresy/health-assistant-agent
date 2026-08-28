"""Agent 静态工具白名单、参数校验和执行路由。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
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
    ) = None

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
    patch: dict[str, Any]


class PrepareDeleteArguments(
    ToolInputModel
):
    """删除草稿参数。"""

    event_id: str


TOOL_DEFINITIONS: tuple[
    dict[str, Any],
    ...,
] = (
    {
        "type": "function",
        "function": {
            "name": (
                "prepare_health_event"
            ),
            "description": (
                "生成饮食、饮水、体重或"
                "运动事件保存草稿。"
                "只生成草稿，不执行保存。"
            ),
            "parameters": (
                PrepareHealthEventArguments
                .model_json_schema()
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": (
                "query_health_events"
            ),
            "description": (
                "查询用户已经保存的"
                "健康事件时间线。"
            ),
            "parameters": (
                QueryEventsArguments
                .model_json_schema()
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": (
                "get_daily_health_summary"
            ),
            "description": (
                "读取已保存事件并生成"
                "指定日期的确定性汇总。"
            ),
            "parameters": (
                DailySummaryArguments
                .model_json_schema()
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": (
                "prepare_update_health_event"
            ),
            "description": (
                "生成事件更新前后对比。"
                "只生成草稿，不执行更新。"
            ),
            "parameters": (
                PrepareUpdateArguments
                .model_json_schema()
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": (
                "prepare_delete_health_event"
            ),
            "description": (
                "展示待删除事件并生成草稿。"
                "只生成草稿，不执行删除。"
            ),
            "parameters": (
                PrepareDeleteArguments
                .model_json_schema()
            ),
        },
    },
)


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

    if tool_name == (
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

    if tool_name == (
        "get_daily_health_summary"
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
    ) -> None:
        self._store = store
        self._available_tools = (
            "prepare_health_event",
            "query_health_events",
            "get_daily_health_summary",
            "prepare_update_health_event",
            "prepare_delete_health_event",
        )

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
            not in self._available_tools
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

        missing_parameters = (
            _find_missing_parameters(
                tool_name,
                arguments,
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
            if tool_name == (
                "prepare_health_event"
            ):
                validated = (
                    PrepareHealthEventArguments
                    .model_validate(
                        arguments
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

            elif tool_name == (
                "query_health_events"
            ):
                validated = (
                    QueryEventsArguments
                    .model_validate(
                        arguments
                    )
                )

                result = (
                    query_health_events(
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

            elif tool_name == (
                "get_daily_health_summary"
            ):
                validated = (
                    DailySummaryArguments
                    .model_validate(
                        arguments
                    )
                )

                result = (
                    get_daily_health_summary(
                        user_id=user_id,
                        date=validated.date,
                        timezone_name=(
                            validated
                            .timezone_name
                            or timezone_name
                        ),
                        store=self._store,
                    )
                )

            elif tool_name == (
                "prepare_update_health_event"
            ):
                validated = (
                    PrepareUpdateArguments
                    .model_validate(
                        arguments
                    )
                )

                result = (
                    prepare_update_health_event(
                        event_id=(
                            validated.event_id
                        ),
                        user_id=user_id,
                        patch=validated.patch,
                        idempotency_key=(
                            _idempotency_key(
                                session_id=(
                                    session_id
                                ),
                                call_id=call_id,
                            )
                        ),
                        store=self._store,
                    )
                )

            else:
                validated = (
                    PrepareDeleteArguments
                    .model_validate(
                        arguments
                    )
                )

                result = (
                    prepare_delete_health_event(
                        event_id=(
                            validated.event_id
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
                        store=self._store,
                    )
                )

        except ValidationError as exc:
            return ToolDispatchResult(
                status="invalid",
                tool_name=tool_name,
                arguments=arguments,
                result=_tool_error(
                    "VALIDATION_ERROR",
                    f"工具参数校验失败：{exc}",
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

        return _tool_error(
            "UNSUPPORTED_CONFIRMATION",
            "不支持的确认操作",
        )