"""Agent 消息、状态、工具调用和运行结果模型。"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from enum import Enum
from typing import (
    Any,
    Literal,
    Protocol,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AgentBaseModel(BaseModel):
    """Agent 模型公共配置。"""

    model_config = ConfigDict(
        extra="forbid",
    )


class AgentState(
    str,
    Enum,
):
    """Agent 当前状态。"""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_CLARIFICATION = (
        "awaiting_clarification"
    )
    AWAITING_CONFIRMATION = (
        "awaiting_confirmation"
    )
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentFinishReason(
    str,
    Enum,
):
    """本轮结束原因。"""

    COMPLETED = "completed"
    NEEDS_CLARIFICATION = (
        "needs_clarification"
    )
    AWAITING_CONFIRMATION = (
        "awaiting_confirmation"
    )
    INVALID_ARGUMENTS = (
        "invalid_arguments"
    )
    TOOL_ERROR = "tool_error"
    LOOP_LIMIT = "loop_limit"
    CANCELLED = "cancelled"


class AgentMessage(
    AgentBaseModel
):
    """Provider 无关的 Agent 消息。"""

    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]
    content: str
    tool_call_id: (
        str
        | None
    ) = None
    tool_name: (
        str
        | None
    ) = None


class ModelToolCall(
    AgentBaseModel
):
    """模型提出的一次工具调用。"""

    call_id: str = Field(
        min_length=1,
        max_length=128,
    )
    name: str = Field(
        min_length=1,
        max_length=128,
    )
    arguments: dict[str, Any] = (
        Field(
            default_factory=dict
        )
    )

    @field_validator(
        "call_id",
        "name",
        mode="before",
    )
    @classmethod
    def strip_text(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return value.strip()

        return value


class AgentModelReply(
    AgentBaseModel
):
    """一次标准化模型响应。"""

    content: (
        str
        | None
    ) = None
    tool_calls: tuple[
        ModelToolCall,
        ...,
    ] = ()

    @model_validator(mode="after")
    def require_content_or_tool(
        self,
    ) -> "AgentModelReply":
        has_content = (
            isinstance(
                self.content,
                str,
            )
            and bool(
                self.content.strip()
            )
        )

        if (
            not has_content
            and not self.tool_calls
        ):
            raise ValueError(
                "模型响应必须包含文本"
                "或工具调用"
            )

        return self


class PendingTask(
    AgentBaseModel
):
    """等待下一轮补充参数的短期任务。"""

    tool_name: str
    arguments: dict[str, Any]
    missing_parameters: list[str]
    question: str


class PendingConfirmation(
    AgentBaseModel
):
    """等待用户明确确认的副作用草稿。"""

    action: Literal[
        "save",
        "update",
        "delete",
        "profile_update",
        "goal_change",
        "reminder_create",
        "reminder_change",
    ]
    tool_name: str
    draft_data: dict[str, Any]


class AgentToolStep(
    AgentBaseModel
):
    """一次真正执行过的工具步骤。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class AgentRunResult(
    AgentBaseModel
):
    """一次 Agent 回合返回给 UI 的结果。"""

    answer: str
    finish_reason: (
        AgentFinishReason
    )
    state: AgentState
    model_rounds: int = Field(
        ge=0
    )
    tool_steps: tuple[
        AgentToolStep,
        ...,
    ] = ()
    pending_task: (
        PendingTask
        | None
    ) = None
    pending_confirmation: (
        PendingConfirmation
        | None
    ) = None


class SessionState(
    AgentBaseModel
):
    """一个用户会话的短期状态。"""

    session_id: str = Field(
        min_length=1,
        max_length=128,
    )
    user_id: str = Field(
        min_length=1,
        max_length=128,
    )
    timezone_name: str = Field(
        default="Asia/Shanghai",
        min_length=1,
        max_length=100,
    )
    state: AgentState = (
        AgentState.IDLE
    )
    messages: tuple[
        AgentMessage,
        ...,
    ]
    turn_count: int = Field(
        default=0,
        ge=0,
    )
    pending_task: (
        PendingTask
        | None
    ) = None
    pending_confirmation: (
        PendingConfirmation
        | None
    ) = None


class AgentTurnOutcome(
    AgentBaseModel
):
    """Agent 本轮结果及新的会话状态。"""

    result: AgentRunResult
    session_state: SessionState


class AgentModel(Protocol):
    """Agent Runner 依赖的模型接口。"""

    def complete(
        self,
        messages: Sequence[
            AgentMessage
        ],
        tool_definitions: Sequence[
            dict[str, Any]
        ],
    ) -> AgentModelReply:
        """返回文本或一个工具调用。"""
