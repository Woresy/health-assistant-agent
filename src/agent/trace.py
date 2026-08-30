"""Agent 运行轨迹的脱敏记录与持久化。

只记录 Agent 的运行元数据，不记录：

- 用户原始输入；
- 体重、饮水量、运动时长等参数值；
- 模型完整回答；
- 工具完整返回值；
- confirmation_token。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.agent.models import (
    AgentRunResult,
    AgentToolStep,
)
from src.agent.runner import (
    AgentRunner,
    ConversationSession,
)


PROJECT_ROOT = (
    Path(__file__).resolve()
    .parents[2]
)

DEFAULT_AGENT_TRACE_PATH = (
    PROJECT_ROOT
    / "data"
    / "agent_traces.jsonl"
)

TraceAction = Literal[
    "send",
    "confirm",
    "cancel",
]

ConfirmationAction = Literal[
    "save",
    "update",
    "delete",
]


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""

    return datetime.now(
        timezone.utc
    )


class AgentTraceWriteError(
    RuntimeError
):
    """Agent Trace 写入失败。"""

    error_code = (
        "AGENT_TRACE_WRITE_FAILED"
    )


class AgentTraceReadError(
    RuntimeError
):
    """Agent Trace 读取失败。"""

    error_code = (
        "AGENT_TRACE_READ_FAILED"
    )


class AgentTraceToolStep(
    BaseModel
):
    """一次工具调用的脱敏信息。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    tool_name: str = Field(
        min_length=1,
        max_length=120,
    )

    argument_names: tuple[
        str,
        ...,
    ] = ()

    ok: bool | None = None

    error_code: str | None = None


class AgentTrace(
    BaseModel
):
    """一次 Agent 操作对应的 Trace。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    schema_version: Literal[
        "1.0"
    ] = "1.0"

    created_at: str

    session_hash: str = Field(
        min_length=64,
        max_length=64,
    )

    user_hash: str = Field(
        min_length=64,
        max_length=64,
    )

    action: TraceAction

    turn_count: int = Field(
        ge=0,
    )

    input_length: int | None = Field(
        default=None,
        ge=0,
    )

    input_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )

    state: str

    finish_reason: str

    model_rounds: int = Field(
        ge=0,
    )

    duration_ms: float = Field(
        ge=0,
    )

    tool_steps: tuple[
        AgentTraceToolStep,
        ...,
    ] = ()

    has_pending_task: bool = False

    pending_tool_name: (
        str
        | None
    ) = None

    has_pending_confirmation: bool = (
        False
    )

    confirmation_action: (
        ConfirmationAction
        | None
    ) = None

    error_type: (
        str
        | None
    ) = None


class AgentTraceSink(
    Protocol
):
    """Trace 存储需要实现的接口。"""

    def append(
        self,
        trace: AgentTrace,
    ) -> None:
        """保存一条 Trace。"""


class AgentTraceStore:
    """使用 JSONL 文件保存 Agent Trace。"""

    def __init__(
        self,
        path: (
            str
            | Path
        ) = DEFAULT_AGENT_TRACE_PATH,
    ) -> None:
        self.path = Path(
            path
        )

    def append(
        self,
        trace: AgentTrace,
    ) -> None:
        """追加 Trace，失败时回滚半条记录。"""

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        line = (
            trace.model_dump_json()
            + "\n"
        )
        encoded = line.encode(
            "utf-8"
        )

        file_descriptor: (
            int
            | None
        ) = None
        original_size = 0

        try:
            file_descriptor = os.open(
                self.path,
                (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_APPEND
                ),
                0o600,
            )

            original_size = os.lseek(
                file_descriptor,
                0,
                os.SEEK_END,
            )

            written = os.write(
                file_descriptor,
                encoded,
            )

            if written != len(
                encoded
            ):
                os.ftruncate(
                    file_descriptor,
                    original_size,
                )

                raise AgentTraceWriteError(
                    "Agent Trace "
                    "没有完整写入。"
                )

            os.fsync(
                file_descriptor
            )

        except AgentTraceWriteError:
            raise

        except OSError as exc:
            if (
                file_descriptor
                is not None
            ):
                try:
                    os.ftruncate(
                        file_descriptor,
                        original_size,
                    )
                except OSError:
                    pass

            raise AgentTraceWriteError(
                "Agent Trace "
                "文件写入失败。"
            ) from exc

        finally:
            if (
                file_descriptor
                is not None
            ):
                os.close(
                    file_descriptor
                )

    def read_recent(
        self,
        limit: int = 20,
    ) -> tuple[
        AgentTrace,
        ...,
    ]:
        """读取最近的 Trace，从新到旧排列。"""

        if (
            isinstance(
                limit,
                bool,
            )
            or not isinstance(
                limit,
                int,
            )
        ):
            raise ValueError(
                "limit 必须是正整数。"
            )

        if limit <= 0:
            raise ValueError(
                "limit 必须大于 0。"
            )

        if not self.path.exists():
            return ()

        try:
            raw_text = (
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except OSError as exc:
            raise AgentTraceReadError(
                "Agent Trace "
                "文件读取失败。"
            ) from exc

        traces: list[
            AgentTrace
        ] = []

        for (
            line_number,
            raw_line,
        ) in enumerate(
            raw_text.splitlines(),
            start=1,
        ):
            stripped = (
                raw_line.strip()
            )

            if not stripped:
                continue

            try:
                trace = (
                    AgentTrace
                    .model_validate_json(
                        stripped
                    )
                )
            except ValidationError as exc:
                raise AgentTraceReadError(
                    "Agent Trace "
                    f"第 {line_number} "
                    "行格式错误。"
                ) from exc

            traces.append(
                trace
            )

        return tuple(
            reversed(
                traces[-limit:]
            )
        )


def _hash_text(
    value: str,
) -> str:
    """计算文本 SHA-256。"""

    return sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def _safe_tool_steps(
    tool_steps: tuple[
        AgentToolStep,
        ...,
    ],
) -> tuple[
    AgentTraceToolStep,
    ...,
]:
    """只提取工具步骤中的安全字段。"""

    safe_steps: list[
        AgentTraceToolStep
    ] = []

    for step in tool_steps:
        result = (
            step.result
        )

        raw_ok = result.get(
            "ok"
        )

        ok: (
            bool
            | None
        )

        if isinstance(
            raw_ok,
            bool,
        ):
            ok = raw_ok
        else:
            ok = None

        error_code: (
            str
            | None
        ) = None

        raw_error = result.get(
            "error"
        )

        if isinstance(
            raw_error,
            dict,
        ):
            raw_error_code = (
                raw_error.get(
                    "error_code"
                )
            )

            if isinstance(
                raw_error_code,
                str,
            ):
                error_code = (
                    raw_error_code
                )

        safe_steps.append(
            AgentTraceToolStep(
                tool_name=(
                    step.tool_name
                ),
                argument_names=tuple(
                    sorted(
                        str(name)
                        for name
                        in step.arguments
                        .keys()
                    )
                ),
                ok=ok,
                error_code=(
                    error_code
                ),
            )
        )

    return tuple(
        safe_steps
    )


class TracedConversationSession(
    ConversationSession
):
    """自动写入脱敏 Trace 的 Agent 会话。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        session_id: str,
        user_id: str,
        timezone_name: str = (
            "Asia/Shanghai"
        ),
        trace_store: AgentTraceSink,
    ) -> None:
        super().__init__(
            runner=runner,
            session_id=session_id,
            user_id=user_id,
            timezone_name=(
                timezone_name
            ),
        )

        self._trace_store = (
            trace_store
        )

        self._last_trace_warning: (
            str
            | None
        ) = None

    @property
    def last_trace_warning(
        self,
    ) -> str | None:
        """返回最近一次 Trace 写入警告。"""

        return (
            self._last_trace_warning
        )

    def send(
        self,
        user_text: str,
    ) -> AgentRunResult:
        """发送消息并记录 Trace。"""

        started_at = (
            perf_counter()
        )

        try:
            result = super().send(
                user_text
            )
        except Exception as exc:
            self._record_exception(
                action="send",
                started_at=(
                    started_at
                ),
                user_text=(
                    user_text
                ),
                error=exc,
                confirmation_action=None,
            )
            raise

        self._record_result(
            action="send",
            started_at=(
                started_at
            ),
            result=result,
            user_text=user_text,
            confirmation_action=None,
        )

        return result

    def confirm(
        self,
    ) -> AgentRunResult:
        """确认草稿并记录 Trace。"""

        started_at = (
            perf_counter()
        )

        confirmation_action = (
            self
            ._current_confirmation_action()
        )

        try:
            result = (
                super().confirm()
            )
        except Exception as exc:
            self._record_exception(
                action="confirm",
                started_at=(
                    started_at
                ),
                user_text=None,
                error=exc,
                confirmation_action=(
                    confirmation_action
                ),
            )
            raise

        self._record_result(
            action="confirm",
            started_at=(
                started_at
            ),
            result=result,
            user_text=None,
            confirmation_action=(
                confirmation_action
            ),
        )

        return result

    def cancel(
        self,
    ) -> AgentRunResult:
        """取消操作并记录 Trace。"""

        started_at = (
            perf_counter()
        )

        confirmation_action = (
            self
            ._current_confirmation_action()
        )

        try:
            result = (
                super().cancel()
            )
        except Exception as exc:
            self._record_exception(
                action="cancel",
                started_at=(
                    started_at
                ),
                user_text=None,
                error=exc,
                confirmation_action=(
                    confirmation_action
                ),
            )
            raise

        self._record_result(
            action="cancel",
            started_at=(
                started_at
            ),
            result=result,
            user_text=None,
            confirmation_action=(
                confirmation_action
            ),
        )

        return result

    def _current_confirmation_action(
        self,
    ) -> ConfirmationAction | None:
        """读取当前待确认动作。"""

        pending = (
            self.state
            .pending_confirmation
        )

        if pending is None:
            return None

        action = pending.action

        if action == "save":
            return "save"

        if action == "update":
            return "update"

        if action == "delete":
            return "delete"

        return None

    def _record_result(
        self,
        *,
        action: TraceAction,
        started_at: float,
        result: AgentRunResult,
        user_text: str | None,
        confirmation_action: (
            ConfirmationAction
            | None
        ),
    ) -> None:
        """记录正常返回的 Agent 结果。"""

        pending_task = (
            result.pending_task
        )

        pending_confirmation = (
            result
            .pending_confirmation
        )

        effective_action = (
            confirmation_action
        )

        if (
            effective_action
            is None
            and pending_confirmation
            is not None
        ):
            pending_action = (
                pending_confirmation
                .action
            )

            if pending_action == "save":
                effective_action = "save"
            elif pending_action == "update":
                effective_action = "update"
            elif pending_action == "delete":
                effective_action = "delete"

        trace = AgentTrace(
            created_at=(
                _utc_now()
                .isoformat()
            ),
            session_hash=_hash_text(
                self.state.session_id
            ),
            user_hash=_hash_text(
                self.state.user_id
            ),
            action=action,
            turn_count=(
                self.state.turn_count
            ),
            input_length=(
                len(user_text)
                if user_text
                is not None
                else None
            ),
            input_sha256=(
                _hash_text(
                    user_text
                )
                if user_text
                is not None
                else None
            ),
            state=(
                result.state.value
            ),
            finish_reason=(
                result
                .finish_reason
                .value
            ),
            model_rounds=(
                result.model_rounds
            ),
            duration_ms=round(
                max(
                    0.0,
                    (
                        perf_counter()
                        - started_at
                    ),
                )
                * 1000,
                3,
            ),
            tool_steps=(
                _safe_tool_steps(
                    result.tool_steps
                )
            ),
            has_pending_task=(
                pending_task
                is not None
            ),
            pending_tool_name=(
                pending_task.tool_name
                if pending_task
                is not None
                else None
            ),
            has_pending_confirmation=(
                pending_confirmation
                is not None
            ),
            confirmation_action=(
                effective_action
            ),
            error_type=None,
        )

        self._safe_append(
            trace
        )

    def _record_exception(
        self,
        *,
        action: TraceAction,
        started_at: float,
        user_text: str | None,
        error: Exception,
        confirmation_action: (
            ConfirmationAction
            | None
        ),
    ) -> None:
        """记录未被 Runner 转换的异常。"""

        pending_task = (
            self.state.pending_task
        )

        pending_confirmation = (
            self.state
            .pending_confirmation
        )

        trace = AgentTrace(
            created_at=(
                _utc_now()
                .isoformat()
            ),
            session_hash=_hash_text(
                self.state.session_id
            ),
            user_hash=_hash_text(
                self.state.user_id
            ),
            action=action,
            turn_count=(
                self.state.turn_count
            ),
            input_length=(
                len(user_text)
                if user_text
                is not None
                else None
            ),
            input_sha256=(
                _hash_text(
                    user_text
                )
                if user_text
                is not None
                else None
            ),
            state=(
                self.state
                .state
                .value
            ),
            finish_reason=(
                "exception"
            ),
            model_rounds=0,
            duration_ms=round(
                max(
                    0.0,
                    (
                        perf_counter()
                        - started_at
                    ),
                )
                * 1000,
                3,
            ),
            tool_steps=(),
            has_pending_task=(
                pending_task
                is not None
            ),
            pending_tool_name=(
                pending_task.tool_name
                if pending_task
                is not None
                else None
            ),
            has_pending_confirmation=(
                pending_confirmation
                is not None
            ),
            confirmation_action=(
                confirmation_action
            ),
            error_type=(
                type(error).__name__
            ),
        )

        self._safe_append(
            trace
        )

    def _safe_append(
        self,
        trace: AgentTrace,
    ) -> None:
        """安全写入，Trace 失败不影响主流程。"""

        self._last_trace_warning = (
            None
        )

        try:
            self._trace_store.append(
                trace
            )
        except Exception as exc:
            self._last_trace_warning = (
                "TRACE_WRITE_FAILED:"
                f"{type(exc).__name__}"
            )
