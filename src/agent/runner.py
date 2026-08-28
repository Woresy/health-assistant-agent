"""有限轮次健康 Agent Loop 和多轮会话。"""

from __future__ import annotations

import json
from copy import deepcopy

from src.agent.models import (
    AgentFinishReason,
    AgentMessage,
    AgentModel,
    AgentRunResult,
    AgentState,
    AgentToolStep,
    AgentTurnOutcome,
    PendingConfirmation,
    PendingTask,
    SessionState,
)
from src.agent.tool_router import (
    HealthToolRouter,
)


SYSTEM_PROMPT = """
你是个人健康管理助理 Agent。

你只能协助记录饮食、饮水、体重和运动，
查询健康时间线以及生成确定性每日汇总。

规则：
1. 不做诊断，不替代医生，不夸大健康结论。
2. 模型只能提出工具调用，不能声称工具已经成功。
3. 保存、修改和删除必须先生成草稿。
4. 草稿生成后必须等待用户明确确认。
5. 不得调用工具白名单以外的函数。
6. 缺少必填参数时必须追问，不能猜测。
7. 每日汇总必须读取已保存事件。
8. 饮食营养值必须来自检索和确定性计算。
""".strip()


def _redact_result(
    result: dict[str, object],
) -> dict[str, object]:
    """从 ToolStep 展示结果中移除确认令牌。"""

    redacted = deepcopy(
        result
    )

    data = redacted.get("data")

    if isinstance(data, dict):
        if (
            "confirmation_token"
            in data
        ):
            data[
                "confirmation_token"
            ] = "***redacted***"

    return redacted


def _tool_request_message(
    *,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
) -> AgentMessage:
    """保存模型提出的工具调用。"""

    return AgentMessage(
        role="assistant",
        content=json.dumps(
            {
                "tool_call": {
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": (
                        arguments
                    ),
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_call_id=call_id,
        tool_name=tool_name,
    )


def _tool_result_message(
    *,
    call_id: str,
    tool_name: str,
    result: dict[str, object],
) -> AgentMessage:
    """把工具结果交回模型。"""

    return AgentMessage(
        role="tool",
        content=json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
        ),
        tool_call_id=call_id,
        tool_name=tool_name,
    )


def _preview_answer(
    draft_data: dict[str, object],
) -> str:
    """将副作用草稿转成待确认文本。"""

    action = draft_data.get(
        "action"
    )

    if action == "save":
        preview = draft_data.get(
            "preview",
            {},
        )

        return (
            "已生成待保存草稿，"
            "尚未写入健康记录。\n\n"
            + json.dumps(
                preview,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n请核对后明确确认保存。"
        )

    if action == "update":
        current_event = (
            draft_data.get(
                "current_event",
                {},
            )
        )
        proposed_event = (
            draft_data.get(
                "proposed_event",
                {},
            )
        )

        return (
            "已生成更新草稿，"
            "尚未修改健康记录。\n\n"
            "修改前：\n"
            + json.dumps(
                current_event,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n修改后：\n"
            + json.dumps(
                proposed_event,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n请核对后明确确认修改。"
        )

    target_event = draft_data.get(
        "target_event",
        {},
    )

    return (
        "已找到待删除记录，"
        "尚未执行删除。\n\n"
        + json.dumps(
            target_event,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n请核对后明确确认删除。"
    )


class AgentRunner:
    """协调模型、pending_task 和工具 Router。"""

    def __init__(
        self,
        *,
        model: AgentModel,
        router: HealthToolRouter,
        max_model_rounds: int = 4,
    ) -> None:
        if max_model_rounds <= 0:
            raise ValueError(
                "max_model_rounds "
                "必须大于 0"
            )

        self._model = model
        self._router = router
        self._max_model_rounds = (
            max_model_rounds
        )

    @property
    def router(
        self,
    ) -> HealthToolRouter:
        """供确认操作使用同一个 Router。"""

        return self._router

    def create_session_state(
        self,
        *,
        session_id: str,
        user_id: str,
        timezone_name: str = (
            "Asia/Shanghai"
        ),
    ) -> SessionState:
        """创建新会话状态。"""

        return SessionState(
            session_id=session_id,
            user_id=user_id,
            timezone_name=(
                timezone_name
            ),
            messages=(
                AgentMessage(
                    role="system",
                    content=SYSTEM_PROMPT,
                ),
            ),
        )

    def run_turn(
        self,
        *,
        session_state: SessionState,
        user_text: str,
    ) -> AgentTurnOutcome:
        """执行一个用户对话回合。"""

        normalized_text = (
            user_text.strip()
        )

        if not normalized_text:
            result = AgentRunResult(
                answer="请输入内容。",
                finish_reason=(
                    AgentFinishReason
                    .INVALID_ARGUMENTS
                ),
                state=(
                    session_state.state
                ),
                model_rounds=0,
                pending_task=(
                    session_state
                    .pending_task
                ),
                pending_confirmation=(
                    session_state
                    .pending_confirmation
                ),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        if (
            session_state
            .pending_confirmation
            is not None
        ):
            result = AgentRunResult(
                answer=(
                    "当前已有待确认操作。"
                    "请先点击确认或取消，"
                    "不会继续执行新的写操作。"
                ),
                finish_reason=(
                    AgentFinishReason
                    .AWAITING_CONFIRMATION
                ),
                state=(
                    AgentState
                    .AWAITING_CONFIRMATION
                ),
                model_rounds=0,
                pending_confirmation=(
                    session_state
                    .pending_confirmation
                ),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        messages = list(
            session_state.messages
        )

        messages.append(
            AgentMessage(
                role="user",
                content=normalized_text,
            )
        )

        tool_steps: list[
            AgentToolStep
        ] = []

        pending_task = (
            session_state.pending_task
        )

        for model_round in range(
            1,
            self._max_model_rounds
            + 1,
        ):
            reply = self._model.complete(
                messages,
                self._router
                .tool_definitions,
            )

            if not reply.tool_calls:
                assert (
                    reply.content
                    is not None
                )

                answer = (
                    reply.content.strip()
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=answer,
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .COMPLETED
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                    )
                )

                result = AgentRunResult(
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .COMPLETED
                    ),
                    state=(
                        AgentState
                        .COMPLETED
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            if len(
                reply.tool_calls
            ) != 1:
                answer = (
                    "一次模型响应只能"
                    "提出一个工具调用。"
                )

                return self._failed_outcome(
                    session_state=(
                        session_state
                    ),
                    messages=messages,
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .INVALID_ARGUMENTS
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tool_steps,
                )

            tool_call = (
                reply.tool_calls[0]
            )

            merged_arguments = dict(
                tool_call.arguments
            )

            if (
                pending_task
                is not None
                and pending_task
                .tool_name
                == tool_call.name
            ):
                merged_arguments = {
                    **pending_task.arguments,
                    **tool_call.arguments,
                }

            dispatch = (
                self._router.dispatch(
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                    user_id=(
                        session_state
                        .user_id
                    ),
                    timezone_name=(
                        session_state
                        .timezone_name
                    ),
                    session_id=(
                        session_state
                        .session_id
                    ),
                    call_id=(
                        tool_call.call_id
                    ),
                )
            )

            if (
                dispatch.status
                == "needs_clarification"
            ):
                assert (
                    dispatch.question
                    is not None
                )

                pending = PendingTask(
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                    missing_parameters=list(
                        dispatch
                        .missing_parameters
                    ),
                    question=(
                        dispatch.question
                    ),
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=(
                            dispatch.question
                        ),
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .AWAITING_CLARIFICATION
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                        pending_task=pending,
                    )
                )

                result = AgentRunResult(
                    answer=(
                        dispatch.question
                    ),
                    finish_reason=(
                        AgentFinishReason
                        .NEEDS_CLARIFICATION
                    ),
                    state=(
                        AgentState
                        .AWAITING_CLARIFICATION
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                    pending_task=pending,
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            if (
                dispatch.status
                == "invalid"
            ):
                assert (
                    dispatch.result
                    is not None
                )

                error = (
                    dispatch.result.get(
                        "error"
                    )
                )

                if isinstance(
                    error,
                    dict,
                ):
                    answer = str(
                        error.get(
                            "message",
                            "工具参数无效",
                        )
                    )
                else:
                    answer = (
                        "工具参数无效"
                    )

                return self._failed_outcome(
                    session_state=(
                        session_state
                    ),
                    messages=messages,
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .INVALID_ARGUMENTS
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tool_steps,
                )

            assert (
                dispatch.result
                is not None
            )

            tool_step = AgentToolStep(
                call_id=(
                    tool_call.call_id
                ),
                tool_name=(
                    tool_call.name
                ),
                arguments=(
                    merged_arguments
                ),
                result=_redact_result(
                    dispatch.result
                ),
            )

            tool_steps.append(
                tool_step
            )

            if not dispatch.result.get(
                "ok"
            ):
                error = (
                    dispatch.result.get(
                        "error"
                    )
                )

                if isinstance(
                    error,
                    dict,
                ):
                    answer = str(
                        error.get(
                            "message",
                            "工具执行失败",
                        )
                    )
                else:
                    answer = (
                        "工具执行失败"
                    )

                return self._failed_outcome(
                    session_state=(
                        session_state
                    ),
                    messages=messages,
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .TOOL_ERROR
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tool_steps,
                )

            result_data = (
                dispatch.result.get(
                    "data"
                )
            )

            if (
                tool_call.name
                in {
                    "prepare_health_event",
                    "prepare_update_health_event",
                    "prepare_delete_health_event",
                }
                and isinstance(
                    result_data,
                    dict,
                )
            ):
                action = str(
                    result_data[
                        "action"
                    ]
                )

                pending_confirmation = (
                    PendingConfirmation(
                        action=action,
                        tool_name=(
                            tool_call.name
                        ),
                        draft_data=(
                            result_data
                        ),
                    )
                )

                answer = (
                    _preview_answer(
                        result_data
                    )
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=answer,
                    )
                )

                new_state = (
                    SessionState(
                        session_id=(
                            session_state
                            .session_id
                        ),
                        user_id=(
                            session_state
                            .user_id
                        ),
                        timezone_name=(
                            session_state
                            .timezone_name
                        ),
                        state=(
                            AgentState
                            .AWAITING_CONFIRMATION
                        ),
                        messages=tuple(
                            messages
                        ),
                        turn_count=(
                            session_state
                            .turn_count
                            + 1
                        ),
                        pending_confirmation=(
                            pending_confirmation
                        ),
                    )
                )

                result = AgentRunResult(
                    answer=answer,
                    finish_reason=(
                        AgentFinishReason
                        .AWAITING_CONFIRMATION
                    ),
                    state=(
                        AgentState
                        .AWAITING_CONFIRMATION
                    ),
                    model_rounds=(
                        model_round
                    ),
                    tool_steps=tuple(
                        tool_steps
                    ),
                    pending_confirmation=(
                        pending_confirmation
                    ),
                )

                return AgentTurnOutcome(
                    result=result,
                    session_state=(
                        new_state
                    ),
                )

            messages.append(
                _tool_request_message(
                    call_id=(
                        tool_call.call_id
                    ),
                    tool_name=(
                        tool_call.name
                    ),
                    arguments=(
                        merged_arguments
                    ),
                )
            )

            messages.append(
                _tool_result_message(
                    call_id=(
                        tool_call.call_id
                    ),
                    tool_name=(
                        tool_call.name
                    ),
                    result=(
                        dispatch.result
                    ),
                )
            )

            pending_task = None

        return self._failed_outcome(
            session_state=session_state,
            messages=messages,
            answer=(
                "Agent 达到最大模型轮数，"
                "仍未产生最终回答。"
            ),
            finish_reason=(
                AgentFinishReason
                .LOOP_LIMIT
            ),
            model_rounds=(
                self._max_model_rounds
            ),
            tool_steps=tool_steps,
        )

    def _failed_outcome(
        self,
        *,
        session_state: SessionState,
        messages: list[
            AgentMessage
        ],
        answer: str,
        finish_reason: (
            AgentFinishReason
        ),
        model_rounds: int,
        tool_steps: list[
            AgentToolStep
        ],
    ) -> AgentTurnOutcome:
        """统一构造失败结果。"""

        messages.append(
            AgentMessage(
                role="assistant",
                content=answer,
            )
        )

        new_state = SessionState(
            session_id=(
                session_state.session_id
            ),
            user_id=(
                session_state.user_id
            ),
            timezone_name=(
                session_state
                .timezone_name
            ),
            state=AgentState.FAILED,
            messages=tuple(messages),
            turn_count=(
                session_state.turn_count
                + 1
            ),
        )

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                finish_reason
            ),
            state=AgentState.FAILED,
            model_rounds=(
                model_rounds
            ),
            tool_steps=tuple(
                tool_steps
            ),
        )

        return AgentTurnOutcome(
            result=result,
            session_state=new_state,
        )

    def confirm_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        """明确确认当前待执行草稿。"""

        pending = (
            session_state
            .pending_confirmation
        )

        if pending is None:
            result = AgentRunResult(
                answer="当前没有待确认操作。",
                finish_reason=(
                    AgentFinishReason
                    .INVALID_ARGUMENTS
                ),
                state=(
                    session_state.state
                ),
                model_rounds=0,
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    session_state
                ),
            )

        tool_result = (
            self._router.confirm(
                pending
            )
        )

        execution_tool_names = {
            "save": "save_health_event",
            "update": (
                "update_health_event"
            ),
            "delete": (
                "delete_health_event"
            ),
        }

        step = AgentToolStep(
            call_id=(
                "user-confirmation"
            ),
            tool_name=(
                execution_tool_names[
                    pending.action
                ]
            ),
            arguments={
                "confirmed": True,
                "action": (
                    pending.action
                ),
            },
            result=tool_result,
        )

        if tool_result.get("ok"):
            answer = {
                "save": (
                    "健康事件已确认保存。"
                ),
                "update": (
                    "健康事件已确认修改。"
                ),
                "delete": (
                    "健康事件已确认删除。"
                ),
            }[pending.action]

            messages = (
                *session_state.messages,
                AgentMessage(
                    role="assistant",
                    content=answer,
                ),
            )

            new_state = SessionState(
                session_id=(
                    session_state
                    .session_id
                ),
                user_id=(
                    session_state.user_id
                ),
                timezone_name=(
                    session_state
                    .timezone_name
                ),
                state=(
                    AgentState.COMPLETED
                ),
                messages=messages,
                turn_count=(
                    session_state
                    .turn_count
                ),
            )

            result = AgentRunResult(
                answer=answer,
                finish_reason=(
                    AgentFinishReason
                    .COMPLETED
                ),
                state=(
                    AgentState.COMPLETED
                ),
                model_rounds=0,
                tool_steps=(step,),
            )

            return AgentTurnOutcome(
                result=result,
                session_state=(
                    new_state
                ),
            )

        error = tool_result.get(
            "error"
        )

        if isinstance(error, dict):
            answer = str(
                error.get(
                    "message",
                    "确认执行失败",
                )
            )
        else:
            answer = "确认执行失败"

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                AgentFinishReason
                .TOOL_ERROR
            ),
            state=(
                AgentState
                .AWAITING_CONFIRMATION
            ),
            model_rounds=0,
            tool_steps=(step,),
            pending_confirmation=(
                pending
            ),
        )

        return AgentTurnOutcome(
            result=result,
            session_state=(
                session_state
            ),
        )

    def cancel_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        """取消 pending_task 或待确认草稿。"""

        answer = (
            "已取消当前任务，"
            "没有写入或修改健康记录。"
        )

        messages = (
            *session_state.messages,
            AgentMessage(
                role="assistant",
                content=answer,
            ),
        )

        new_state = SessionState(
            session_id=(
                session_state.session_id
            ),
            user_id=(
                session_state.user_id
            ),
            timezone_name=(
                session_state
                .timezone_name
            ),
            state=AgentState.CANCELLED,
            messages=messages,
            turn_count=(
                session_state.turn_count
            ),
        )

        result = AgentRunResult(
            answer=answer,
            finish_reason=(
                AgentFinishReason
                .CANCELLED
            ),
            state=(
                AgentState.CANCELLED
            ),
            model_rounds=0,
        )

        return AgentTurnOutcome(
            result=result,
            session_state=new_state,
        )


class ConversationSession:
    """保存多轮 Agent 会话状态。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        session_id: str,
        user_id: str,
        timezone_name: str = (
            "Asia/Shanghai"
        ),
    ) -> None:
        self._runner = runner
        self._state = (
            runner.create_session_state(
                session_id=session_id,
                user_id=user_id,
                timezone_name=(
                    timezone_name
                ),
            )
        )

    @property
    def state(
        self,
    ) -> SessionState:
        """返回当前会话状态。"""

        return self._state

    def send(
        self,
        user_text: str,
    ) -> AgentRunResult:
        """提交一条用户消息。"""

        outcome = (
            self._runner.run_turn(
                session_state=(
                    self._state
                ),
                user_text=user_text,
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result

    def confirm(
        self,
    ) -> AgentRunResult:
        """执行用户明确确认的草稿。"""

        outcome = (
            self._runner
            .confirm_pending(
                self._state
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result

    def cancel(
        self,
    ) -> AgentRunResult:
        """取消当前短期任务或草稿。"""

        outcome = (
            self._runner
            .cancel_pending(
                self._state
            )
        )

        self._state = (
            outcome.session_state
        )

        return outcome.result