"""使用 LangGraph 编排现有健康 Agent 工具与状态机。

领域工具、确认令牌、幂等和 JSONL 存储仍由现有模块负责。LangGraph 只负责：

- 模型与工具之间的有限循环；
- 缺参时暂停并等待下一条用户输入；
- 写操作草稿暂停并等待确认或取消；
- 以 thread_id 为会话保存 checkpoint。
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from src.agent.models import (
    AgentFinishReason,
    AgentMessage,
    AgentModel,
    AgentRunResult,
    AgentState,
    AgentToolStep,
    AgentTurnOutcome,
    ModelToolCall,
    PendingConfirmation,
    PendingTask,
    SessionState,
)
from src.agent.runner import (
    SYSTEM_PROMPT,
    TOOL_REQUIRED_RETRY_PROMPT,
    _preview_answer,
    _redact_result,
    _requires_health_tool,
    _tool_request_message,
    _tool_result_message,
)
from src.agent.tool_router import HealthToolRouter


GraphRoute = Literal[
    "model",
    "text",
    "tool",
    "parallel_error",
    "clarification",
    "confirmation",
    "execute_confirmation",
    "end",
]


class HealthAgentGraphState(TypedDict, total=False):
    """可由 LangGraph checkpoint 保存的纯数据状态。"""

    session_id: str
    user_id: str
    timezone_name: str
    messages: list[dict[str, Any]]
    turn_count: int

    agent_state: str
    finish_reason: str | None
    answer: str
    user_text: str
    model_rounds: int
    tool_steps: list[dict[str, Any]]

    pending_task: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    model_reply: dict[str, Any] | None
    next_route: GraphRoute


def _messages_from_state(
    state: HealthAgentGraphState,
) -> list[AgentMessage]:
    return [
        AgentMessage.model_validate(item)
        for item in state.get("messages", [])
    ]


def _tool_steps_from_state(
    state: HealthAgentGraphState,
) -> list[AgentToolStep]:
    return [
        AgentToolStep.model_validate(item)
        for item in state.get("tool_steps", [])
    ]


def _error_message(
    result: dict[str, Any],
    fallback: str,
) -> str:
    error = result.get("error")
    if isinstance(error, dict):
        return str(error.get("message", fallback))
    return fallback


def _failed_update(
    state: HealthAgentGraphState,
    *,
    answer: str,
    finish_reason: AgentFinishReason,
    tool_steps: list[AgentToolStep] | None = None,
) -> HealthAgentGraphState:
    messages = _messages_from_state(state)
    messages.append(
        AgentMessage(role="assistant", content=answer)
    )
    effective_steps = (
        tool_steps
        if tool_steps is not None
        else _tool_steps_from_state(state)
    )
    return {
        "messages": [
            message.model_dump(mode="json")
            for message in messages
        ],
        "answer": answer,
        "finish_reason": finish_reason.value,
        "agent_state": AgentState.FAILED.value,
        "tool_steps": [
            step.model_dump(mode="json")
            for step in effective_steps
        ],
        "pending_task": None,
        "pending_confirmation": None,
        "model_reply": None,
        "next_route": "end",
    }


def _cancelled_update(
    state: HealthAgentGraphState,
) -> HealthAgentGraphState:
    answer = "已取消当前任务，没有写入或修改任何健康数据。"
    messages = _messages_from_state(state)
    messages.append(
        AgentMessage(role="assistant", content=answer)
    )
    return {
        "messages": [
            message.model_dump(mode="json")
            for message in messages
        ],
        "answer": answer,
        "finish_reason": AgentFinishReason.CANCELLED.value,
        "agent_state": AgentState.CANCELLED.value,
        "model_rounds": 0,
        "tool_steps": [],
        "pending_task": None,
        "pending_confirmation": None,
        "model_reply": None,
        "next_route": "end",
    }


def _confirmation_interrupt_payload(
    pending: PendingConfirmation,
) -> dict[str, Any]:
    """只向 UI 暴露草稿预览，不暴露 confirmation_token。"""

    data = pending.draft_data
    payload: dict[str, Any] = {
        "kind": "confirmation",
        "action": pending.action,
        "tool_name": pending.tool_name,
    }

    if pending.action == "save":
        payload["preview"] = data.get("preview", {})
    elif pending.action == "update":
        payload["current_event"] = data.get("current_event", {})
        payload["proposed_event"] = data.get("proposed_event", {})
    elif pending.action == "delete":
        payload["target_event"] = data.get("target_event", {})
    else:
        payload["preview"] = data.get("preview", {})

    return payload


class LangGraphAgentRunner:
    """与原 AgentRunner 保持相同外部协议的 StateGraph 实现。"""

    def __init__(
        self,
        *,
        model: AgentModel,
        router: HealthToolRouter,
        max_model_rounds: int = 4,
        checkpointer: Any | None = None,
    ) -> None:
        if max_model_rounds <= 0:
            raise ValueError("max_model_rounds 必须大于 0")

        self._model = model
        self._router = router
        self._max_model_rounds = max_model_rounds
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = self._build_graph()

    @property
    def router(self) -> HealthToolRouter:
        return self._router

    @property
    def graph(self) -> Any:
        """供测试和开发者证据页检查图状态。"""

        return self._graph

    def _build_graph(self) -> Any:
        builder = StateGraph(HealthAgentGraphState)

        builder.add_node("call_model", self._call_model)
        builder.add_node("handle_text", self._handle_text)
        builder.add_node("reject_parallel_tools", self._reject_parallel_tools)
        builder.add_node("dispatch_tool", self._dispatch_tool)
        builder.add_node("await_clarification", self._await_clarification)
        builder.add_node("await_confirmation", self._await_confirmation)
        builder.add_node("execute_confirmation", self._execute_confirmation)

        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            self._route,
            {
                "text": "handle_text",
                "tool": "dispatch_tool",
                "parallel_error": "reject_parallel_tools",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "handle_text",
            self._route,
            {"model": "call_model", "end": END},
        )
        builder.add_edge("reject_parallel_tools", END)
        builder.add_conditional_edges(
            "dispatch_tool",
            self._route,
            {
                "model": "call_model",
                "clarification": "await_clarification",
                "confirmation": "await_confirmation",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "await_clarification",
            self._route,
            {"model": "call_model", "end": END},
        )
        builder.add_conditional_edges(
            "await_confirmation",
            self._route,
            {
                "execute_confirmation": "execute_confirmation",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "execute_confirmation",
            self._route,
            {"confirmation": "await_confirmation", "end": END},
        )

        return builder.compile(checkpointer=self._checkpointer)

    @staticmethod
    def _route(state: HealthAgentGraphState) -> GraphRoute:
        return state.get("next_route", "end")

    def _call_model(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        rounds = int(state.get("model_rounds", 0))
        if rounds >= self._max_model_rounds:
            return _failed_update(
                state,
                answer="Agent 达到最大模型轮数，仍未产生最终回答。",
                finish_reason=AgentFinishReason.LOOP_LIMIT,
            )

        messages = _messages_from_state(state)
        current_context = self._router.minimal_user_context(
            user_id=state["user_id"],
            timezone_name=state["timezone_name"],
        )
        if messages and messages[0].role == "system":
            messages[0] = AgentMessage(
                role="system",
                content=SYSTEM_PROMPT + current_context,
            )
        reply = self._model.complete(
            messages,
            self._router.tool_definitions,
        )

        if not reply.tool_calls:
            route: GraphRoute = "text"
        elif len(reply.tool_calls) == 1:
            route = "tool"
        else:
            route = "parallel_error"

        return {
            "model_rounds": rounds + 1,
            "model_reply": reply.model_dump(mode="json"),
            "next_route": route,
        }

    def _handle_text(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        raw_reply = state.get("model_reply")
        if not isinstance(raw_reply, dict):
            return _failed_update(
                state,
                answer="模型响应格式无效。",
                finish_reason=AgentFinishReason.TOOL_ERROR,
            )

        content = raw_reply.get("content")
        answer = str(content or "").strip()
        tool_steps = _tool_steps_from_state(state)
        requires_tool = (
            state.get("pending_task") is not None
            or (
                not tool_steps
                and _requires_health_tool(state.get("user_text", ""))
            )
        )

        if requires_tool:
            if int(state.get("model_rounds", 0)) < self._max_model_rounds:
                messages = _messages_from_state(state)
                messages.append(
                    AgentMessage(
                        role="system",
                        content=TOOL_REQUIRED_RETRY_PROMPT,
                    )
                )
                return {
                    "messages": [
                        message.model_dump(mode="json")
                        for message in messages
                    ],
                    "model_reply": None,
                    "next_route": "model",
                }

            return _failed_update(
                state,
                answer=(
                    "模型没有按照协议调用健康工具，本轮操作已终止。"
                    "没有写入或修改健康记录。"
                ),
                finish_reason=AgentFinishReason.TOOL_ERROR,
            )

        messages = _messages_from_state(state)
        messages.append(
            AgentMessage(role="assistant", content=answer)
        )
        return {
            "messages": [
                message.model_dump(mode="json")
                for message in messages
            ],
            "answer": answer,
            "finish_reason": AgentFinishReason.COMPLETED.value,
            "agent_state": AgentState.COMPLETED.value,
            "pending_task": None,
            "pending_confirmation": None,
            "model_reply": None,
            "next_route": "end",
        }

    def _reject_parallel_tools(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        return _failed_update(
            state,
            answer="一次模型响应只能提出一个工具调用。",
            finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
        )

    def _dispatch_tool(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        raw_reply = state.get("model_reply")
        if not isinstance(raw_reply, dict):
            return _failed_update(
                state,
                answer="模型工具调用格式无效。",
                finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
            )

        raw_calls = raw_reply.get("tool_calls")
        if not isinstance(raw_calls, list) or len(raw_calls) != 1:
            return _failed_update(
                state,
                answer="一次模型响应只能提出一个工具调用。",
                finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
            )

        tool_call = ModelToolCall.model_validate(raw_calls[0])
        merged_arguments = dict(tool_call.arguments)

        raw_pending_task = state.get("pending_task")
        if isinstance(raw_pending_task, dict):
            pending_task = PendingTask.model_validate(raw_pending_task)
            if pending_task.tool_name == tool_call.name:
                merged_arguments = {
                    **pending_task.arguments,
                    **tool_call.arguments,
                }

        dispatch = self._router.dispatch(
            tool_name=tool_call.name,
            arguments=merged_arguments,
            user_id=state["user_id"],
            timezone_name=state["timezone_name"],
            session_id=state["session_id"],
            call_id=tool_call.call_id,
        )

        if dispatch.status == "needs_clarification":
            assert dispatch.question is not None
            pending = PendingTask(
                tool_name=tool_call.name,
                arguments=merged_arguments,
                missing_parameters=list(dispatch.missing_parameters),
                question=dispatch.question,
            )
            messages = _messages_from_state(state)
            messages.append(
                AgentMessage(role="assistant", content=dispatch.question)
            )
            return {
                "messages": [
                    message.model_dump(mode="json")
                    for message in messages
                ],
                "answer": dispatch.question,
                "finish_reason": AgentFinishReason.NEEDS_CLARIFICATION.value,
                "agent_state": AgentState.AWAITING_CLARIFICATION.value,
                "pending_task": pending.model_dump(mode="json"),
                "pending_confirmation": None,
                "model_reply": None,
                "next_route": "clarification",
            }

        if dispatch.status == "invalid":
            assert dispatch.result is not None
            return _failed_update(
                state,
                answer=_error_message(dispatch.result, "工具参数无效"),
                finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
            )

        assert dispatch.result is not None
        tool_steps = _tool_steps_from_state(state)
        tool_steps.append(
            AgentToolStep(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=merged_arguments,
                result=_redact_result(dispatch.result),
            )
        )

        if not dispatch.result.get("ok"):
            return _failed_update(
                state,
                answer=_error_message(dispatch.result, "工具执行失败"),
                finish_reason=AgentFinishReason.TOOL_ERROR,
                tool_steps=tool_steps,
            )

        result_data = dispatch.result.get("data")
        draft_tools = {
            "prepare_health_event",
            "prepare_event_change",
            "prepare_update_health_event",
            "prepare_delete_health_event",
            "prepare_profile_update",
            "prepare_goal_change",
            "create_reminder_draft",
            "list_or_cancel_reminders",
        }
        if (
            tool_call.name in draft_tools
            and isinstance(result_data, dict)
            and result_data.get("action") in {
                "save", "update", "delete", "profile_update", "goal_change",
                "reminder_create", "reminder_change",
            }
        ):
            pending_confirmation = PendingConfirmation(
                action=str(result_data["action"]),
                tool_name=tool_call.name,
                draft_data=result_data,
            )
            answer = _preview_answer(result_data)
            messages = _messages_from_state(state)
            messages.append(
                AgentMessage(role="assistant", content=answer)
            )
            return {
                "messages": [
                    message.model_dump(mode="json")
                    for message in messages
                ],
                "answer": answer,
                "finish_reason": AgentFinishReason.AWAITING_CONFIRMATION.value,
                "agent_state": AgentState.AWAITING_CONFIRMATION.value,
                "tool_steps": [
                    step.model_dump(mode="json")
                    for step in tool_steps
                ],
                "pending_task": None,
                "pending_confirmation": pending_confirmation.model_dump(
                    mode="json"
                ),
                "model_reply": None,
                "next_route": "confirmation",
            }

        messages = _messages_from_state(state)
        messages.append(
            _tool_request_message(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments=merged_arguments,
            )
        )
        messages.append(
            _tool_result_message(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                result=dispatch.result,
            )
        )
        return {
            "messages": [
                message.model_dump(mode="json")
                for message in messages
            ],
            "tool_steps": [
                step.model_dump(mode="json")
                for step in tool_steps
            ],
            "pending_task": None,
            "model_reply": None,
            "next_route": "model",
        }

    def _await_clarification(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        raw_pending = state.get("pending_task")
        if not isinstance(raw_pending, dict):
            return _failed_update(
                state,
                answer="缺参任务状态不存在。",
                finish_reason=AgentFinishReason.TOOL_ERROR,
            )

        pending = PendingTask.model_validate(raw_pending)
        response = interrupt(
            {
                "kind": "clarification",
                "tool_name": pending.tool_name,
                "missing_parameters": pending.missing_parameters,
                "question": pending.question,
            }
        )

        if not isinstance(response, dict):
            return {
                "next_route": "clarification",
            }

        if response.get("action") == "cancel":
            return _cancelled_update(state)

        text = str(response.get("text", "")).strip()
        if response.get("action") != "clarify" or not text:
            return {
                "answer": pending.question,
                "finish_reason": AgentFinishReason.NEEDS_CLARIFICATION.value,
                "agent_state": AgentState.AWAITING_CLARIFICATION.value,
                "next_route": "clarification",
            }

        messages = _messages_from_state(state)
        messages.append(AgentMessage(role="user", content=text))
        return {
            "messages": [
                message.model_dump(mode="json")
                for message in messages
            ],
            "turn_count": int(state.get("turn_count", 0)) + 1,
            "user_text": text,
            "answer": "",
            "finish_reason": None,
            "agent_state": AgentState.RUNNING.value,
            "model_rounds": 0,
            "tool_steps": [],
            "model_reply": None,
            "next_route": "model",
        }

    def _await_confirmation(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        raw_pending = state.get("pending_confirmation")
        if not isinstance(raw_pending, dict):
            return _failed_update(
                state,
                answer="待确认草稿状态不存在。",
                finish_reason=AgentFinishReason.TOOL_ERROR,
            )

        pending = PendingConfirmation.model_validate(raw_pending)
        response = interrupt(_confirmation_interrupt_payload(pending))

        if not isinstance(response, dict):
            return {"next_route": "confirmation"}

        if response.get("action") == "cancel":
            return _cancelled_update(state)

        if response.get("action") != "confirm":
            return {
                "agent_state": AgentState.AWAITING_CONFIRMATION.value,
                "finish_reason": AgentFinishReason.AWAITING_CONFIRMATION.value,
                "next_route": "confirmation",
            }

        return {
            "answer": "",
            "finish_reason": None,
            "agent_state": AgentState.RUNNING.value,
            "model_rounds": 0,
            "tool_steps": [],
            "next_route": "execute_confirmation",
        }

    def _execute_confirmation(
        self,
        state: HealthAgentGraphState,
    ) -> HealthAgentGraphState:
        raw_pending = state.get("pending_confirmation")
        if not isinstance(raw_pending, dict):
            return _failed_update(
                state,
                answer="待确认草稿状态不存在。",
                finish_reason=AgentFinishReason.TOOL_ERROR,
            )

        pending = PendingConfirmation.model_validate(raw_pending)
        tool_result = self._router.confirm(pending)
        execution_tool_names = {
            "save": "save_health_event",
            "update": "update_health_event",
            "delete": "delete_health_event",
            "profile_update": "prepare_profile_update",
            "goal_change": "prepare_goal_change",
            "reminder_create": "execute_reminder",
            "reminder_change": "list_or_cancel_reminders",
        }
        step = AgentToolStep(
            call_id="user-confirmation",
            tool_name=execution_tool_names[pending.action],
            arguments={"confirmed": True, "action": pending.action},
            result=tool_result,
        )

        if tool_result.get("ok"):
            answer = {
                "save": "健康事件已确认保存。",
                "update": "健康事件已确认修改。",
                "delete": "健康事件已确认删除。",
                "profile_update": "个人档案已确认更新。",
                "goal_change": "健康目标已确认更新，历史版本已保留。",
                "reminder_create": "提醒已确认安排。",
                "reminder_change": "提醒状态已确认更新。",
            }[pending.action]
            messages = _messages_from_state(state)
            messages.append(
                AgentMessage(role="assistant", content=answer)
            )
            return {
                "messages": [
                    message.model_dump(mode="json")
                    for message in messages
                ],
                "answer": answer,
                "finish_reason": AgentFinishReason.COMPLETED.value,
                "agent_state": AgentState.COMPLETED.value,
                "tool_steps": [step.model_dump(mode="json")],
                "pending_task": None,
                "pending_confirmation": None,
                "next_route": "end",
            }

        answer = _error_message(tool_result, "确认执行失败")
        return {
            "answer": answer,
            "finish_reason": AgentFinishReason.TOOL_ERROR.value,
            "agent_state": AgentState.AWAITING_CONFIRMATION.value,
            "tool_steps": [step.model_dump(mode="json")],
            "pending_confirmation": pending.model_dump(mode="json"),
            "next_route": "confirmation",
        }

    @staticmethod
    def _config(session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}

    def create_session_state(
        self,
        *,
        session_id: str,
        user_id: str,
        timezone_name: str = "Asia/Shanghai",
    ) -> SessionState:
        return SessionState(
            session_id=session_id,
            user_id=user_id,
            timezone_name=timezone_name,
            messages=(AgentMessage(role="system", content=SYSTEM_PROMPT),),
        )

    def _current_graph_state(
        self,
        session_id: str,
    ) -> HealthAgentGraphState:
        snapshot = self._graph.get_state(self._config(session_id))
        return dict(snapshot.values)

    @staticmethod
    def _session_from_graph(
        state: HealthAgentGraphState,
    ) -> SessionState:
        raw_pending_task = state.get("pending_task")
        raw_pending_confirmation = state.get("pending_confirmation")
        return SessionState(
            session_id=state["session_id"],
            user_id=state["user_id"],
            timezone_name=state["timezone_name"],
            state=AgentState(state.get("agent_state", AgentState.IDLE.value)),
            messages=tuple(_messages_from_state(state)),
            turn_count=int(state.get("turn_count", 0)),
            pending_task=(
                PendingTask.model_validate(raw_pending_task)
                if isinstance(raw_pending_task, dict)
                else None
            ),
            pending_confirmation=(
                PendingConfirmation.model_validate(raw_pending_confirmation)
                if isinstance(raw_pending_confirmation, dict)
                else None
            ),
        )

    @staticmethod
    def _result_from_graph(
        state: HealthAgentGraphState,
    ) -> AgentRunResult:
        raw_pending_task = state.get("pending_task")
        raw_pending_confirmation = state.get("pending_confirmation")
        raw_finish_reason = state.get("finish_reason")
        if raw_finish_reason is None:
            raw_finish_reason = AgentFinishReason.TOOL_ERROR.value
        return AgentRunResult(
            answer=state.get("answer", ""),
            finish_reason=AgentFinishReason(raw_finish_reason),
            state=AgentState(state.get("agent_state", AgentState.FAILED.value)),
            model_rounds=int(state.get("model_rounds", 0)),
            tool_steps=tuple(_tool_steps_from_state(state)),
            pending_task=(
                PendingTask.model_validate(raw_pending_task)
                if isinstance(raw_pending_task, dict)
                else None
            ),
            pending_confirmation=(
                PendingConfirmation.model_validate(raw_pending_confirmation)
                if isinstance(raw_pending_confirmation, dict)
                else None
            ),
        )

    def _outcome_from_checkpoint(self, session_id: str) -> AgentTurnOutcome:
        state = self._current_graph_state(session_id)
        return AgentTurnOutcome(
            result=self._result_from_graph(state),
            session_state=self._session_from_graph(state),
        )

    def run_turn(
        self,
        *,
        session_state: SessionState,
        user_text: str,
    ) -> AgentTurnOutcome:
        normalized_text = user_text.strip()
        if not normalized_text:
            result = AgentRunResult(
                answer="请输入内容。",
                finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
                state=session_state.state,
                model_rounds=0,
                pending_task=session_state.pending_task,
                pending_confirmation=session_state.pending_confirmation,
            )
            return AgentTurnOutcome(result=result, session_state=session_state)

        if session_state.pending_confirmation is not None:
            result = AgentRunResult(
                answer=(
                    "当前已有待确认操作。请先点击确认或取消，"
                    "不会继续执行新的写操作。"
                ),
                finish_reason=AgentFinishReason.AWAITING_CONFIRMATION,
                state=AgentState.AWAITING_CONFIRMATION,
                model_rounds=0,
                pending_confirmation=session_state.pending_confirmation,
            )
            return AgentTurnOutcome(result=result, session_state=session_state)

        config = self._config(session_state.session_id)
        if session_state.pending_task is not None:
            self._graph.invoke(
                Command(
                    resume={"action": "clarify", "text": normalized_text}
                ),
                config=config,
            )
        else:
            messages = list(session_state.messages)
            messages.append(AgentMessage(role="user", content=normalized_text))
            initial_state: HealthAgentGraphState = {
                "session_id": session_state.session_id,
                "user_id": session_state.user_id,
                "timezone_name": session_state.timezone_name,
                "messages": [
                    message.model_dump(mode="json")
                    for message in messages
                ],
                "turn_count": session_state.turn_count + 1,
                "agent_state": AgentState.RUNNING.value,
                "finish_reason": None,
                "answer": "",
                "user_text": normalized_text,
                "model_rounds": 0,
                "tool_steps": [],
                "pending_task": None,
                "pending_confirmation": None,
                "model_reply": None,
                "next_route": "model",
            }
            self._graph.invoke(initial_state, config=config)

        return self._outcome_from_checkpoint(session_state.session_id)

    def confirm_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        if session_state.pending_confirmation is None:
            result = AgentRunResult(
                answer="当前没有待确认操作。",
                finish_reason=AgentFinishReason.INVALID_ARGUMENTS,
                state=session_state.state,
                model_rounds=0,
            )
            return AgentTurnOutcome(result=result, session_state=session_state)

        self._graph.invoke(
            Command(resume={"action": "confirm"}),
            config=self._config(session_state.session_id),
        )
        return self._outcome_from_checkpoint(session_state.session_id)

    def cancel_pending(
        self,
        session_state: SessionState,
    ) -> AgentTurnOutcome:
        if (
            session_state.pending_task is not None
            or session_state.pending_confirmation is not None
        ):
            self._graph.invoke(
                Command(resume={"action": "cancel"}),
                config=self._config(session_state.session_id),
            )
            return self._outcome_from_checkpoint(session_state.session_id)

        update = _cancelled_update(
            {
                "messages": [
                    message.model_dump(mode="json")
                    for message in session_state.messages
                ]
            }
        )
        new_state = SessionState(
            session_id=session_state.session_id,
            user_id=session_state.user_id,
            timezone_name=session_state.timezone_name,
            state=AgentState.CANCELLED,
            messages=tuple(
                AgentMessage.model_validate(item)
                for item in update["messages"]
            ),
            turn_count=session_state.turn_count,
        )
        result = AgentRunResult(
            answer=update["answer"],
            finish_reason=AgentFinishReason.CANCELLED,
            state=AgentState.CANCELLED,
            model_rounds=0,
        )
        return AgentTurnOutcome(result=result, session_state=new_state)
