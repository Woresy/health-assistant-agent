"""个人健康管理助理 Gradio 应用。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import partial
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import gradio as gr
from dotenv import load_dotenv

from src.agent.models import (
    AgentRunResult,
    PendingConfirmation,
)
from src.agent.langgraph_runner import (
    LangGraphAgentRunner,
)
from src.agent.openai_model import (
    AgentConfigurationError,
    AgentProviderError,
    create_agent_model_from_environment,
)
from src.agent.runner import (
    AgentRunner,
    ConversationSession,
    format_health_event_summary,
    health_event_label,
)
from src.agent.tool_router import HealthToolRouter
from src.agent.trace import (
    AgentTraceReadError,
    AgentTraceStore,
    DEFAULT_AGENT_TRACE_PATH,
    TracedConversationSession,
)
from src.health.models import (
    ExercisePayload,
    HealthEvent,
    MealPayload,
    WaterPayload,
    WeightPayload,
)
from src.nutrition.calculator import (
    NutritionCalculationError,
    calculate_nutrition,
    parse_grams,
)
from src.nutrition.repository import (
    FoodRepository,
    NutritionDataError,
)
from src.storage.jsonl_store import HealthEventStore
from src.storage.healthos_store import HealthOSStore
from src.storage.conversation_store import (
    ConversationStore,
)
from src.storage.sqlite_store import (
    SQLiteConversationStore,
    SQLiteDatabase,
    SQLiteHealthEventStore,
    SQLiteHealthOSStore,
    migrate_legacy_storage,
)
from src.tools.get_daily_health_summary import (
    get_daily_health_summary,
)
from src.tools.prepare_health_event import (
    prepare_health_event,
)
from src.tools.query_health_events import (
    query_health_events,
)
from src.tools.retrieve_nutrition_candidates import (
    retrieve_nutrition_candidates,
)
from src.tools.save_health_event import (
    save_health_event,
)
from src.tools.healthos import (
    get_daily_summary as get_healthos_daily_summary,
    get_health_goals,
    get_period_summary,
    get_user_profile,
    list_or_cancel_reminders,
)
from src.ui.image_input import validate_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

APP_CSS = (
    PROJECT_ROOT
    / "src"
    / "ui"
    / "theme.css"
).read_text(encoding="utf-8")

# Gradio 会跟随浏览器的 prefers-color-scheme。页面采用固定的浅色产品
# 视觉，因此 dark token 也需要显式映射到同一套浅色调，避免组件局部变黑。
APP_THEME = gr.themes.Base().set(
    body_text_size="15px",
    body_background_fill="#e9efe9",
    body_background_fill_dark="#e9efe9",
    body_text_color="#17382f",
    body_text_color_dark="#17382f",
    body_text_color_subdued="#53675d",
    body_text_color_subdued_dark="#53675d",
    background_fill_primary="#fffdf7",
    background_fill_primary_dark="#fffdf7",
    background_fill_secondary="#f4f7f1",
    background_fill_secondary_dark="#f4f7f1",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_background_fill="#ffffff",
    block_label_background_fill_dark="#ffffff",
    block_label_text_color="#31564b",
    block_label_text_color_dark="#31564b",
    block_title_text_color="#31564b",
    block_title_text_color_dark="#31564b",
    block_label_text_size="13px",
    block_info_text_size="12px",
    panel_background_fill="#f4f7f1",
    panel_background_fill_dark="#f4f7f1",
    border_color_primary="#d6e1d8",
    border_color_primary_dark="#d6e1d8",
    input_background_fill="#f7f9f5",
    input_background_fill_dark="#f7f9f5",
    input_border_color="#d6e1d8",
    input_border_color_dark="#d6e1d8",
    input_placeholder_color="#53675d",
    input_placeholder_color_dark="#53675d",
    input_text_size="14px",
    button_large_text_size="14px",
    button_medium_text_size="13px",
    button_small_text_size="12px",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#ffffff",
    table_odd_background_fill="#f7f9f5",
    table_odd_background_fill_dark="#f7f9f5",
    table_text_color="#17382f",
    table_text_color_dark="#17382f",
    table_border_color="#d6e1d8",
    table_border_color_dark="#d6e1d8",
)

APP_HEAD = """
<style>
  :root { color-scheme: only light; }
  html, body {
    min-height: 100%;
    background: #e9efe9 !important;
    color: #17382f !important;
  }
  @media (prefers-color-scheme: dark) {
    html, body {
      background: #e9efe9 !important;
      color: #17382f !important;
    }
  }
</style>
<script>
  document.addEventListener("DOMContentLoaded", () => {
    const classifyHealthOSNavigation = () => {
      const nav = document.querySelector('#main-tabs .tab-nav') ||
        document.querySelector('#main-tabs > div:first-child');
      if (!nav) return;
      const tabs = Array.from(nav.querySelectorAll('button[role="tab"]'));
      const resolveTab = (panelId, fallbackLabel) => {
        const panel = document.getElementById(panelId);
        const labelledBy = panel?.getAttribute("aria-labelledby");
        return (labelledBy && document.getElementById(labelledBy)) ||
          nav.querySelector(`[aria-controls="${panelId}"]`) ||
          tabs.find((tab) => tab.textContent.trim() === fallbackLabel);
      };
      const today = resolveTab("healthos-today", "今天");
      const chat = resolveTab("healthos-record", "对话");
      const timeline = resolveTab("healthos-timeline", "健康时间线");
      const meal = resolveTab("healthos-meal", "餐食图片");
      const evidence = resolveTab("healthos-evidence", "运行证据");
      const privacy = resolveTab("healthos-privacy", "数据与隐私");
      if (timeline) timeline.dataset.healthosNav = "support";
      if (meal) meal.dataset.healthosNav = "support";
      if (evidence) evidence.dataset.healthosNav = "utility-start";
      if (privacy) privacy.dataset.healthosNav = "utility";
      if (
        chat &&
        today &&
        (
          chat.compareDocumentPosition(today) &
          Node.DOCUMENT_POSITION_PRECEDING
        )
      ) {
        nav.insertBefore(chat, today);
      }
      if (today && !nav.querySelector('[data-healthos-group="daily"]')) {
        const label = document.createElement("span");
        label.dataset.healthosGroup = "daily";
        label.className = "healthos-nav-group";
        label.textContent = "日常工作";
        nav.insertBefore(label, chat || today);
      }
      if (evidence && !nav.querySelector('[data-healthos-group="system"]')) {
        const label = document.createElement("span");
        label.dataset.healthosGroup = "system";
        label.className = "healthos-nav-group";
        label.textContent = "系统控制";
        nav.insertBefore(label, evidence);
      }
    };
    const observer = new MutationObserver(classifyHealthOSNavigation);
    observer.observe(document.body, { childList: true, subtree: true });
    classifyHealthOSNavigation();
  });
</script>
"""

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "health_events.jsonl"
)

CONVERSATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "conversations"
)

HEALTHOS_STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "healthos_state.json"
)

SQLITE_PATH = (
    PROJECT_ROOT
    / os.getenv("SQLITE_DATABASE_PATH", "data/healthos.db").strip()
)

STORAGE_BACKEND = os.getenv(
    "STORAGE_BACKEND",
    "sqlite",
).strip().lower()

LOCAL_USER_ID = "local-demo-user"

APP_TIMEZONE = os.getenv(
    "APP_TIMEZONE",
    "Asia/Shanghai",
).strip()

AGENT_ORCHESTRATOR = os.getenv(
    "AGENT_ORCHESTRATOR",
    "langgraph",
).strip().lower()


repository = FoodRepository()

if STORAGE_BACKEND == "sqlite":
    sqlite_database = SQLiteDatabase(SQLITE_PATH)
    sqlite_counts = sqlite_database.integrity_check()["counts"]
    if not any(sqlite_counts.values()):
        migrate_legacy_storage(
            sqlite_database,
            events_path=EVENTS_PATH,
            healthos_path=HEALTHOS_STATE_PATH,
            conversations_path=CONVERSATIONS_PATH,
        )
    event_store = SQLiteHealthEventStore(sqlite_database)
    healthos_store = SQLiteHealthOSStore(sqlite_database)
    conversation_store = SQLiteConversationStore(sqlite_database)
elif STORAGE_BACKEND == "json":
    sqlite_database = None
    event_store = HealthEventStore(EVENTS_PATH)
    healthos_store = HealthOSStore(HEALTHOS_STATE_PATH)
    conversation_store = ConversationStore(CONVERSATIONS_PATH)
else:
    raise RuntimeError("STORAGE_BACKEND 只能是 sqlite 或 json")

tool_router = HealthToolRouter(
    event_store,
    healthos_store=healthos_store,
    nutrition_repository=repository,
)

agent_trace_store = AgentTraceStore(
    DEFAULT_AGENT_TRACE_PATH
)

try:
    if AGENT_ORCHESTRATOR not in {
        "legacy",
        "langgraph",
    }:
        raise AgentConfigurationError(
            "AGENT_ORCHESTRATOR 只能是 "
            "legacy 或 langgraph"
        )

    (
        agent_model,
        AGENT_PROVIDER_STATUS,
    ) = create_agent_model_from_environment()

    AGENT_PROVIDER_STATUS = (
        f"{AGENT_PROVIDER_STATUS.rstrip('。；;')}；"
        f"编排器：{AGENT_ORCHESTRATOR}"
    )
except AgentConfigurationError as exc:
    agent_model = None
    AGENT_PROVIDER_STATUS = (
        "Agent 配置错误："
        f"{exc}"
    )


_AGENT_SESSIONS: dict[
    str,
    TracedConversationSession,
] = {}

_AGENT_SESSIONS_LOCK = RLock()

_REQUEST_SESSION_ALIASES: dict[
    str,
    str,
] = {}


def _error_text(
    error_code: str,
    message: str,
) -> str:
    """生成统一的 UI 错误文本。"""

    return (
        f"错误 [{error_code}]："
        f"{message}"
    )


def _timezone() -> ZoneInfo:
    """取得应用时区。"""

    try:
        return ZoneInfo(
            APP_TIMEZONE
        )
    except (
        ZoneInfoNotFoundError,
        ValueError,
    ) as exc:
        raise ValueError(
            "无法加载 APP_TIMEZONE："
            f"{APP_TIMEZONE}"
        ) from exc


def _today_string() -> str:
    """返回应用时区中的今天。"""

    return (
        datetime.now(
            _timezone()
        )
        .date()
        .isoformat()
    )


def _request_session_key(
    request: gr.Request,
) -> str:
    """获取当前页面实例的 Gradio 会话标识。"""

    session_hash = getattr(
        request,
        "session_hash",
        None,
    )

    if (
        isinstance(session_hash, str)
        and session_hash.strip()
    ):
        return session_hash.strip()

    return "local-fallback-session"


def _conversation_id(
    stored_value: Any,
) -> str:
    """校验或生成浏览器长期保存的匿名会话标识。"""

    try:
        parsed = UUID(
            str(stored_value).strip()
        )
    except (ValueError, TypeError):
        parsed = uuid4()

    return parsed.hex


def _bind_conversation(
    stored_value: Any,
    request: gr.Request,
) -> tuple[str, str]:
    """把页面临时 session_hash 绑定到稳定会话。"""

    browser_id = _conversation_id(
        stored_value
    )
    session_key = (
        f"conversation-{browser_id}"
    )
    request_key = _request_session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        _REQUEST_SESSION_ALIASES[
            request_key
        ] = session_key

    return browser_id, session_key


def _session_key(
    request: gr.Request,
) -> str:
    """返回页面已经绑定的稳定会话标识。"""

    request_key = _request_session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        return _REQUEST_SESSION_ALIASES.get(
            request_key,
            request_key,
        )


def _get_agent_session(
    request: gr.Request,
) -> TracedConversationSession | None:
    """按浏览器会话获取带 Trace 的 Agent Session。"""

    if agent_model is None:
        return None

    key = _session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        existing = _AGENT_SESSIONS.get(
            key
        )

        if existing is not None:
            return existing

        restored_state = None
        if key.startswith(
            "conversation-"
        ):
            restored_state = (
                conversation_store.load(
                    key
                )
            )

        runner_class = (
            AgentRunner
            if (
                restored_state is not None
                and (
                    restored_state.pending_task
                    is not None
                    or restored_state.pending_confirmation
                    is not None
                )
            )
            else (
                LangGraphAgentRunner
                if AGENT_ORCHESTRATOR
                == "langgraph"
                else AgentRunner
            )
        )

        runner = runner_class(
            model=agent_model,
            router=tool_router,
            max_model_rounds=4,
        )

        session = TracedConversationSession(
            runner=runner,
            session_id=key,
            user_id=LOCAL_USER_ID,
            timezone_name=APP_TIMEZONE,
            trace_store=agent_trace_store,
            session_state=restored_state,
        )

        _AGENT_SESSIONS[key] = session

        return session


def _persist_agent_session(
    session: TracedConversationSession,
) -> bool:
    """保存会话，失败时不影响本轮健康记录操作。"""

    if not session.state.session_id.startswith(
        "conversation-"
    ):
        return False

    try:
        conversation_store.save(
            session.state
        )
    except (OSError, ValueError):
        return False

    return True


def cleanup_agent_session(
    request: gr.Request,
) -> None:
    """页面关闭时释放内存，会话继续保存在本地文件。"""

    request_key = _request_session_key(
        request
    )

    with _AGENT_SESSIONS_LOCK:
        stable_key = (
            _REQUEST_SESSION_ALIASES.pop(
                request_key,
                None,
            )
        )
        if stable_key is not None:
            _AGENT_SESSIONS.pop(
                stable_key,
                None,
            )


def _result_state_json(
    *,
    session: ConversationSession,
    result: AgentRunResult,
) -> dict[str, Any]:
    """生成不包含健康参数和确认令牌的状态证据。"""

    pending_task = (
        session.state.pending_task
    )

    pending_confirmation = (
        session.state.pending_confirmation
    )

    trace_warning = getattr(
        session,
        "last_trace_warning",
        None,
    )

    return {
        "state": (
            session.state.state.value
        ),
        "turn_count": (
            session.state.turn_count
        ),
        "finish_reason": (
            result.finish_reason.value
        ),
        "model_rounds": (
            result.model_rounds
        ),
        "pending_task": (
            {
                "tool_name": (
                    pending_task.tool_name
                ),
                "missing_parameters": (
                    pending_task.missing_parameters
                ),
                "question": (
                    pending_task.question
                ),
            }
            if pending_task is not None
            else None
        ),
        "pending_confirmation": (
            {
                "action": (
                    pending_confirmation.action
                ),
                "tool_name": (
                    pending_confirmation.tool_name
                ),
            }
            if pending_confirmation is not None
            else None
        ),
        "trace": {
            "enabled": True,
            "write_warning": (
                trace_warning
            ),
        },
    }


def _tool_steps_json(
    result: AgentRunResult,
) -> list[dict[str, Any]]:
    """将工具步骤转换为脱敏的开发者证据。"""

    safe_steps: list[
        dict[str, Any]
    ] = []

    for step in result.tool_steps:
        raw_ok = step.result.get(
            "ok"
        )

        ok = (
            raw_ok
            if isinstance(raw_ok, bool)
            else None
        )

        error_code: str | None = None

        raw_error = step.result.get(
            "error"
        )

        if isinstance(raw_error, dict):
            raw_code = raw_error.get("error_code") or raw_error.get("code")

            if isinstance(raw_code, str):
                error_code = raw_code

        source = "本地 SQLite 业务数据"
        if step.tool_name == "retrieve_nutrition_candidates":
            source = "本地食物库与 Hybrid RAG 索引"
        elif step.tool_name == "calculate_nutrition":
            source = "用户选中的结构化食物数据行"
        elif step.tool_name == "retrieve_health_knowledge":
            source = "受控健康知识库及其引用来源"

        safe_steps.append(
            {
                "tool": (
                    step.tool_name
                ),
                "status": "成功" if ok is True else "失败" if ok is False else "已执行",
                "source": source,
                "failure": error_code,
            }
        )

    return safe_steps


def _pending_confirmation_content(
    pending: PendingConfirmation,
) -> str:
    """生成对话区内的待确认卡片。"""

    data = pending.draft_data
    action = pending.action

    if action == "save":
        record = data.get("preview", {})
        label = health_event_label(record)
        title = f"确认{label}记录"
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>{escape(format_health_event_summary(record))}</strong>"
            "<span>确认后将写入你的今日健康记录。</span>"
            "</div>"
        )
        consequence = (
            "内容确认后只会保存一次。"
            "取消则不会产生新记录。"
        )
    elif action == "update":
        current = data.get("current_event", {})
        proposed = data.get("proposed_event", {})
        label = health_event_label(proposed)
        title = f"确认修改{label}记录"
        summary_html = (
            '<div class="confirmation-compare">'
            '<div><span>修改前</span>'
            f"<strong>{escape(format_health_event_summary(current))}</strong></div>"
            '<div><span>修改后</span>'
            f"<strong>{escape(format_health_event_summary(proposed))}</strong></div>"
            "</div>"
        )
        consequence = (
            "确认后将替换原记录。"
            "取消则保留原内容。"
        )
    elif action == "delete":
        record = data.get("target_event", {})
        label = health_event_label(record)
        title = f"确认删除{label}记录"
        summary_html = (
            '<div class="confirmation-summary confirmation-summary-danger">'
            f"<strong>{escape(format_health_event_summary(record))}</strong>"
            "<span>这条记录将从健康时间线中删除。</span>"
            "</div>"
        )
        consequence = (
            "删除后无法在页面内撤销。"
            "如果不确定，请选择取消。"
        )
    elif action == "profile_update":
        preview = data.get("preview", {})
        after = preview.get("after", {}) if isinstance(preview, dict) else {}
        style = {
            "gentle": "温和陪伴",
            "rational": "理性复盘",
            "concise": "简洁提醒",
            "goal_focused": "目标督促",
        }.get(str(after.get("coach_style", "")), "保持当前风格")
        label = "档案"
        title = "确认更新个人档案"
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>教练风格：{escape(style)}</strong>"
            f"<span>时区：{escape(str(after.get('timezone_name', APP_TIMEZONE)))}</span>"
            "</div>"
        )
        consequence = "确认后更新已明确选择的偏好；健康事实和安全规则不会改变。"
    elif action == "goal_change":
        preview = data.get("preview", {})
        after = preview.get("after", {}) if isinstance(preview, dict) else {}
        label = "目标"
        title = "确认健康目标变更"
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>{escape(str(after.get('title', '健康目标')))}</strong>"
            f"<span>{escape(str(after.get('target_value', '')))} "
            f"{escape(str(after.get('unit', '')))} · {escape(str(after.get('period', '')))}</span>"
            "</div>"
        )
        consequence = "确认后生成一个新版本，历史目标不会被覆盖。"
    elif action == "reminder_create":
        preview = data.get("preview", {})
        label = "提醒"
        title = "确认安排提醒"
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>{escape(str(preview.get('content', '健康提醒')))}</strong>"
            f"<span>{escape(str(preview.get('scheduled_for', '')))} · "
            f"{escape(str(preview.get('timezone_name', APP_TIMEZONE)))}</span>"
            "</div>"
        )
        consequence = "确认后在本地提醒中心创建一次；重复确认不会重复创建。"
    else:
        preview = data.get("preview", {})
        operation = {
            "cancel": "取消",
            "snooze": "延后",
            "pause": "暂停",
            "resume": "恢复",
        }.get(str(preview.get("operation", "")), "修改")
        label = "提醒"
        title = f"确认{operation}提醒"
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>{escape(operation)}当前提醒</strong>"
            "<span>确认前提醒状态保持不变。</span>"
            "</div>"
        )
        consequence = "确认后记录状态变化及原因，之后仍可在提醒中心回查。"

    marker = escape(label[:1] or "记")

    return (
        '<section class="confirmation-card" role="status" '
        'aria-live="polite">'
        '<div class="confirmation-title">'
        f'<i aria-hidden="true">{marker}</i>'
        "<div>"
        f"<strong>{escape(title)}</strong>"
        "<span>请检查下面的信息</span>"
        "</div>"
        '<b class="confirmation-status">等待确认</b>'
        "</div>"
        f"{summary_html}"
        f'<p class="confirmation-consequence">{escape(consequence)}</p>'
        "</section>"
    )


def _confirmation_updates(
    pending: PendingConfirmation | None,
) -> tuple[Any, Any, Any]:
    """同步内联确认卡片和两个操作按钮的可见状态。"""

    if pending is None:
        return (
            gr.Markdown(
                value="",
                visible=False,
                sanitize_html=False,
                container=False,
            ),
            gr.Button(visible=False),
            gr.Button(visible=False),
        )

    confirm_label = {
        "save": "确认并保存",
        "update": "确认修改",
        "delete": "确认删除",
        "profile_update": "确认更新档案",
        "goal_change": "确认目标变更",
        "reminder_create": "确认安排提醒",
        "reminder_change": "确认提醒变更",
    }[pending.action]

    return (
        gr.Markdown(
            value=(
                _pending_confirmation_content(
                    pending
                )
            ),
            visible=True,
            sanitize_html=False,
            container=False,
        ),
        gr.Button(
            value=confirm_label,
            visible=True,
            variant=(
                "stop"
                if pending.action == "delete"
                else "primary"
            ),
            size="md",
        ),
        gr.Button(
            value="取消",
            visible=True,
            variant="secondary",
            size="md",
        ),
    )


def _agent_status_text(
    result: AgentRunResult,
) -> str:
    """将内部 Agent 状态转成用户可理解的进度提示。"""

    return {
        "idle": "可以开始记录。",
        "running": "正在整理你的信息。",
        "awaiting_clarification": (
            "还需要一点信息，请继续回复。"
        ),
        "awaiting_confirmation": (
            "记录已整理好，等待你确认。"
        ),
        "completed": "本轮操作已完成。",
        "failed": (
            "本轮没有完成，请查看对话中的提示。"
        ),
        "cancelled": (
            "已取消，没有修改健康记录。"
        ),
    }.get(
        result.state.value,
        "请查看对话中的最新提示。",
    )


def _welcome_chat_history() -> list[dict[str, str]]:
    """返回新会话的欢迎消息。"""

    return [
        {
            "role": "assistant",
            "content": (
                "你好，我在这里。直接告诉我刚刚发生的事，"
                "或者问我今天吃了什么、记录了什么。"
            ),
        }
    ]


def _restored_chat_history(
    session: TracedConversationSession,
) -> list[dict[str, str]]:
    """从 Agent 状态恢复用户可见的对话消息。"""

    history: list[dict[str, str]] = []

    for message in session.state.messages:
        if message.role not in {
            "user",
            "assistant",
        }:
            continue

        content = message.content
        marker = "用户原始请求："
        if (
            message.role == "user"
            and content.startswith(
                "[内部选中记录]"
            )
            and marker in content
        ):
            content = content.split(
                marker,
                maxsplit=1,
            )[1]

        history.append(
            {
                "role": message.role,
                "content": content,
            }
        )

    return history or _welcome_chat_history()


def restore_agent_conversation(
    stored_value: Any,
    request: gr.Request,
) -> tuple[Any, ...]:
    """页面加载时恢复稳定会话、消息和待确认操作。"""

    browser_id, _ = _bind_conversation(
        stored_value,
        request,
    )
    session = _get_agent_session(request)

    if session is None:
        return (
            browser_id,
            _welcome_chat_history(),
            AGENT_PROVIDER_STATUS,
            *_confirmation_updates(None),
        )

    history = _restored_chat_history(
        session
    )
    visible_message_count = len(history)
    has_existing_history = (
        session.state.turn_count > 0
        or visible_message_count > 1
    )
    status = (
        "已恢复上次对话，"
        f"共 {visible_message_count} 条消息。"
        if has_existing_history
        else "对话已准备好，可以开始记录。"
    )
    _persist_agent_session(session)

    return (
        browser_id,
        history,
        status,
        *_confirmation_updates(
            session.state.pending_confirmation
        ),
    )


def begin_agent_activity(
    user_text: str,
    selected_record: (
        dict[str, Any]
        | None
    ),
) -> Any:
    """用可验证的执行步骤替代默认计时提示。"""

    normalized_text = (
        user_text.strip()
        if isinstance(user_text, str)
        else ""
    )
    has_selected_record = bool(
        isinstance(selected_record, dict)
        and selected_record.get("event_id")
    )

    if has_selected_record:
        steps = (
            "已关联你选择的健康记录",
            "正在理解修改内容",
            "接下来会生成待确认草稿",
        )
    elif any(term in normalized_text for term in ("目标", "档案", "教练风格", "偏好")):
        steps = (
            "已识别个人设置或目标请求",
            "正在读取当前版本与历史",
            "接下来生成可确认的变更草稿",
        )
    elif "提醒" in normalized_text:
        steps = (
            "已识别提醒行动",
            "正在核对时间、时区和当前状态",
            "接下来生成可确认的提醒草稿",
        )
    elif any(term in normalized_text for term in ("建议", "健康知识", "怎么吃", "怎么运动")):
        steps = (
            "已识别一般健康知识问题",
            "正在检查安全边界与可信来源",
            "接下来整理带引用的回答",
        )
    elif any(
        term in normalized_text
        for term in (
            "查询",
            "查看",
            "多少",
            "汇总",
            "今天有哪些",
        )
    ):
        steps = (
            "已识别查询范围",
            "正在读取已确认记录",
            "接下来整理成自然语言结果",
        )
    elif any(
        term in normalized_text
        for term in (
            "记录",
            "喝了",
            "吃了",
            "跑步",
            "体重",
        )
    ):
        steps = (
            "已识别健康记录请求",
            "正在检查必要信息",
            "接下来生成待确认草稿",
        )
    else:
        steps = (
            "已收到你的问题",
            "正在选择合适的健康工具",
            "接下来整理清晰的回答",
        )

    items = "".join(
        '<li class="done">'
        if index == 0
        else (
            '<li class="active">'
            if index == 1
            else "<li>"
        )
        + escape(step)
        + "</li>"
        for index, step in enumerate(steps)
    )

    return gr.Markdown(
        value=(
            '<section class="agent-activity" '
            'role="status" aria-live="polite">'
            "<strong>小满正在处理</strong>"
            f"<ol>{items}</ol>"
            "</section>"
        ),
        visible=True,
        sanitize_html=False,
        container=False,
    )


def finish_agent_activity(
    status_text: str,
) -> Any:
    """用简洁结果收起本轮执行过程。"""

    normalized_status = (
        status_text.strip()
        if isinstance(status_text, str)
        else "本轮处理已结束。"
    )

    return gr.Markdown(
        value=(
            '<section class="agent-activity complete" '
            'role="status" aria-live="polite">'
            "<strong>本轮处理完成</strong>"
            f"<p>{escape(normalized_status)}</p>"
            "</section>"
        ),
        visible=True,
        sanitize_html=False,
        container=False,
    )


def refresh_agent_traces(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """读取最近的脱敏 Agent Trace。"""

    try:
        traces = (
            agent_trace_store
            .read_recent(
                limit=limit
            )
        )
    except AgentTraceReadError as exc:
        return [
            {
                "ok": False,
                "error": {
                    "code": (
                        exc.error_code
                    ),
                    "message": str(
                        exc
                    ),
                },
            }
        ]
    except ValueError as exc:
        return [
            {
                "ok": False,
                "error": {
                    "code": (
                        "INVALID_TRACE_LIMIT"
                    ),
                    "message": str(
                        exc
                    ),
                },
            }
        ]

    return [
        trace.model_dump(
            mode="json"
        )
        for trace in traces
    ]


def _event_detail(
    event: HealthEvent,
) -> str:
    """将四类 payload 转为时间线摘要。"""

    payload = event.payload

    if isinstance(
        payload,
        MealPayload,
    ):
        return (
            f"{payload.food.name}，"
            f"{payload.portion.grams:g} g，"
            "约 "
            f"{payload.nutrition.calories_kcal:g} kcal"
        )

    if isinstance(
        payload,
        WaterPayload,
    ):
        return (
            f"{payload.beverage} "
            f"{payload.amount_ml:g} ml"
            + (
                f"，{payload.note}"
                if payload.note
                else ""
            )
        )

    if isinstance(
        payload,
        WeightPayload,
    ):
        return (
            f"{payload.weight_kg:g} kg"
            + (
                f"，{payload.note}"
                if payload.note
                else ""
            )
        )

    if isinstance(
        payload,
        ExercisePayload,
    ):
        details = (
            f"{payload.activity_type}，"
            f"{payload.duration_minutes:g} 分钟"
        )

        if payload.distance_km is not None:
            details += (
                f"，{payload.distance_km:g} km"
            )

        if payload.intensity is not None:
            details += (
                "，"
                + {
                    "low": "低强度",
                    "medium": "中等强度",
                    "high": "高强度",
                }[payload.intensity.value]
            )

        if payload.note:
            details += (
                f"，{payload.note}"
            )

        return details

    return "未知事件"


def _event_type_label(
    event: HealthEvent,
) -> str:
    """把内部事件类型转换为用户熟悉的名称。"""

    return {
        "meal": "饮食",
        "water": "饮水",
        "weight": "体重",
        "exercise": "运动",
    }[event.event_type.value]


def _input_source_label(
    event: HealthEvent,
) -> str:
    """把内部录入来源转换为用户可读文本。"""

    return {
        "chat": "对话记录",
        "image": "图片记录",
        "model": "智能整理",
    }[event.input_source.value]


def _record_status(
    event: HealthEvent,
    local_timezone: ZoneInfo,
) -> str:
    """区分首次确认和后续修改。"""

    if event.updated_at == event.created_at:
        return "已确认"

    updated_text = (
        event.updated_at
        .astimezone(local_timezone)
        .strftime("%m月%d日 %H:%M")
    )
    return f"已修改，{updated_text}"


def _timeline_rows(
    events: list[
        dict[str, Any]
    ],
) -> list[list[Any]]:
    """将工具返回事件转换为表格行。"""

    local_timezone = _timezone()

    rows: list[
        list[Any]
    ] = []

    for raw_event in events:
        event = (
            HealthEvent
            .model_validate(
                raw_event
            )
        )

        rows.append(
            [
                (
                    event.occurred_at
                    .astimezone(
                        local_timezone
                    )
                    .strftime(
                        "%m月%d日 "
                        "%H:%M"
                    )
                ),
                _event_type_label(event),
                _event_detail(
                    event
                ),
                _input_source_label(event),
                _record_status(
                    event,
                    local_timezone,
                ),
            ]
        )

    return rows


def _summary_markdown(
    summary: dict[str, Any],
    goal_gaps: list[dict[str, Any]] | None = None,
) -> str:
    """将每日汇总转换为可视化指标卡。"""

    meal = summary["meal"]
    water = summary["water"]
    weight = summary["weight"]
    exercise = summary["exercise"]

    event_count = int(
        summary["event_count"]
    )

    event_status = (
        f"{event_count} 条已确认记录"
        if event_count
        else "今天还没有记录"
    )

    latest_weight_value = (
        f"{weight['latest_weight_kg']:g}"
        if weight["latest_weight_kg"]
        is not None
        else "暂无"
    )

    latest_weight_unit = (
        "kg"
        if weight["latest_weight_kg"]
        is not None
        else "暂无记录"
    )

    calories = float(
        meal["calories_kcal"]
    )

    water_ml = float(
        water["total_ml"]
    )

    exercise_minutes = float(
        exercise[
            "total_duration_minutes"
        ]
    )

    goal_by_type: dict[str, dict[str, Any]] = {}
    for item in (goal_gaps or []):
        unit = str(item.get("unit", "")).lower()
        if unit in {"ml", "毫升"}:
            goal_by_type["water"] = item
        elif unit in {"分钟", "min", "minutes"}:
            goal_by_type["exercise"] = item
        elif unit in {"kcal", "千卡"}:
            goal_by_type["nutrition"] = item
    calorie_target = float(goal_by_type.get("nutrition", {}).get("target_value", 2000))
    water_target = float(goal_by_type.get("water", {}).get("target_value", 1800))
    exercise_target = float(goal_by_type.get("exercise", {}).get("target_value", 60))

    calorie_progress = min(
        calories / calorie_target * 100,
        100,
    ) if calorie_target > 0 else 0

    water_progress = min(
        water_ml / water_target * 100,
        100,
    ) if water_target > 0 else 0

    exercise_progress = min(
        exercise_minutes / exercise_target * 100,
        100,
    ) if exercise_target > 0 else 0

    weight_progress = (
        72
        if weight["latest_weight_kg"]
        is not None
        else 0
    )

    if int(summary.get("event_count", 0)) == 0:
        decision_title = "先留下今天的第一条记录"
        decision_detail = "可以从刚喝的水、最近一餐或一次运动开始。"
    elif water_target > 0 and water_ml < water_target:
        remaining_water = max(water_target - water_ml, 0)
        decision_title = f"饮水目标还差 {remaining_water:.0f} ml"
        decision_detail = "这是根据今天已确认的饮水记录和当前目标计算的差值。"
    elif exercise_target > 0 and exercise_minutes < exercise_target:
        remaining_minutes = max(exercise_target - exercise_minutes, 0)
        decision_title = f"运动目标还差 {remaining_minutes:.0f} 分钟"
        decision_detail = "如果今天不便运动，也可以只记录真实情况，不需要补齐数字。"
    else:
        decision_title = "今天的主要目标已有记录"
        decision_detail = "继续按真实情况记录即可，不需要为了完成指标而补数据。"

    return (
        '<section class="summary-board">'
        '<div class="summary-topline">'
        '<b>今天的状态</b>'
        f'<span>{event_status}</span>'
        '</div>'
        '<div class="summary-decision">'
        '<span>当前最值得关注</span>'
        f'<strong>{decision_title}</strong>'
        f'<small>{decision_detail}</small>'
        '</div>'
        '<div class="metric-grid">'
        '<article class="health-metric" '
        'style="--metric-color:#2f765e;'
        f'--metric-progress:{calorie_progress:.1f}%">'
        '<label>饮食热量 · 估算</label>'
        f'<strong>{calories:.0f}</strong><small>kcal</small>'
        '<div class="metric-track"><i></i></div>'
        f'<small>{meal["count"]} 顿已确认餐食</small>'
        '</article>'
        '<article class="health-metric" '
        'style="--metric-color:#2f765e;'
        f'--metric-progress:{water_progress:.1f}%">'
        '<label>今日饮水</label>'
        f'<strong>{water_ml:.0f}</strong><small>ml</small>'
        '<div class="metric-track"><i></i></div>'
        f'<small>目标 {water_target:g} ml</small>'
        '</article>'
        '<article class="health-metric" '
        'style="--metric-color:#2f765e;'
        f'--metric-progress:{exercise_progress:.1f}%">'
        '<label>运动时长</label>'
        f'<strong>{exercise_minutes:.0f}</strong><small>分钟</small>'
        '<div class="metric-track"><i></i></div>'
        f'<small>目标 {exercise_target:g} 分钟 · {exercise["total_distance_km"]:.1f} km</small>'
        '</article>'
        '<article class="health-metric" '
        'style="--metric-color:#2f765e;'
        f'--metric-progress:{weight_progress}%">'
        '<label>最近体重</label>'
        f'<strong>{latest_weight_value}</strong>'
        f'<small>{latest_weight_unit}</small>'
        '<div class="metric-track"><i></i></div>'
        f'<small>{weight["count"]} 次记录</small>'
        '</article>'
        '</div>'
        '<div class="summary-note">'
        f'统计日期 {summary["summary_date"]} · '
        f'{summary["timezone"]}。'
        '所有数值只来自已确认记录；热量为可追溯估算，'
        '不构成医疗建议。'
        '</div>'
        '</section>'
    )


def refresh_today() -> tuple[
    list[list[Any]],
    str,
]:
    """刷新今天的四类事件和汇总。"""

    try:
        today = _today_string()
    except ValueError as exc:
        return (
            [],
            _error_text(
                "TIMEZONE_INVALID",
                str(exc),
            ),
        )

    result = get_healthos_daily_summary(
        user_id=LOCAL_USER_ID,
        date=today,
        timezone_name=APP_TIMEZONE,
        store=event_store,
        healthos_store=healthos_store,
    )

    if not result["ok"]:
        error = result["error"]

        return (
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
        )

    data = result["data"]

    return (
        _timeline_rows(
            data["events"]
        ),
        _summary_markdown(
            data["summary"],
            data.get("goal_gaps", []),
        ),
    )


_COACH_STYLE_LABELS = {
    "gentle": "温和陪伴",
    "rational": "理性复盘",
    "concise": "简洁提醒",
    "goal_focused": "目标督促",
}

_GOAL_STATUS_LABELS = {
    "active": "进行中",
    "paused": "已暂停",
    "completed": "已完成",
}

_GOAL_PERIOD_LABELS = {
    "daily": "每天",
    "weekly": "每周",
    "monthly": "每月",
    "8_weeks": "8 周",
}

_REMINDER_STATUS_LABELS = {
    "scheduled": "已安排",
    "fired": "已触发",
    "completed": "已完成",
    "snoozed": "已延后",
    "paused": "已暂停",
    "cancelled": "已取消",
    "failed": "执行失败",
}


def _profile_markdown(profile: dict[str, Any]) -> str:
    """将最小档案转换为用户可读设置摘要。"""

    style = _COACH_STYLE_LABELS.get(
        str(profile.get("coach_style", "gentle")),
        "温和陪伴",
    )
    preferences = profile.get("dietary_preferences") or []
    exclusions = profile.get("exclusions") or []
    quiet_start = profile.get("quiet_hours_start")
    quiet_end = profile.get("quiet_hours_end")
    quiet_text = (
        f"{quiet_start}-{quiet_end}"
        if quiet_start and quiet_end
        else "未设置"
    )
    reminder_text = "开启" if profile.get("reminders_enabled", True) else "关闭"
    return (
        '<section class="profile-summary">'
        '<div><span>教练风格</span>'
        f'<strong>{escape(style)}</strong></div>'
        '<div><span>时区</span>'
        f'<strong>{escape(str(profile.get("timezone_name", APP_TIMEZONE)))}</strong></div>'
        '<div><span>提醒</span>'
        f'<strong>{reminder_text}</strong><small>免打扰 {escape(quiet_text)}</small></div>'
        '<div><span>饮食偏好</span>'
        f'<strong>{escape("、".join(preferences) if preferences else "未设置")}</strong>'
        f'<small>忌口：{escape("、".join(exclusions) if exclusions else "未设置")}</small></div>'
        '</section>'
    )


def _goal_rows(goals: list[dict[str, Any]]) -> list[list[Any]]:
    """目标表格只展示业务含义，不展示内部 ID。"""

    rows: list[list[Any]] = []
    for goal in goals:
        versions = goal.get("versions") or []
        if not versions:
            continue
        current = versions[-1]
        created = str(current.get("created_at", ""))
        rows.append(
            [
                current.get("title", "健康目标"),
                f"{current.get('target_value', '')} {current.get('unit', '')}",
                _GOAL_PERIOD_LABELS.get(str(current.get("period", "")), str(current.get("period", ""))),
                _GOAL_STATUS_LABELS.get(str(current.get("status", "")), str(current.get("status", ""))),
                f"第 {current.get('version', len(versions))} 版",
                created[:10] if created else "未记录",
            ]
        )
    return rows


def _reminder_rows(reminders: list[dict[str, Any]]) -> list[list[Any]]:
    """提醒表格隐藏内部标识和确认令牌。"""

    rows: list[list[Any]] = []
    for reminder in reminders:
        scheduled = str(reminder.get("scheduled_for", ""))
        try:
            scheduled_text = datetime.fromisoformat(scheduled).astimezone(_timezone()).strftime("%m月%d日 %H:%M")
        except (ValueError, TypeError):
            scheduled_text = scheduled or "时间未知"
        rows.append(
            [
                reminder.get("content", "健康提醒"),
                scheduled_text,
                _REMINDER_STATUS_LABELS.get(
                    str(reminder.get("status", "")),
                    str(reminder.get("status", "")),
                ),
                reminder.get("timezone_name", APP_TIMEZONE),
                len(reminder.get("transitions") or []),
            ]
        )
    return rows


def _checkin_markdown(
    period: dict[str, Any],
    goals: list[dict[str, Any]],
) -> str:
    """分开呈现事实、数据完整度和下一步，不推断因果。"""

    days = int(period.get("period", {}).get("days", 7))
    completeness = float(period.get("data_completeness", 0))
    exercise = period.get("exercise", {})
    water = period.get("water", {})
    meal = period.get("meal", {})
    weight = period.get("weight", {})
    active_goals = [
        goal for goal in goals
        if goal.get("versions") and goal["versions"][-1].get("status") == "active"
    ]
    if not period.get("event_count"):
        action = "先记录一件最容易的事，例如今天喝了多少水。"
    elif completeness < 0.5:
        action = "记录覆盖还不完整，继续记录几天后再看趋势会更可靠。"
    elif active_goals:
        action = f"回看“{active_goals[0]['versions'][-1].get('title', '当前目标')}”并记录今天的进度。"
    else:
        action = "可以创建一个可衡量的健康目标，让后续复盘有明确参照。"
    change = weight.get("change_kg")
    change_text = "证据不足" if change is None else f"{change:+g} kg"
    return (
        '<section class="checkin-summary">'
        '<div class="checkin-facts">'
        f'<strong>最近 {days} 天的已记录事实</strong>'
        '<ul>'
        f'<li>运动 {float(exercise.get("total_minutes", 0)):g} 分钟</li>'
        f'<li>饮水 {float(water.get("total_ml", 0)):g} ml</li>'
        f'<li>餐食 {int(meal.get("count", 0))} 次，已记录热量约 {float(meal.get("calories_kcal", 0)):g} kcal</li>'
        f'<li>体重变化：{escape(change_text)}</li>'
        '</ul></div>'
        '<div class="checkin-evidence">'
        f'<span>数据完整度</span><strong>{completeness * 100:.0f}%</strong>'
        f'<small>{int(period.get("days_with_data", 0))}/{days} 天有记录</small></div>'
        '<div class="checkin-action"><span>建议下一步</span>'
        f'<strong>{escape(action)}</strong></div>'
        '<p>这里只陈述已保存记录；数据不足时不推断体重或饮食变化的原因。</p>'
        '</section>'
    )


def refresh_healthos_dashboard(period_days: int = 7) -> tuple[Any, ...]:
    """刷新档案、目标、周期复盘和本地提醒。"""

    profile_result = get_user_profile(
        user_id=LOCAL_USER_ID,
        timezone_name=APP_TIMEZONE,
        store=healthos_store,
    )
    goals_result = get_health_goals(
        user_id=LOCAL_USER_ID,
        store=healthos_store,
    )
    period_result = get_period_summary(
        user_id=LOCAL_USER_ID,
        days=int(period_days),
        end_date=None,
        timezone_name=APP_TIMEZONE,
        store=event_store,
        healthos_store=healthos_store,
    )
    reminders_result = list_or_cancel_reminders(
        user_id=LOCAL_USER_ID,
        action="list",
        idempotency_key="ui-read-only",
        store=healthos_store,
    )
    results = [profile_result, goals_result, period_result, reminders_result]
    failed = next((item for item in results if not item.get("ok")), None)
    if failed is not None:
        error = failed.get("error") or {}
        status = _error_text(
            str(error.get("error_code", "HEALTHOS_READ_ERROR")),
            str(error.get("message", "无法读取 HealthOS 数据")),
        )
        return status, [], status, [], status

    profile = profile_result["data"]["profile"]
    goals = goals_result["data"]["goals"]
    period = period_result["data"]
    reminders = reminders_result["data"]["reminders"]
    return (
        _profile_markdown(profile),
        _goal_rows(goals),
        _checkin_markdown(period, goals),
        _reminder_rows(reminders),
        f"已读取 {len(reminders)} 条本地提醒；取消、延后、暂停和恢复都需要确认。",
    )


def open_healthos_action(prompt: str) -> tuple[Any, str, dict[str, Any], Any]:
    """从目标或提醒页面进入统一对话入口。"""

    return (
        gr.Tabs(selected="chat"),
        prompt,
        {},
        gr.Markdown(value="", visible=False, sanitize_html=False, container=False),
    )


def open_profile_settings() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("请先显示我的个人设置，我想调整教练风格或提醒偏好。")


def open_goal_creation() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("我想创建一个健康目标：")


def open_goal_management() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("请先列出我的健康目标，我想调整、暂停或恢复其中一个。")


def open_period_review() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("请根据我最近的已确认记录和目标做一次复盘，分开说明事实、数据不足和建议。")


def open_knowledge_question() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("我想询问一个一般健康生活问题，请给出可信来源：")


def open_reminder_creation() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("我想创建一个提醒：")


def open_reminder_management() -> tuple[Any, str, dict[str, Any], Any]:
    return open_healthos_action("请列出我的提醒，我想延后、暂停、恢复或取消其中一个。")


def _tool_contract_rows() -> list[list[str]]:
    """为开发者证据页展示 15 个工具的安全契约。"""

    boundaries = {
        "get_user_profile": "最小字段",
        "prepare_profile_update": "确认后写入",
        "get_health_goals": "含版本历史",
        "prepare_goal_change": "确认后追加版本",
        "get_health_events": "仅 committed 事实",
        "prepare_health_event": "缺参追问",
        "prepare_event_change": "展示前后对比",
        "retrieve_nutrition_candidates": "Top-K 与来源",
        "calculate_nutrition": "确定性公式",
        "retrieve_health_knowledge": "引用与拒答",
        "get_daily_summary": "当天事实与完整度",
        "get_period_summary": "7/14/30 天、不推因果",
        "create_reminder_draft": "确认前不安排",
        "execute_reminder": "令牌与幂等键",
        "list_or_cancel_reminders": "写操作生成草稿",
    }
    risk_labels = {
        "read": "读取",
        "draft": "草稿",
        "retrieval": "检索",
        "calculation": "计算",
        "write": "写入",
        "read_or_draft": "读取/草稿",
    }
    return [
        [
            name,
            risk_labels.get(str(tool_router.tool_contracts[name]["risk_level"]), "受控"),
            f"{boundaries[name]} · {tool_router.tool_contracts[name]['timeout_seconds']}s 预算",
        ]
        for name in tool_router.available_tools
    ]


def _selected_record_content(
    event: HealthEvent,
) -> str:
    """生成从每日记录进入对话后的编辑上下文。"""

    occurred_at = (
        event.occurred_at
        .astimezone(_timezone())
        .strftime("%m月%d日 %H:%M")
    )
    summary = _event_detail(event)

    return (
        '<section class="selected-record-card" '
        'role="status" aria-live="polite">'
        '<div><span>正在修改</span>'
        f'<strong>{escape(_event_type_label(event))}记录</strong></div>'
        f'<p>{escape(occurred_at)}，{escape(summary)}</p>'
        '<small>请在输入框说明要改成什么。保存前仍会请你确认。</small>'
        "</section>"
    )


def open_today_record_in_chat(
    event: gr.SelectData,
) -> tuple[Any, str, dict[str, str], Any]:
    """选择今日记录后进入对话修改流程。"""

    raw_index = event.index
    row_index = (
        raw_index[0]
        if isinstance(
            raw_index,
            (list, tuple),
        )
        else raw_index
    )

    try:
        selected_index = int(row_index)
        today = _today_string()
    except (TypeError, ValueError):
        return (
            gr.Tabs(selected="today"),
            "",
            {},
            gr.Markdown(
                value=(
                    "没有识别到这条记录，"
                    "请刷新今日记录后重试。"
                ),
                visible=True,
            ),
        )

    result = get_daily_health_summary(
        user_id=LOCAL_USER_ID,
        date=today,
        timezone_name=APP_TIMEZONE,
        store=event_store,
    )
    events = (
        result.get("data", {}).get(
            "events",
            [],
        )
        if result.get("ok")
        else []
    )

    if not (
        0 <= selected_index < len(events)
    ):
        return (
            gr.Tabs(selected="today"),
            "",
            {},
            gr.Markdown(
                value=(
                    "这条记录刚刚发生了变化，"
                    "请刷新后重新选择。"
                ),
                visible=True,
            ),
        )

    selected_event = (
        HealthEvent.model_validate(
            events[selected_index]
        )
    )
    occurred_at = (
        selected_event.occurred_at
        .astimezone(_timezone())
        .strftime("%m月%d日 %H:%M")
    )
    selected_state = {
        "event_id": str(
            selected_event.event_id
        ),
        "event_type": (
            selected_event.event_type.value
        ),
        "occurred_at": occurred_at,
        "summary": _event_detail(
            selected_event
        ),
    }

    return (
        gr.Tabs(selected="chat"),
        "请把这条记录修改为：",
        selected_state,
        gr.Markdown(
            value=_selected_record_content(
                selected_event
            ),
            visible=True,
            sanitize_html=False,
            container=False,
        ),
    )


def open_record_workspace() -> Any:
    """从工作台进入主要记录入口。"""

    return gr.Tabs(selected="chat")


def open_today_workspace() -> Any:
    """从记录对话查看已确认的今日结果。"""

    return gr.Tabs(selected="today")


def conversation_starter_message(
    message: str,
) -> str:
    """把首页快捷动作转换成真实对话输入。"""

    return message


def refresh_timeline(
    date_value: str,
    event_type_value: str,
) -> tuple[
    list[list[Any]],
    str,
]:
    """按日期和事件类型刷新时间线。"""

    selected_date = (
        date_value.strip()
        if isinstance(
            date_value,
            str,
        )
        else ""
    )

    if not selected_date:
        try:
            selected_date = (
                _today_string()
            )
        except ValueError as exc:
            return (
                [],
                _error_text(
                    "TIMEZONE_INVALID",
                    str(exc),
                ),
            )

    selected_event_type = (
        event_type_value.strip()
        if isinstance(
            event_type_value,
            str,
        )
        else ""
    )

    result = query_health_events(
        user_id=LOCAL_USER_ID,
        event_type=(
            selected_event_type
            or None
        ),
        date=selected_date,
        timezone_name=APP_TIMEZONE,
        newest_first=False,
        limit=500,
        store=event_store,
    )

    if not result["ok"]:
        error = result["error"]

        return (
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
        )

    data = result["data"]

    return (
        _timeline_rows(
            data["events"]
        ),
        (
            f"找到 {data['matched_count']} "
            "条已保存事件。"
        ),
    )


def search_candidates(
    image_path: str | None,
    food_query: str,
) -> tuple[
    gr.Dropdown,
    list[list[Any]],
    str,
    dict[str, Any],
]:
    """校验图片并检索人工填写的食物名称。"""

    image_result = validate_image(
        image_path
    )

    if not image_result.ok:
        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                (
                    image_result.error_code
                    or "IMAGE_INVALID"
                ),
                image_result.message,
            ),
            {},
        )

    tool_result = (
        retrieve_nutrition_candidates(
            query=food_query,
            top_k=5,
            repository=repository,
        )
    )

    if not tool_result["ok"]:
        error = tool_result["error"]

        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                error["error_code"],
                error["message"],
            ),
            {},
        )

    data = tool_result["data"]
    trace = data["trace"]

    if data["status"] == "not_found":
        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            [],
            _error_text(
                "NOT_FOUND",
                "没有找到可靠食物候选，"
                "不会猜测营养值。",
            ),
            trace,
        )

    choices: list[
        tuple[str, str]
    ] = []

    rows: list[
        list[Any]
    ] = []

    for candidate in data["candidates"]:
        choices.append(
            (
                f"{candidate['name']}｜"
                f"{candidate['category']}｜"
                f"stage {candidate['stage']} "
                f"{candidate['match_type']}",
                candidate["food_id"],
            )
        )

        rows.append(
            [
                candidate["food_id"],
                candidate["name"],
                candidate["category"],
                candidate["stage"],
                candidate["match_type"],
                candidate["matched_term"],
                candidate["score"],
                candidate["source"],
                candidate["source_version"],
                candidate["candidate_source"],
            ]
        )

    selected_value = (
        data["candidates"][0]["food_id"]
        if data["auto_select_allowed"]
        else None
    )

    warning_text = ""

    if data["trace_warning"] is not None:
        warning = data["trace_warning"]

        warning_text = (
            "\n\nTrace 警告 "
            f"[{warning['error_code']}]："
            f"{warning['message']}"
        )

    status = (
        f"已加载 {len(rows)} 个候选。"
        "归一化检索词："
        f"`{data['normalized_query']}`；"
        f"模式：`{data['selection_mode']}`；"
        f"数据集：`{data['dataset_id']}`；"
        f"耗时：{data['elapsed_ms']:.3f} ms。"
        + (
            "已按规则预选。"
            if selected_value
            else "候选有歧义，请手动选择。"
        )
        + warning_text
    )

    return (
        gr.Dropdown(
            choices=choices,
            value=selected_value,
        ),
        rows,
        status,
        trace,
    )


def calculate_meal_preview(
    image_path: str | None,
    food_query: str,
    selected_food_id: str | None,
    raw_grams: Any,
) -> tuple[
    str,
    dict[str, Any] | None,
    str,
    str,
]:
    """计算饮食营养并生成保存草稿。"""

    image_result = validate_image(
        image_path
    )

    if not image_result.ok:
        return (
            _error_text(
                (
                    image_result.error_code
                    or "IMAGE_INVALID"
                ),
                image_result.message,
            ),
            None,
            "",
            "",
        )

    try:
        search_result = repository.search(
            food_query,
            top_k=5,
        )

        if search_result.status == "not_found":
            return (
                _error_text(
                    "NOT_FOUND",
                    "没有可靠食物数据，"
                    "不能计算或保存。",
                ),
                None,
                "",
                "",
            )

        candidate_ids = {
            candidate.food_id
            for candidate
            in search_result.candidates
        }

        if (
            not selected_food_id
            or selected_food_id
            not in candidate_ids
        ):
            return (
                _error_text(
                    "CANDIDATE_REQUIRED",
                    "请选择当前检索结果"
                    "中的一个候选。",
                ),
                None,
                "",
                "",
            )

        food = repository.get_by_food_id(
            selected_food_id
        )

        grams = parse_grams(
            raw_grams
        )

        estimate = calculate_nutrition(
            food=food,
            raw_grams=grams,
            retrieval_query=food_query,
        )

    except NutritionDataError as exc:
        return (
            _error_text(
                exc.error_code,
                exc.message,
            ),
            None,
            "",
            "",
        )

    except NutritionCalculationError as exc:
        return (
            _error_text(
                exc.error_code,
                exc.message,
            ),
            None,
            "",
            "",
        )

    now = datetime.now(
        timezone.utc
    )

    draft_result = prepare_health_event(
        event_input={
            "event_type": "meal",
            "payload": {
                "food": {
                    "food_id": (
                        food.food_id
                    ),
                    "name": (
                        food.name
                    ),
                    "category": (
                        food.category
                    ),
                },
                "portion": {
                    "grams": (
                        float(grams)
                    ),
                    "unit": "g",
                },
                "nutrition": (
                    estimate.model_dump(
                        mode="json"
                    )
                ),
                "retrieval_query": (
                    food_query.strip()
                ),
                "candidate_source": (
                    "manual"
                ),
                "estimated": True,
            },
            "source_refs": [
                estimate.source_ref
            ],
            "input_source": "image",
            "occurred_at": (
                now.isoformat()
            ),
        },
        user_id=LOCAL_USER_ID,
        idempotency_key=str(
            uuid4()
        ),
        now=now,
    )

    if not draft_result["ok"]:
        error = draft_result["error"]

        return (
            _error_text(
                error["error_code"],
                error["message"],
            ),
            None,
            "",
            "",
        )

    preview_state = draft_result["data"]

    summary = (
        "### 待确认饮食记录\n\n"
        f"- 食物：{food.name}（{food.food_id}）\n"
        f"- 份量：{float(grams):g} g\n"
        f"- 热量估算：{estimate.calories_kcal:.2f} kcal\n"
        f"- 蛋白质估算：{estimate.protein_g:.2f} g\n"
        f"- 脂肪估算：{estimate.fat_g:.2f} g\n"
        f"- 碳水估算：{estimate.carbs_g:.2f} g\n"
        f"- 数据来源：{food.source}\n"
        f"- 来源版本：{food.source_version}\n"
        f"- 检索词：{estimate.retrieval_query}\n"
        f"- 份量假设：{estimate.portion_assumption}\n\n"
        "**尚未保存。以上均为估算值，"
        "仅供学习，不构成医疗建议。**"
    )

    evidence = (
        "### 可重算证据\n\n"
        f"- food_id：`{food.food_id}`\n"
        f"- 克重：`{float(grams):g} g`\n"
        f"- 热量：`{food.calories_per_100g:g} "
        "kcal/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.calories_kcal:.2f} kcal`\n"
        f"- 蛋白质：`{food.protein_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.protein_g:.2f} g`\n"
        f"- 脂肪：`{food.fat_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.fat_g:.2f} g`\n"
        f"- 碳水：`{food.carbs_per_100g:g} "
        "g/100g × "
        f"{float(grams):g}g ÷ 100 "
        f"= {estimate.carbs_g:.2f} g`"
    )

    return (
        summary,
        preview_state,
        (
            "计算完成，请核对后"
            "点击确认保存。"
        ),
        evidence,
    )


def confirm_meal_save(
    preview_state: (
        dict[str, Any]
        | None
    ),
) -> tuple[
    str,
    list[list[Any]],
    str,
]:
    """明确点击后保存饮食草稿。"""

    if not preview_state:
        rows, summary = refresh_today()

        return (
            _error_text(
                "PREVIEW_REQUIRED",
                "请先完成检索和"
                "营养计算。",
            ),
            rows,
            summary,
        )

    result = save_health_event(
        event_input=(
            preview_state["event"]
        ),
        confirmation_token=(
            preview_state[
                "confirmation_token"
            ]
        ),
        idempotency_key=(
            preview_state[
                "idempotency_key"
            ]
        ),
        store=event_store,
    )

    rows, summary = refresh_today()

    if not result["ok"]:
        error = result["error"]

        return (
            _error_text(
                error["error_code"],
                error["message"],
            ),
            rows,
            summary,
        )

    if result["data"]["idempotent"]:
        status = (
            "该饮食草稿已保存过，"
            "没有新增重复记录。"
        )
    else:
        status = (
            "饮食记录已保存。"
            "今日概览已同步更新。"
        )

    return (
        status,
        rows,
        summary,
    )


def cancel_meal_preview() -> tuple[
    None,
    str,
    str,
    str,
]:
    """取消饮食草稿。"""

    return (
        None,
        (
            "已取消，未写入"
            "任何饮食记录。"
        ),
        "尚未计算待确认记录。",
        (
            "### 可重算证据\n\n"
            "尚未选择数据行并计算。"
        ),
    )


def send_chat_message(
    user_text: str,
    history: (
        list[dict[str, Any]]
        | None
    ),
    selected_record: (
        dict[str, Any]
        | None
    ),
    request: gr.Request,
) -> tuple[Any, ...]:
    """向当前浏览器的 Agent Session 发送消息。"""

    chat_history = list(
        history or []
    )
    selected_state = (
        dict(selected_record)
        if isinstance(
            selected_record,
            dict,
        )
        else {}
    )

    normalized_text = (
        user_text.strip()
        if isinstance(
            user_text,
            str,
        )
        else ""
    )

    if not normalized_text:
        confirmation_updates = (
            _confirmation_updates(None)
        )
        return (
            chat_history,
            "",
            "请输入内容。",
            [],
            {},
            *confirmation_updates,
            selected_state,
            gr.skip(),
        )

    chat_history.append(
        {
            "role": "user",
            "content": normalized_text,
        }
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型当前未启用。"
            "请在 `.env` 中配置 Provider，"
            "或继续使用饮食手动录入、"
            "时间线和每日汇总。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            AGENT_PROVIDER_STATUS,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
            *_confirmation_updates(None),
            selected_state,
            gr.skip(),
        )

    model_text = normalized_text
    selected_event_id = str(
        selected_state.get(
            "event_id",
            "",
        )
    ).strip()
    is_edit_request = any(
        term in normalized_text
        for term in (
            "修改",
            "改成",
            "改为",
            "删除",
            "移除",
            "这条记录",
        )
    )
    if not is_edit_request:
        selected_event_id = ""
    elif selected_event_id:
        try:
            UUID(selected_event_id)
        except ValueError:
            selected_event_id = ""

    if selected_event_id:
        model_text = (
            "[内部选中记录]\n"
            f"event_id={selected_event_id}\n"
            "请仅将该标识用于工具调用，"
            "不得在回答中展示。\n"
            "用户原始请求："
            f"{normalized_text}"
        )

    try:
        result = session.send(
            model_text
        )

    except AgentProviderError as exc:
        answer = (
            "模型调用失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            answer,
            [],
            {
                "state": (
                    "provider_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
            *_confirmation_updates(
                session.state.pending_confirmation
            ),
            selected_state,
            gr.skip(),
        )

    except Exception as exc:
        answer = (
            "Agent 运行失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            "",
            answer,
            [],
            {
                "state": (
                    "agent_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
            *_confirmation_updates(
                session.state.pending_confirmation
            ),
            selected_state,
            gr.skip(),
        )

    _persist_agent_session(session)

    visible_answer = result.answer
    if result.pending_confirmation is not None:
        visible_answer = (
            "我已经整理好这条健康记录。"
            "请在下方核对内容，"
            "确认后才会写入。"
        )

    chat_history.append(
        {
            "role": "assistant",
            "content": visible_answer,
        }
    )

    return (
        chat_history,
        "",
        _agent_status_text(result),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
        *_confirmation_updates(
            session.state.pending_confirmation
        ),
        {},
        gr.Markdown(
            value="",
            visible=False,
            sanitize_html=False,
            container=False,
        ),
    )


def confirm_agent_action(
    history: (
        list[dict[str, Any]]
        | None
    ),
    request: gr.Request,
) -> tuple[Any, ...]:
    """用户点击按钮后确认 Agent 草稿。"""

    chat_history = list(
        history or []
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型未启用，"
            "当前没有可确认草稿。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        rows, summary = refresh_today()

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
            rows,
            summary,
            *_confirmation_updates(None),
        )

    try:
        result = session.confirm()

    except Exception as exc:
        answer = (
            "确认执行失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        rows, summary = refresh_today()

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "confirmation_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
            rows,
            summary,
            *_confirmation_updates(
                session.state.pending_confirmation
            ),
        )

    _persist_agent_session(session)

    chat_history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    rows, summary = refresh_today()

    return (
        chat_history,
        _agent_status_text(result),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
        rows,
        summary,
        *_confirmation_updates(
            session.state.pending_confirmation
        ),
    )


def cancel_agent_action(
    history: (
        list[dict[str, Any]]
        | None
    ),
    request: gr.Request,
) -> tuple[Any, ...]:
    """取消 Agent pending_task 或待确认草稿。"""

    chat_history = list(
        history or []
    )

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = (
            "Agent 模型未启用，"
            "当前没有待取消任务。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "provider_disabled"
                )
            },
            *_confirmation_updates(None),
        )

    try:
        result = session.cancel()

    except Exception as exc:
        answer = (
            "取消操作失败："
            f"{exc}"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            answer,
            [],
            {
                "state": (
                    "cancellation_error"
                ),
                "trace": {
                    "enabled": True,
                    "write_warning": (
                        session
                        .last_trace_warning
                    ),
                },
            },
            *_confirmation_updates(
                session.state.pending_confirmation
            ),
        )

    _persist_agent_session(session)

    chat_history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    return (
        chat_history,
        _agent_status_text(result),
        _tool_steps_json(
            result
        ),
        _result_state_json(
            session=session,
            result=result,
        ),
        *_confirmation_updates(
            session.state.pending_confirmation
        ),
    )


def reset_agent_conversation(
    request: gr.Request,
) -> tuple[Any, ...]:
    """清空当前浏览器绑定的本地对话状态。"""

    session_key = _session_key(request)

    with _AGENT_SESSIONS_LOCK:
        _AGENT_SESSIONS.pop(
            session_key,
            None,
        )

    if session_key.startswith(
        "conversation-"
    ):
        try:
            conversation_store.delete(
                session_key
            )
        except (OSError, ValueError):
            pass

    return (
        _welcome_chat_history(),
        "会话已重置。",
        [],
        {
            "state": "idle",
            "trace": {
                "enabled": True,
                "path": (
                    "data/"
                    "agent_traces.jsonl"
                ),
            },
        },
        *_confirmation_updates(None),
        {},
        gr.Markdown(
            value="",
            visible=False,
            sanitize_html=False,
            container=False,
        ),
    )


def build_demo() -> gr.Blocks:
    """构建小满健康助理单页应用。"""

    initial_date = (
        _today_string()
        if APP_TIMEZONE
        else ""
    )

    with gr.Blocks(
        title="小满 · 个人健康助理",
        fill_width=True,
    ) as demo:
        browser_conversation_id = gr.BrowserState(
            default_value="",
            storage_key=(
                "xiaoman-health-conversation"
            ),
        )
        selected_record_state = gr.State(
            value={}
        )

        gr.Markdown(
            """
            <header class="brand-shell">
              <div class="brand-lockup">
                <div class="brand-mark">
                  <i>小</i>
                  <span><b>HealthOS</b><small>小满</small></span>
                </div>
                <div class="brand-message">
                  <h1>个人健康工作台</h1>
                  <p>说出刚刚发生的事，再把今天整理清楚。</p>
                </div>
              </div>
              <aside class="brand-aside">
                <div class="brand-aside-label"><i></i> 本地健康空间</div>
                <strong>每一次写入都由你决定</strong>
                <small>对话和记录可以恢复，也可以查询、修改或删除。</small>
              </aside>
            </header>
            """,
            sanitize_html=False,
            container=False,
        )

        gr.Markdown(
            """
            <div class="safety-strip">
              <span>非医疗服务</span>
              小满用于个人健康记录与学习演示，不提供诊断、治疗或紧急医疗服务。
            </div>
            """,
            sanitize_html=False,
            container=False,
        )

        with gr.Tabs(
            elem_id="main-tabs",
            selected="chat",
        ) as main_tabs:
            with gr.Tab(
                "今天",
                id="today",
                elem_id="healthos-today",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    with gr.Row(elem_classes="workspace-heading"):
                        gr.Markdown(
                            f"""
                            <div class="page-title">
                              <p class="workspace-date">{escape(initial_date)}</p>
                              <h2>今天，都记录在这里。</h2>
                              <p>每次确认保存后，这里会更新健康事实、目标差距和最近记录。草稿不会进入统计。</p>
                            </div>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        quick_record_button = gr.Button(
                            "继续记录",
                            variant="primary",
                            size="md",
                            scale=0,
                            min_width=118,
                        )

                    with gr.Row(
                        equal_height=True,
                        elem_classes="today-command-row",
                    ):
                        with gr.Column(scale=3, min_width=520):
                            today_summary = gr.Markdown(
                                "正在读取今天的汇总……",
                                sanitize_html=False,
                                container=False,
                            )

                        with gr.Column(
                            scale=1,
                            min_width=230,
                            elem_classes="day-prompt",
                        ):
                            gr.Markdown(
                                """
                                <div class="prompt-orbit">下一步</div>
                                <strong>补充刚刚发生的事</strong>
                                <p>用一句自然语言记录饮食、饮水、体重或运动。小满会先整理成草稿。</p>
                                """,
                                sanitize_html=False,
                                container=False,
                            )

                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="card-heading-row"):
                            gr.Markdown(
                                """
                              <div>
                                  <div class="section-heading">今天的记录</div>
                                  <p class="section-copy">点击任意记录即可进入对话修改，保存前仍需要确认。</p>
                              </div>
                                """,
                                sanitize_html=False,
                                container=False,
                            )
                            refresh_today_button = gr.Button(
                                "刷新今日",
                                variant="secondary",
                                size="sm",
                                scale=0,
                            )

                        today_table = gr.Dataframe(
                            headers=[
                                "时间",
                                "记录类型",
                                "记录内容",
                                "记录来源",
                                "状态",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                            ],
                            value=[],
                            interactive=False,
                            show_label=False,
                            max_height=360,
                            wrap=True,
                            elem_classes="timeline-table",
                        )

            with gr.Tab(
                "对话",
                id="chat",
                elem_id="healthos-record",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    with gr.Row(elem_classes="workspace-heading"):
                        gr.Markdown(
                            """
                            <div class="page-title conversation-title">
                              <h2>刚刚发生了什么？</h2>
                              <p>直接说一句话。小满会补齐必要信息，整理成草稿，保存前再请你确认。</p>
                            </div>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        view_today_button = gr.Button(
                            "查看今天",
                            variant="secondary",
                            size="md",
                            scale=0,
                            min_width=112,
                        )

                    with gr.Column(
                        scale=0,
                        min_width=0,
                        elem_classes="conversation-starters"
                    ):
                        gr.Markdown(
                            """
                            <div class="starter-heading">
                              <strong>从一句话开始</strong>
                              <span>像平时说话一样，不用先选表单。</span>
                            </div>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        with gr.Row(elem_classes="starter-actions"):
                            starter_buttons = [
                                (
                                    gr.Button(
                                        "我刚喝了水",
                                        variant="secondary",
                                        size="sm",
                                    ),
                                    "我刚喝了水",
                                ),
                                (
                                    gr.Button(
                                        "我今天吃了什么",
                                        variant="secondary",
                                        size="sm",
                                    ),
                                    "我今天吃了什么",
                                ),
                                (
                                    gr.Button(
                                        "我刚刚运动了",
                                        variant="secondary",
                                        size="sm",
                                    ),
                                    "我刚刚运动了",
                                ),
                                (
                                    gr.Button(
                                        "我刚称重了",
                                        variant="secondary",
                                        size="sm",
                                    ),
                                    "我刚称重了",
                                ),
                            ]

                    with gr.Row(
                        equal_height=False,
                        elem_classes="responsive-split-row",
                    ):
                        with gr.Column(
                            scale=3,
                            min_width=520,
                            elem_classes=["care-card", "chat-panel"],
                        ):
                            chatbot = gr.Chatbot(
                                value=(
                                    _welcome_chat_history()
                                ),
                                height=430,
                                show_label=False,
                                layout="bubble",
                                buttons=["copy_all"],
                                placeholder="从一件小事开始记录吧。",
                                elem_id="health-chat",
                            )

                            agent_activity = gr.Markdown(
                                value="",
                                visible=False,
                                sanitize_html=False,
                                container=False,
                                elem_classes=(
                                    "agent-activity-wrap"
                                ),
                            )

                            selected_record_context = gr.Markdown(
                                value="",
                                visible=False,
                                sanitize_html=False,
                                container=False,
                                elem_classes=(
                                    "selected-record-context"
                                ),
                            )

                            pending_agent_card = gr.Markdown(
                                value="",
                                visible=False,
                                sanitize_html=False,
                                container=False,
                                elem_classes="inline-confirmation",
                            )

                            with gr.Row(
                                elem_classes=(
                                    "inline-confirmation-actions"
                                )
                            ):
                                cancel_agent_button = gr.Button(
                                    "取消",
                                    variant="secondary",
                                    size="md",
                                    visible=False,
                                    scale=0,
                                    min_width=96,
                                )

                                confirm_agent_button = gr.Button(
                                    "确认并保存",
                                    variant="primary",
                                    size="md",
                                    visible=False,
                                    scale=0,
                                    min_width=126,
                                )

                            with gr.Row(elem_classes="composer-row"):
                                chat_input = gr.Textbox(
                                    show_label=False,
                                    placeholder="直接说：我刚喝了水",
                                    lines=1,
                                    max_lines=4,
                                    scale=8,
                                    container=True,
                                )

                                send_button = gr.Button(
                                    "发送",
                                    variant="primary",
                                    size="md",
                                    scale=1,
                                    min_width=96,
                                )

                        with gr.Column(
                            scale=1,
                            min_width=270,
                            elem_classes=["care-card", "status-card"],
                        ):
                            gr.Markdown(
                                """
                                <div class="section-heading">记录会去哪里</div>
                                <p class="section-copy">草稿先留在对话里。确认保存后，“今天”会自动更新。</p>
                                """,
                                sanitize_html=False,
                                container=False,
                            )

                            agent_status = gr.Markdown(
                                (
                                    "对话服务已连接，可以开始记录。"
                                    if agent_model is not None
                                    else (
                                        "对话服务尚未启用。"
                                        "你仍可以使用今日概览、"
                                        "时间线和饮食记录。"
                                    )
                                ),
                                elem_classes="agent-status",
                                container=False,
                            )

                            with gr.Column(elem_classes="action-row"):
                                reset_agent_button = gr.Button(
                                    "重置本次会话",
                                    variant="secondary",
                                    size="sm",
                                )

                            gr.Markdown(
                                """
                                <div class="memory-note">
                                  <strong>本地记忆已开启</strong>
                                  <span>刷新页面会恢复这段对话。点击“重置本次会话”会清除历史。</span>
                                </div>
                                <p>需要确认时，操作卡片会直接出现在对话下方。</p>
                                """,
                                sanitize_html=False,
                                elem_classes="action-help",
                                container=False,
                            )

            with gr.Tab(
                "健康时间线",
                id="timeline",
                elem_id="healthos-timeline",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>每一条，都有迹可循。</h2>
                          <p>筛选已经确认的健康记录。需要修改或删除时，告诉小满记录时间和内容即可。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="filter-row"):
                            timeline_date = gr.Textbox(
                                label="日期",
                                info="YYYY-MM-DD",
                                value=initial_date,
                                scale=2,
                            )

                            timeline_event_type = gr.Dropdown(
                                label="事件类型",
                                choices=[
                                    ("全部记录", ""),
                                    ("饮食", "meal"),
                                    ("饮水", "water"),
                                    ("体重", "weight"),
                                    ("运动", "exercise"),
                                ],
                                value="",
                                scale=2,
                            )

                            refresh_timeline_button = gr.Button(
                                "查询记录",
                                variant="primary",
                                size="md",
                                scale=1,
                            )

                        timeline_status = gr.Markdown(
                            container=False,
                        )

                        timeline_table = gr.Dataframe(
                            headers=[
                                "时间",
                                "记录类型",
                                "记录内容",
                                "记录来源",
                                "状态",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                            ],
                            value=[],
                            interactive=False,
                            show_label=False,
                            max_height=520,
                            wrap=True,
                            elem_classes="timeline-table",
                        )

            with gr.Tab(
                "目标与趋势",
                id="goals",
                elem_id="healthos-goals",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>让记录，慢慢靠近你的目标。</h2>
                          <p>目标保留每一次调整，复盘只使用已确认的健康事实；没有足够数据时会明确说明。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    with gr.Row(
                        equal_height=False,
                        elem_classes="responsive-split-row",
                    ):
                        with gr.Column(scale=2, min_width=420, elem_classes="care-card"):
                            with gr.Row(elem_classes="card-heading-row"):
                                gr.Markdown(
                                    '<div><div class="section-heading">个人设置</div>'
                                    '<p class="section-copy">只保存你明确确认的单位、偏好、时区和表达风格。</p></div>',
                                    sanitize_html=False,
                                    container=False,
                                )
                                edit_profile_button = gr.Button(
                                    "调整设置", variant="secondary", size="sm", scale=0
                                )
                            profile_summary = gr.Markdown(
                                "正在读取个人设置……",
                                sanitize_html=False,
                                container=False,
                            )

                        with gr.Column(scale=1, min_width=260, elem_classes=["care-card", "knowledge-note"]):
                            gr.Markdown(
                                """
                                <div class="section-heading">可信知识边界</div>
                                <p>一般生活建议只使用带来源的本地知识条目。目前收录 WHO 的运动与健康饮食资料。</p>
                                <p>诊断、用药、急症或证据不足的问题会停止普通建议流程。</p>
                                """,
                                sanitize_html=False,
                                container=False,
                            )
                            ask_knowledge_button = gr.Button(
                                "向小满提问", variant="secondary", size="sm"
                            )

                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="card-heading-row"):
                            gr.Markdown(
                                '<div><div class="section-heading">健康目标</div>'
                                '<p class="section-copy">调整、暂停或恢复会新增版本，不覆盖过去。</p></div>',
                                sanitize_html=False,
                                container=False,
                            )
                            create_goal_button = gr.Button(
                                "创建目标", variant="primary", size="sm", scale=0
                            )
                            manage_goal_button = gr.Button(
                                "调整目标", variant="secondary", size="sm", scale=0
                            )
                        goals_table = gr.Dataframe(
                            headers=["目标", "目标值", "周期", "状态", "版本", "更新时间"],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            value=[],
                            interactive=False,
                            show_label=False,
                            max_height=320,
                            wrap=True,
                            elem_classes="goals-table",
                        )

                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="filter-row"):
                            gr.Markdown(
                                '<div><div class="section-heading">周期复盘</div>'
                                '<p class="section-copy">事实、完整度与一个可执行的下一步。</p></div>',
                                sanitize_html=False,
                                container=False,
                            )
                            period_days = gr.Dropdown(
                                choices=[("最近 7 天", 7), ("最近 14 天", 14), ("最近 30 天", 30)],
                                value=7,
                                label="复盘周期",
                                min_width=150,
                                scale=0,
                            )
                            refresh_healthos_button = gr.Button(
                                "刷新复盘", variant="secondary", size="sm", scale=0
                            )
                        checkin_summary = gr.Markdown(
                            "正在整理已确认记录……",
                            sanitize_html=False,
                            container=False,
                        )
                        ask_review_button = gr.Button(
                            "在对话中继续复盘", variant="secondary", size="sm"
                        )

            with gr.Tab(
                "提醒",
                id="reminders",
                elem_id="healthos-reminders",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>提醒行动，也由你掌控。</h2>
                          <p>小满先展示时间、时区和影响范围。确认后才安排，之后可以延后、暂停或取消。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )
                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="card-heading-row"):
                            gr.Markdown(
                                '<div><div class="section-heading">本地提醒</div>'
                                '<p class="section-copy">当前为本地模拟 Provider，不会写入外部日历或系统通知。</p></div>',
                                sanitize_html=False,
                                container=False,
                            )
                            create_reminder_button = gr.Button(
                                "创建提醒", variant="primary", size="sm", scale=0
                            )
                            manage_reminder_button = gr.Button(
                                "管理提醒", variant="secondary", size="sm", scale=0
                            )
                        reminders_table = gr.Dataframe(
                            headers=["提醒内容", "计划时间", "状态", "时区", "状态记录"],
                            datatype=["str", "str", "str", "str", "number"],
                            value=[],
                            interactive=False,
                            show_label=False,
                            max_height=420,
                            wrap=True,
                            elem_classes="reminders-table",
                        )
                        reminder_status = gr.Markdown(container=False)

                    gr.Markdown(
                        """
                        <section class="reminder-boundary">
                          <strong>本地提醒的能力边界</strong>
                          <p>页面运行时可以安排和回查提醒状态；关闭应用后不会像手机系统闹钟一样主动弹出通知。</p>
                        </section>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

            with gr.Tab(
                "餐食图片",
                id="meal",
                elem_id="healthos-meal",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>一张照片，认真确认这一餐。</h2>
                          <p>图片只作为记录入口。你确认食物与份量后，小满才从数据源检索并计算营养。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    meal_preview_state = gr.State(value=None)

                    with gr.Row(
                        equal_height=False,
                        elem_classes="responsive-split-row",
                    ):
                        with gr.Column(
                            scale=1,
                            min_width=330,
                            elem_classes=["care-card", "meal-step"],
                        ):
                            gr.Markdown(
                                """
                                <div class="section-heading"><i class="meal-step-number">1</i>添加与描述</div>
                                <p class="section-copy">上传 JPG 或 PNG，并手动填写你认为最接近的食物名称。</p>
                                """,
                                sanitize_html=False,
                                container=False,
                            )

                            image_input = gr.File(
                                label="餐食图片",
                                file_count="single",
                                file_types=[
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                ],
                                type="filepath",
                                height=190,
                                elem_classes="meal-dropzone",
                            )

                            food_query = gr.Textbox(
                                label="食物名称",
                                placeholder="例如：西红柿炒蛋",
                            )

                            grams_input = gr.Number(
                                label="估计份量（g）",
                                minimum=0.01,
                                maximum=10000,
                            )

                            search_button = gr.Button(
                                "查找可靠候选",
                                variant="secondary",
                            )

                            candidate_status = gr.Markdown(
                                container=False,
                            )

                            with gr.Accordion(
                                "查看候选检索证据",
                                open=False,
                            ):
                                candidate_table = gr.Dataframe(
                                    headers=[
                                        "food_id",
                                        "name",
                                        "category",
                                        "stage",
                                        "match_type",
                                        "matched_term",
                                        "score",
                                        "source",
                                        "source_version",
                                        "candidate_source",
                                    ],
                                    datatype=[
                                        "str",
                                        "str",
                                        "str",
                                        "number",
                                        "str",
                                        "str",
                                        "number",
                                        "str",
                                        "str",
                                        "str",
                                    ],
                                    value=[],
                                    interactive=False,
                                    show_label=False,
                                    max_height=250,
                                    elem_classes="candidate-table",
                                )

                        with gr.Column(
                            scale=1,
                            min_width=330,
                            elem_classes=["care-card", "meal-step"],
                        ):
                            gr.Markdown(
                                """
                                <div class="section-heading"><i class="meal-step-number">2</i>核对与保存</div>
                                <p class="section-copy">候选不会被静默选中。请核对数据行，再生成待确认草稿。</p>
                                """,
                                sanitize_html=False,
                                container=False,
                            )

                            selected_food = gr.Dropdown(
                                label="确认食物候选",
                                choices=[],
                                value=None,
                                interactive=True,
                            )

                            calculate_button = gr.Button(
                                "生成营养估算",
                                variant="primary",
                            )

                            calculation_status = gr.Markdown(
                                container=False,
                            )

                            meal_preview = gr.Markdown(
                                "尚未生成待确认记录。",
                                elem_classes="meal-preview",
                            )

                            with gr.Accordion(
                                "查看确定性计算公式",
                                open=False,
                            ):
                                recompute_evidence = gr.Markdown(
                                    "### 可重算证据\n\n"
                                    "尚未选择数据行并计算。"
                                )

                            with gr.Row(elem_classes="action-row"):
                                meal_save_button = gr.Button(
                                    "确认保存",
                                    variant="primary",
                                )

                                meal_cancel_button = gr.Button(
                                    "取消草稿",
                                    variant="stop",
                                )

                            meal_save_status = gr.Markdown(
                                container=False,
                            )

            with gr.Tab(
                "运行证据",
                id="developer",
                elem_id="healthos-evidence",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>透明，是产品的一部分。</h2>
                          <p>供开发与验收使用。只展示脱敏状态、工具名称和错误码，不展示健康参数值或确认令牌。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    with gr.Accordion(
                        "HealthOS · 15 个受控工具契约",
                        open=False,
                    ):
                        gr.Markdown(
                            "模型只能提出这些工具调用。所有参数先过 Schema；产生副作用的操作还要经过统一确认中间件。",
                            container=False,
                        )
                        gr.Dataframe(
                            headers=["工具", "性质", "关键边界"],
                            datatype=["str", "str", "str"],
                            value=_tool_contract_rows(),
                            interactive=False,
                            show_label=False,
                            max_height=480,
                            wrap=True,
                            elem_classes="tool-contract-table",
                        )

                    with gr.Accordion(
                        "本轮上下文 · 五层 Prompt Pipeline",
                        open=False,
                    ):
                        gr.Markdown(
                            "每层只装载完成当前任务所需的信息。页面展示来源和状态，"
                            "不展示隐藏思维、原始 Tool Result 或敏感参数。",
                            container=False,
                        )
                        gr.Dataframe(
                            headers=["层级", "内容", "可信边界"],
                            datatype=["str", "str", "str"],
                            value=[
                                ["1 · 系统规则", "安全、工具与确认协议", "应用版本控制"],
                                ["2 · 用户输入", "本轮明确表达", "只作为当前请求"],
                                ["3 · 用户档案", "时区、单位、已确认偏好", "不保存模型推断"],
                                ["4 · 目标与待办", "活动目标、待补充任务", "旧版本可追溯"],
                                ["5 · 可信结果", "工具结果与用户确认事实", "工具结果优先"],
                            ],
                            interactive=False,
                            show_label=False,
                            wrap=True,
                        )

                    with gr.Row(
                        equal_height=False,
                        elem_classes="responsive-split-row",
                    ):
                        with gr.Column(
                            scale=1,
                            elem_classes="care-card",
                        ):
                            latest_agent_steps = gr.JSON(
                                value=[],
                                label="最近一次工具执行",
                            )

                        with gr.Column(
                            scale=1,
                            elem_classes="care-card",
                        ):
                            latest_agent_state = gr.JSON(
                                value={
                                    "state": "idle",
                                    "trace": {
                                        "enabled": True,
                                        "path": (
                                            "data/"
                                            "agent_traces.jsonl"
                                        ),
                                    },
                                },
                                label="当前运行状态",
                            )

                    with gr.Column(elem_classes="care-card"):
                        gr.Markdown(
                            "#### 持久化 Agent Trace\n"
                            "每次发送、确认和取消都会写入脱敏 Trace。"
                        )

                        with gr.Row(elem_classes="filter-row"):
                            agent_trace_limit = gr.Number(
                                label="读取条数",
                                value=20,
                                minimum=1,
                                maximum=200,
                                precision=0,
                                scale=1,
                            )

                            refresh_agent_trace_button = gr.Button(
                                "刷新 Trace",
                                variant="secondary",
                                size="md",
                                scale=1,
                            )

                        recent_agent_traces = gr.JSON(
                            value=refresh_agent_traces(),
                            label="最近 Agent Trace",
                        )

                    with gr.Accordion(
                        "最近一次饮食检索 Trace",
                        open=False,
                    ):
                        latest_retrieval_trace = gr.JSON(
                            value={},
                            label="RetrievalTrace",
                        )

            with gr.Tab(
                "数据与隐私",
                id="privacy",
                elem_id="healthos-privacy",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>你的健康数据，只属于你。</h2>
                          <p>小满把数据边界放在界面上，而不是藏在一段很长的服务条款里。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    gr.Markdown(
                        """
                        <section class="care-card privacy-grid">
                          <article class="privacy-item"><i>本</i><div><b>业务数据，本地保存</b><span>档案、目标、健康事实、提醒与会话写入本机 SQLite；支持事务和索引查询。</span></div></article>
                          <article class="privacy-item"><i>迹</i><div><b>JSONL 只做 Trace 与导出</b><span>运行轨迹默认脱敏，不保存健康参数值、确认令牌或隐藏思维。</span></div></article>
                          <article class="privacy-item"><i>图</i><div><b>餐食图片不复制</b><span>原始图片只作为当次输入，不复制进健康记录。</span></div></article>
                          <article class="privacy-item"><i>会</i><div><b>对话历史，本地保存</b><span>刷新页面会恢复历史与上下文；重置本次会话后删除。</span></div></article>
                          <article class="privacy-item"><i>确</i><div><b>写操作必须确认</b><span>保存、修改和删除都先生成草稿，再由你确认。</span></div></article>
                          <article class="privacy-item"><i>界</i><div><b>迁移可以回滚</b><span>旧 JSON/JSONL 不会被迁移程序删除；切换存储配置即可回退。</span></div></article>
                        </section>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

        quick_record_button.click(
            fn=open_record_workspace,
            outputs=[main_tabs],
            show_progress="hidden",
        )

        view_today_button.click(
            fn=open_today_workspace,
            outputs=[main_tabs],
            show_progress="hidden",
        )

        refresh_today_button.click(
            fn=refresh_today,
            outputs=[
                today_table,
                today_summary,
            ],
        )

        refresh_timeline_button.click(
            fn=refresh_timeline,
            inputs=[
                timeline_date,
                timeline_event_type,
            ],
            outputs=[
                timeline_table,
                timeline_status,
            ],
        )

        healthos_outputs = [
            profile_summary,
            goals_table,
            checkin_summary,
            reminders_table,
            reminder_status,
        ]

        refresh_healthos_button.click(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )

        period_days.change(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )

        healthos_action_outputs = [
            main_tabs,
            chat_input,
            selected_record_state,
            selected_record_context,
        ]
        for action_button, action_function in (
            (edit_profile_button, open_profile_settings),
            (create_goal_button, open_goal_creation),
            (manage_goal_button, open_goal_management),
            (ask_review_button, open_period_review),
            (ask_knowledge_button, open_knowledge_question),
            (create_reminder_button, open_reminder_creation),
            (manage_reminder_button, open_reminder_management),
        ):
            action_button.click(
                fn=action_function,
                outputs=healthos_action_outputs,
                show_progress="hidden",
            )

        search_button.click(
            fn=search_candidates,
            inputs=[
                image_input,
                food_query,
            ],
            outputs=[
                selected_food,
                candidate_table,
                candidate_status,
                latest_retrieval_trace,
            ],
        )

        calculate_button.click(
            fn=calculate_meal_preview,
            inputs=[
                image_input,
                food_query,
                selected_food,
                grams_input,
            ],
            outputs=[
                meal_preview,
                meal_preview_state,
                calculation_status,
                recompute_evidence,
            ],
        )

        meal_save_event = meal_save_button.click(
            fn=confirm_meal_save,
            inputs=[
                meal_preview_state
            ],
            outputs=[
                meal_save_status,
                today_table,
                today_summary,
            ],
        )

        meal_save_event.then(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )

        meal_cancel_button.click(
            fn=cancel_meal_preview,
            outputs=[
                meal_preview_state,
                meal_save_status,
                meal_preview,
                recompute_evidence,
            ],
        )

        today_table.select(
            fn=open_today_record_in_chat,
            outputs=[
                main_tabs,
                chat_input,
                selected_record_state,
                selected_record_context,
            ],
            show_progress="hidden",
        )

        def bind_agent_turn(
            activity_event: Any,
        ) -> None:
            """让发送、回车和快捷语句复用同一条 Agent 链路。"""

            result_event = activity_event.then(
                fn=send_chat_message,
                inputs=[
                    chat_input,
                    chatbot,
                    selected_record_state,
                ],
                outputs=[
                    chatbot,
                    chat_input,
                    agent_status,
                    latest_agent_steps,
                    latest_agent_state,
                    pending_agent_card,
                    confirm_agent_button,
                    cancel_agent_button,
                    selected_record_state,
                    selected_record_context,
                ],
                show_progress="hidden",
            )

            result_event.then(
                fn=finish_agent_activity,
                inputs=[agent_status],
                outputs=[agent_activity],
                queue=False,
                show_progress="hidden",
            )

        send_activity_event = send_button.click(
            fn=begin_agent_activity,
            inputs=[
                chat_input,
                selected_record_state,
            ],
            outputs=[agent_activity],
            queue=False,
            show_progress="hidden",
        )
        bind_agent_turn(send_activity_event)

        submit_activity_event = chat_input.submit(
            fn=begin_agent_activity,
            inputs=[
                chat_input,
                selected_record_state,
            ],
            outputs=[agent_activity],
            queue=False,
            show_progress="hidden",
        )
        bind_agent_turn(submit_activity_event)

        for starter_button, starter_message in starter_buttons:
            starter_event = starter_button.click(
                fn=partial(
                    conversation_starter_message,
                    starter_message,
                ),
                outputs=[chat_input],
                show_progress="hidden",
            )
            starter_activity_event = starter_event.then(
                fn=begin_agent_activity,
                inputs=[
                    chat_input,
                    selected_record_state,
                ],
                outputs=[agent_activity],
                queue=False,
                show_progress="hidden",
            )
            bind_agent_turn(starter_activity_event)

        confirm_agent_event = confirm_agent_button.click(
            fn=confirm_agent_action,
            inputs=[
                chatbot
            ],
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
                today_table,
                today_summary,
                pending_agent_card,
                confirm_agent_button,
                cancel_agent_button,
            ],
            show_progress="hidden",
        )

        confirm_agent_event.then(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )

        cancel_agent_button.click(
            fn=cancel_agent_action,
            inputs=[
                chatbot
            ],
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
                pending_agent_card,
                confirm_agent_button,
                cancel_agent_button,
            ],
            show_progress="hidden",
        )

        reset_agent_button.click(
            fn=reset_agent_conversation,
            outputs=[
                chatbot,
                agent_status,
                latest_agent_steps,
                latest_agent_state,
                pending_agent_card,
                confirm_agent_button,
                cancel_agent_button,
                selected_record_state,
                selected_record_context,
            ],
        )

        refresh_agent_trace_button.click(
            fn=refresh_agent_traces,
            inputs=[
                agent_trace_limit
            ],
            outputs=[
                recent_agent_traces
            ],
        )

        demo.load(
            fn=refresh_today,
            outputs=[
                today_table,
                today_summary,
            ],
        )

        demo.load(
            fn=restore_agent_conversation,
            inputs=[
                browser_conversation_id
            ],
            outputs=[
                browser_conversation_id,
                chatbot,
                agent_status,
                pending_agent_card,
                confirm_agent_button,
                cancel_agent_button,
            ],
            show_progress="hidden",
        )

        demo.load(
            fn=refresh_agent_traces,
            outputs=[
                recent_agent_traces
            ],
        )

        demo.load(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )

        demo.unload(
            cleanup_agent_session
        )

    return demo


demo = build_demo()
