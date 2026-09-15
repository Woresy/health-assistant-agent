"""个人健康管理助理 Gradio 应用。"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from functools import partial
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ui.health_charts import goal_cards, trend_charts

import gradio as gr
from dotenv import load_dotenv

from src.automation.feishu import FeishuWebhookConfig, FeishuWebhookNotifier
from src.automation.reminder_scheduler import ReminderScheduler
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
from src.observability.langsmith import configure_langsmith
from src.health.models import (
    ExercisePayload,
    HealthEvent,
    MealPayload,
    WaterPayload,
    WeightPayload,
)
from src.healthos.memory_control import (
    clear_user_memories,
    delete_user_memory,
    export_user_memories,
    list_user_memories,
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
from src.tools.detect_food import detect_food
from src.vision.config import FoodDetectionConfig
from src.vision.factory import build_food_detector


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
  (() => {
    const classifyHealthOSNavigation = () => {
      const nav = document.querySelector('#main-tabs [role="tablist"]') ||
        document.querySelector('#main-tabs .tab-nav');
      if (!nav) return;
      const tabScope = nav.parentElement || nav;
      const tabs = Array.from(nav.querySelectorAll('button[role="tab"]'));
      const allButtons = Array.from(tabScope.querySelectorAll("button:not([tabindex='-1'])"));
      const resolveTab = (panelId, fallbackLabel) => {
        const panel = document.getElementById(panelId);
        const labelledBy = panel?.getAttribute("aria-labelledby");
        return (labelledBy && document.getElementById(labelledBy)) ||
          tabScope.querySelector(`[aria-controls="${panelId}"]`) ||
          tabs.find((tab) => tab.textContent.trim() === fallbackLabel) ||
          allButtons.find((tab) => tab.textContent.trim() === fallbackLabel);
      };
      const todaySummary = resolveTab("healthos-today", "今天");
      const today = resolveTab("healthos-record", "今日观察");
      const conversations = resolveTab("healthos-conversations", "对话");
      const trends = resolveTab("healthos-trends", "趋势与报告");
      const goals = resolveTab("healthos-goals", "目标与教练");
      const timeline = resolveTab("healthos-timeline", "健康时间线");
      const reminders = resolveTab("healthos-reminders", "提醒");
      const evidence = resolveTab("healthos-evidence", "运行证据");
      const privacy = resolveTab("healthos-privacy", "数据与隐私");
      [today, conversations, trends, goals, reminders].forEach((tab) => {
        if (tab) tab.dataset.healthosNav = "primary";
      });
      if (todaySummary) todaySummary.dataset.healthosNav = "support";
      if (conversations) conversations.dataset.healthosDesktop = "hidden";
      if (reminders) delete reminders.dataset.healthosDesktop;
      if (timeline) {
        timeline.dataset.healthosNav = "utility";
        timeline.dataset.healthosDesktop = "hidden";
      }
      if (evidence) evidence.dataset.healthosNav = "utility-start";
      // 「数据与隐私」统一从右上角「更多」进入（该菜单在所有宽度下都可见），
      // 侧边栏只保留日常记录会反复用到的页面。
      if (privacy) {
        privacy.dataset.healthosNav = "utility";
        privacy.dataset.healthosDesktop = "hidden";
      }
      if (!nav.dataset.healthosOrdered) {
        [today, conversations, trends, goals, reminders, evidence, timeline, privacy, todaySummary]
          .filter(Boolean)
          .forEach((tab) => nav.appendChild(tab));
        nav.dataset.healthosOrdered = "true";
      }
      if (today && !nav.querySelector('[data-healthos-group="daily"]')) {
        const label = document.createElement("span");
        label.dataset.healthosGroup = "daily";
        label.className = "healthos-nav-group";
        label.setAttribute("aria-hidden", "true");
        label.textContent = "日常工作";
        nav.insertBefore(label, today);
      }
      if (evidence && !nav.querySelector('[data-healthos-group="system"]')) {
        const label = document.createElement("span");
        label.dataset.healthosGroup = "system";
        label.className = "healthos-nav-group";
        label.setAttribute("aria-hidden", "true");
        label.textContent = "系统控制";
        nav.insertBefore(label, evidence);
      }
      if (timeline && !nav.querySelector('[data-healthos-group="tools"]')) {
        const label = document.createElement("span");
        label.dataset.healthosGroup = "tools";
        label.className = "healthos-nav-group";
        label.setAttribute("aria-hidden", "true");
        label.textContent = "记录工具";
        nav.insertBefore(label, timeline);
      }

      const pageNames = new Map([
        [today, "今日观察"], [conversations, "历史对话"], [trends, "趋势与报告"],
        [goals, "目标与教练"], [timeline, "健康时间线"],
        [todaySummary, "今日完整汇总"],
        [reminders, "提醒"],
        [evidence, "运行证据"], [privacy, "隐私与数据"],
      ]);
      pageNames.forEach((title, tab) => {
        if (!tab || tab.dataset.healthosTitleBound) return;
        tab.dataset.healthosTitleBound = "true";
        tab.addEventListener("click", () => {
          // 顶栏是 Markdown 组件，重渲染后旧引用会变成游离节点，
          // 所以每次点击都重新取一次，否则标题会一直停在首页。
          const heading = document.querySelector(".app-topbar > div h1");
          if (heading) heading.textContent = title;
        });
      });

      let toggle = document.querySelector(".mobile-tools-toggle");
      let menu = document.querySelector("#mobileToolsMenu");
      if (!toggle || !menu) {
        const topActions = document.querySelector(".app-topbar > p");
        // 顶栏还没渲染好时直接返回，避免把菜单挂到 body 上留下点不到的副本。
        if (!topActions) return;
        toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "mobile-tools-toggle";
        toggle.textContent = "更多";
        toggle.setAttribute("aria-expanded", "false");
        toggle.setAttribute("aria-controls", "mobileToolsMenu");
        menu = document.createElement("div");
        menu.id = "mobileToolsMenu";
        menu.className = "mobile-tools-menu";
        topActions.appendChild(toggle);
        topActions.appendChild(menu);
      }
      if (!document.documentElement.dataset.healthosMenuBound) {
        document.documentElement.dataset.healthosMenuBound = "true";
        document.addEventListener("click", (event) => {
          const activeToggle = event.target.closest?.(".mobile-tools-toggle");
          if (!activeToggle) return;
          const activeMenu = document.querySelector("#mobileToolsMenu");
          const open = activeMenu?.classList.toggle("open") || false;
          activeToggle.setAttribute("aria-expanded", String(open));
        });
      }
      if (!document.documentElement.dataset.healthosConversationNavBound) {
        document.documentElement.dataset.healthosConversationNavBound = "true";
        document.addEventListener("click", (event) => {
          const conversationItem = event.target.closest?.(
            "#sidebar-conversations label:has(input), #mobile-conversations label:has(input)"
          );
          if (!conversationItem || !today) return;
          window.setTimeout(() => today.click(), 0);
        });
      }
      const menuEntries = [
        ["今日完整汇总", todaySummary],
        ["健康时间线", timeline],
        ["数据与隐私", privacy],
      ];
      if (window.__healthosShowDeveloperUi) {
        menuEntries.splice(2, 0, ["运行证据", evidence]);
      }
      menuEntries.forEach(([key, tab]) => {
        if (menu.querySelector(`[data-healthos-tool="${key}"]`)) return;
        const item = document.createElement("button");
        item.type = "button";
        item.dataset.healthosTool = key;
        item.textContent = key;
        item.addEventListener("click", () => {
          if (tab) {
            tab.click();
          } else {
            const moreTabs = tabScope.querySelector('button[aria-label="More tabs"]');
            moreTabs?.click();
            window.setTimeout(() => {
              Array.from(tabScope.querySelectorAll("button"))
                .find((candidate) => candidate.textContent.trim() === key && candidate.offsetParent)
                ?.click();
            }, 0);
          }
          menu.classList.remove("open");
          toggle.setAttribute("aria-expanded", "false");
        });
        menu.appendChild(item);
      });
    };
    const observer = new MutationObserver(classifyHealthOSNavigation);
    const startNavigationEnhancements = () => {
      observer.observe(document.body, { childList: true, subtree: true });
      classifyHealthOSNavigation();
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", startNavigationEnhancements, { once: true });
    } else {
      startNavigationEnhancements();
    }
  })();
</script>
"""

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

LANGSMITH_STATUS = configure_langsmith()

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

SHOW_DEVELOPER_UI = os.getenv(
    "HEALTHOS_SHOW_DEVELOPER_UI",
    "false",
).strip().casefold() in {"1", "true", "yes", "on"}

# 「运行证据」页在普通用户模式下是隐藏 Tab，导航菜单必须同步隐藏，
# 否则菜单里会出现一个点了没有反应的入口。
APP_HEAD = APP_HEAD.replace(
    "<script>",
    "<script>window.__healthosShowDeveloperUi = "
    f"{'true' if SHOW_DEVELOPER_UI else 'false'};",
    1,
)


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

FEISHU_CONFIG = FeishuWebhookConfig.from_environment()
if FEISHU_CONFIG.available:
    try:
        reminder_poll_seconds = float(os.getenv("REMINDER_POLL_SECONDS", "30"))
    except ValueError:
        reminder_poll_seconds = 30.0
    reminder_scheduler: ReminderScheduler | None = ReminderScheduler(
        store=healthos_store,
        notifier=FeishuWebhookNotifier(FEISHU_CONFIG),
        event_store=event_store,
        poll_seconds=reminder_poll_seconds,
    )
    REMINDER_AUTOMATION_STATUS = (
        f"飞书自动发送已启用，目标：{FEISHU_CONFIG.destination_label}"
    )
else:
    reminder_scheduler = None
    REMINDER_AUTOMATION_STATUS = (
        "飞书自动发送未连接，提醒还发不出去；"
        "在 .env 设置 FEISHU_REMINDER_ENABLED=true 和 "
        "FEISHU_WEBHOOK_URL 后重启应用即可开启"
    )

DETECTION_CONFIG = FoodDetectionConfig.from_environment()
# 权重或 onnxruntime 缺失时是 None；识别只是预填增强项，缺它不挡任何事。
food_detector = build_food_detector(DETECTION_CONFIG)

if DETECTION_CONFIG.available:
    # 图片外发的说明放在「数据与隐私」页和 README，不占对话区的版面。
    MEAL_WORKFLOW_STEPS = "名称已经按图片填好，核对一下，再选个份量就行。"
else:
    MEAL_WORKFLOW_STEPS = (
        "填上名称和份量就行，热量会自动算。图片只用于你自己核对。"
    )

tool_router = HealthToolRouter(
    event_store,
    healthos_store=healthos_store,
    nutrition_repository=repository,
    food_detector=food_detector,
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
        f"编排器：{AGENT_ORCHESTRATOR}；"
        f"{LANGSMITH_STATUS.message}"
    )
except AgentConfigurationError as exc:
    agent_model = None
    AGENT_CONFIGURATION_ERROR = str(exc)
    AGENT_PROVIDER_STATUS = (
        "Agent 配置错误："
        f"{exc}"
    )
else:
    AGENT_CONFIGURATION_ERROR = ""


AGENT_SETUP_STEPS = (
    "AGENT_PROVIDER_MODE=openai_compatible、"
    "AGENT_API_KEY、AGENT_MODEL"
)


def _agent_unavailable_reason() -> str:
    """说明对话记录为什么用不了，并给出唯一的下一步。"""

    if AGENT_CONFIGURATION_ERROR:
        return (
            "对话记录还不能用：模型配置有错误——"
            f"{AGENT_CONFIGURATION_ERROR}。"
            "请修正项目根目录 .env 中的这一项后重启应用。"
        )
    return (
        "对话记录还不能用：还没有连接模型服务，"
        "所以我无法把你说的话整理成可确认的草稿。"
        "请在项目根目录的 .env 里填好 "
        f"{AGENT_SETUP_STEPS}，然后重启应用"
        "（没有 .env 时先执行 cp .env.example .env）。"
    )


def _setup_notice_html() -> str:
    """在用户开口之前，先集中说明唯一挡路的缺项和补齐方式。"""

    if AGENT_CONFIGURATION_ERROR:
        detail = (
            "模型配置有错误："
            f"{escape(AGENT_CONFIGURATION_ERROR)}。"
            "修正 .env 中的这一项后重启应用。"
        )
    else:
        detail = (
            "对话记录需要一个支持工具调用的模型服务，现在还没有连接。"
            "在项目根目录的 .env 里填好 "
            f"<code>{escape(AGENT_SETUP_STEPS)}</code>，"
            "然后重启应用；还没有 .env 时先执行 "
            "<code>cp .env.example .env</code>。"
        )

    return (
        '<section class="setup-notice-card">'
        "<strong>还差一步才能用对话记录</strong>"
        f"<p>{detail}</p>"
        "<p>在那之前，下面的快捷按钮和发送不会保存任何内容。"
        "现在就可以用的是：输入框下方的「添加图片」手动记一餐"
        "（不需要模型），以及「今天」「健康时间线」「趋势与报告」。</p>"
        "</section>"
    )


AGENT_UNAVAILABLE_ANSWER = (
    f"{_agent_unavailable_reason()}\n\n"
    "刚才这句话没有写入任何健康记录，"
    "输入框里的内容我保留着，配置好后可以直接重发。\n\n"
    "现在不配置也能做的事：用输入框下方的「添加图片」"
    "手动记一餐（这条路径不需要模型），"
    "以及查看「今天」「健康时间线」「趋势与报告」。"
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
    """生成不暴露内部错误码的用户提示。"""

    if error_code == "TIMEZONE_INVALID":
        return "暂时无法读取今天的记录，请稍后重试。"
    return f"暂时无法完成：{message}"


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
        is_check_in = preview.get("reminder_type") == "check_in"
        title = "确认启用主动问候" if is_check_in else "确认安排提醒"
        destination = str(preview.get("destination_label", "飞书"))
        recurrence = {
            "daily": "每天",
            "weekdays": "工作日",
        }.get(str(preview.get("recurrence", "once")), "单次")
        summary_html = (
            '<div class="confirmation-summary">'
            f"<strong>{'主动健康问候' if is_check_in else escape(str(preview.get('content', '健康提醒')))}</strong>"
            f"<span>发送到 {escape(destination)} · "
            f"{escape(str(preview.get('scheduled_for', '')))} · {escape(recurrence)}</span>"
            "</div>"
        )
        consequence = (
            "确认后才会启用；每次只根据已确认记录询问是否需要补记，不会自动写入。"
            if is_check_in
            else "确认后才会安排自动化任务；到期只尝试发送一次，重复确认不会重复创建。"
        )
    else:
        preview = data.get("preview", {})
        operation = {
            "update": "修改",
            "cancel": "取消",
            "snooze": "延后",
            "pause": "暂停",
            "resume": "恢复",
        }.get(str(preview.get("operation", "")), "修改")
        label = "提醒"
        title = f"确认{operation}提醒"
        after = preview.get("after", {}) if isinstance(preview, dict) else {}
        if operation == "修改" and isinstance(after, dict):
            try:
                scheduled_text = datetime.fromisoformat(
                    str(after.get("scheduled_for", ""))
                ).astimezone(_timezone()).strftime("%m月%d日 %H:%M")
            except (ValueError, TypeError):
                scheduled_text = "时间待确认"
            summary_html = (
                '<div class="confirmation-summary">'
                f"<strong>{escape(str(after.get('content', '健康提醒')))}</strong>"
                f"<span>{escape(scheduled_text)} · 发送到飞书</span>"
                "</div>"
            )
        else:
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
        "<span>请检查；如有误，直接在输入框里说明要改成什么</span>"
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
    if pending.action == "reminder_create":
        preview = pending.draft_data.get("preview", {})
        if isinstance(preview, dict) and preview.get("reminder_type") == "check_in":
            confirm_label = "确认启用"

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


def _conversation_choices() -> list[tuple[str, str]]:
    """生成使用真实持久化时间和首条用户消息的会话选项。"""

    now = datetime.now().astimezone()
    choices: list[tuple[str, str]] = []
    for item in conversation_store.list_summaries(LOCAL_USER_ID, limit=30):
        local_time = item.updated_at.astimezone(now.tzinfo)
        when = (
            f"今天 · {local_time:%H:%M}"
            if local_time.date() == now.date()
            else f"{local_time.month} 月 {local_time.day} 日"
        )
        choices.append((f"{item.title}\n{when}", item.session_id))
    return choices


def refresh_conversation_picker(stored_value: Any) -> Any:
    """刷新历史会话列表并保持当前选择。"""

    browser_id = _conversation_id(stored_value)
    active_id = f"conversation-{browser_id}"
    choices = _conversation_choices()
    values = {value for _, value in choices}
    return gr.update(
        choices=choices,
        value=active_id if active_id in values else None,
    )


def refresh_conversation_pickers(stored_value: Any) -> tuple[Any, Any]:
    """同步桌面侧栏与移动历史页的会话选项。"""

    return (
        refresh_conversation_picker(stored_value),
        refresh_conversation_picker(stored_value),
    )


def create_agent_conversation(request: gr.Request) -> tuple[Any, ...]:
    """创建新的稳定会话，不删除或覆盖历史会话。"""

    browser_id, session_key = _bind_conversation(uuid4().hex, request)
    with _AGENT_SESSIONS_LOCK:
        _AGENT_SESSIONS.pop(session_key, None)
    return (
        browser_id,
        _welcome_chat_history(),
        "新对话已准备好，可以开始记录。",
        *_confirmation_updates(None),
        {},
        gr.Markdown(value="", visible=False, sanitize_html=False, container=False),
        gr.update(choices=_conversation_choices(), value=None),
        gr.update(choices=_conversation_choices(), value=None),
    )


def switch_agent_conversation(
    selected_session_id: str | None,
    request: gr.Request,
) -> tuple[Any, ...]:
    """切换到属于当前用户的历史会话并恢复完整上下文。"""

    session_id = str(selected_session_id or "").strip()
    try:
        state = conversation_store.load(session_id)
    except (OSError, ValueError):
        state = None
    if state is None or state.user_id != LOCAL_USER_ID:
        return (
            gr.skip(),
            gr.skip(),
            "无法打开这段对话，请刷新列表后重试。",
            *_confirmation_updates(None),
            gr.skip(),
            gr.skip(),
        )

    browser_id, _ = _bind_conversation(
        session_id.removeprefix("conversation-"),
        request,
    )
    session = _get_agent_session(request)
    if session is None:
        return (
            browser_id,
            _welcome_chat_history(),
            AGENT_PROVIDER_STATUS,
            *_confirmation_updates(None),
            session_id,
            session_id,
        )
    return (
        browser_id,
        _restored_chat_history(session),
        f"已打开历史对话，共 {len(_restored_chat_history(session))} 条消息。",
        *_confirmation_updates(session.state.pending_confirmation),
        session_id,
        session_id,
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
        (
            '<li class="done"><span></span><div><strong>'
            if index == 0
            else (
                '<li class="active"><span></span><div><strong>'
                if index == 1
                else "<li><span></span><div><strong>"
            )
        )
        + escape(step)
        + (
            "</strong><small>已完成</small></div></li>"
            if index == 0
            else (
                "</strong><small>处理中</small></div></li>"
                if index == 1
                else "</strong><small>等待</small></div></li>"
            )
        )
        for index, step in enumerate(steps)
    )

    return gr.Markdown(
        value=(
            '<section class="agent-process active" '
            'role="status" aria-live="polite">'
            '<header><span class="process-pulse" aria-hidden="true"></span>'
            "<div><strong>小满正在处理</strong>"
            "<small>完成后会告诉你结果，或请你确认下一步</small></div></header>"
            f"<ol>{items}</ol>"
            "</section>"
        ),
        visible=True,
        sanitize_html=False,
        container=False,
    )


def stage_chat_message(
    user_text: str,
    history: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str, str]:
    """立即回显用户消息，并为后台处理保留原始文本。"""

    chat_history = list(history or [])
    normalized_text = user_text.strip() if isinstance(user_text, str) else ""
    if not normalized_text:
        return chat_history, "", ""

    chat_history.append(
        {
            "role": "user",
            "content": normalized_text,
        }
    )
    return chat_history, "", normalized_text


def _visible_tool_step(step: dict[str, Any]) -> tuple[str, str]:
    """把脱敏工具证据转换为用户可理解的处理步骤。"""

    tool_name = str(step.get("tool", ""))
    action = {
        "retrieve_nutrition_candidates": "查找匹配的食物数据",
        "calculate_nutrition": "计算结构化营养数据",
        "retrieve_health_knowledge": "查阅可信健康知识",
        "query_health_events": "读取已确认的健康记录",
        "get_daily_summary": "汇总当天健康事实",
        "get_period_review": "整理近期变化",
        "prepare_health_event": "整理待确认的健康记录",
        "prepare_update_health_event": "整理待确认的记录修改",
        "prepare_delete_health_event": "核对待删除的健康记录",
        "prepare_profile_update": "整理待确认的个人设置",
        "prepare_goal_change": "整理待确认的目标变更",
        "create_reminder_draft": "整理待确认的提醒",
    }.get(tool_name, "调用受控健康能力")
    status = str(step.get("status", "已执行"))
    return action, status


def finish_agent_activity(
    status_text: str,
    tool_steps: list[dict[str, Any]] | None,
    agent_state: dict[str, Any] | None = None,
) -> Any:
    """保留本轮可核验的处理摘要，同时隐藏模型内部推理。"""

    normalized_status = (
        status_text.strip()
        if isinstance(status_text, str)
        else "本轮处理已结束。"
    )

    turn_state = (
        str(agent_state.get("state", ""))
        if isinstance(agent_state, dict)
        else ""
    )
    executed = turn_state not in {
        "provider_disabled",
        "provider_error",
    }

    visible_steps = [
        (
            "理解你的请求与当前对话",
            "已完成" if executed else "未执行",
        ),
    ]
    visible_steps.extend(
        _visible_tool_step(step)
        for step in (tool_steps or [])
        if isinstance(step, dict)
    )
    visible_steps.append(("整理回答与下一步", normalized_status))
    items = "".join(
        "<li><span></span><div>"
        f"<strong>{escape(title)}</strong>"
        f"<small>{escape(detail)}</small>"
        "</div></li>"
        for title, detail in visible_steps
    )

    return gr.Markdown(
        value=(
            '<details class="agent-process complete">'
            "<summary>"
            '<span class="process-mark" aria-hidden="true"></span>'
            "<div><strong>本次处理</strong>"
            f"<small>{'已完成' if executed else '没有执行，也没有写入'}"
            f" · {len(visible_steps)} 个步骤</small></div>"
            "<b>查看</b>"
            "</summary>"
            f"<ol>{items}</ol>"
            "</details>"
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
    calorie_target = goal_by_type.get("nutrition", {}).get("target_value")
    water_target = goal_by_type.get("water", {}).get("target_value")
    exercise_target = goal_by_type.get("exercise", {}).get("target_value")
    calorie_target = float(calorie_target) if calorie_target is not None else None
    water_target = float(water_target) if water_target is not None else None
    exercise_target = float(exercise_target) if exercise_target is not None else None

    calorie_progress = min(
        calories / calorie_target * 100,
        100,
    ) if calorie_target is not None and calorie_target > 0 else 0

    water_progress = min(
        water_ml / water_target * 100,
        100,
    ) if water_target is not None and water_target > 0 else 0

    exercise_progress = min(
        exercise_minutes / exercise_target * 100,
        100,
    ) if exercise_target is not None and exercise_target > 0 else 0

    weight_progress = (
        100
        if weight["latest_weight_kg"]
        is not None
        else 0
    )

    if int(summary.get("event_count", 0)) == 0:
        decision_title = "先留下今天的第一条记录"
        decision_detail = "可以从刚喝的水、最近一餐或一次运动开始。"
    elif water_target is not None and water_target > 0 and water_ml < water_target:
        remaining_water = max(water_target - water_ml, 0)
        decision_title = f"饮水目标还差 {remaining_water:.0f} ml"
        decision_detail = "这是根据今天已确认的饮水记录和当前目标计算的差值。"
    elif exercise_target is not None and exercise_target > 0 and exercise_minutes < exercise_target:
        remaining_minutes = max(exercise_target - exercise_minutes, 0)
        decision_title = f"运动目标还差 {remaining_minutes:.0f} 分钟"
        decision_detail = "如果今天不便运动，也可以只记录真实情况，不需要补齐数字。"
    else:
        decision_title = f"今天已有 {int(summary.get('event_count', 0))} 条确认记录"
        decision_detail = "当前没有可计算的目标差值；继续按真实情况记录即可。"

    water_goal_label = (
        f"目标 {water_target:g} ml"
        if water_target is not None
        else "尚未设置目标"
    )
    exercise_goal_label = (
        f"目标 {exercise_target:g} 分钟"
        if exercise_target is not None
        else "尚未设置目标"
    )

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
        f'<small>{water_goal_label}</small>'
        '</article>'
        '<article class="health-metric" '
        'style="--metric-color:#2f765e;'
        f'--metric-progress:{exercise_progress:.1f}%">'
        '<label>运动时长</label>'
        f'<strong>{exercise_minutes:.0f}</strong><small>分钟</small>'
        '<div class="metric-track"><i></i></div>'
        f'<small>{exercise_goal_label} · {exercise["total_distance_km"]:.1f} km</small>'
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


def _margin_summary_markdown(
    summary: dict[str, Any],
    goal_gaps: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> str:
    """把今日真实汇总压缩为桌面页边笔记。"""

    meal = summary["meal"]
    water = summary["water"]
    exercise = summary["exercise"]
    weight = summary["weight"]
    goal_by_unit = {
        str(item.get("unit", "")).lower(): item
        for item in (goal_gaps or [])
    }
    water_goal = next(
        (
            item
            for unit, item in goal_by_unit.items()
            if unit in {"ml", "毫升"}
        ),
        None,
    )
    water_target = float(water_goal["target_value"]) if water_goal is not None else None
    water_text = (
        f'{float(water["total_ml"]):.0f} / {water_target:.0f} ml'
        if water_target is not None
        else f'{float(water["total_ml"]):.0f} ml · 未设置目标'
    )
    latest_weight = weight.get("latest_weight_kg")
    weight_text = (
        f"{float(latest_weight):g} kg"
        if latest_weight is not None
        else "暂无记录"
    )
    event_count = int(summary.get("event_count", 0))
    source_count = int(meal.get("count", 0))
    record_items: list[str] = []
    for raw_event in (events or [])[-8:][::-1]:
        try:
            event = HealthEvent.model_validate(raw_event)
        except (ValueError, TypeError):
            continue
        local_time = event.occurred_at.astimezone(_timezone()).strftime("%H:%M")
        record_items.append(
            '<li class="margin-record">'
            f'<time>{escape(local_time)}</time>'
            '<span>'
            f'<b>{escape(_event_type_label(event))}</b>'
            f'<small>{escape(_event_detail(event))}</small>'
            '</span>'
            '</li>'
        )
    records_html = (
        '<ol class="margin-records">' + "".join(record_items) + "</ol>"
        if record_items
        else '<p class="margin-empty">今天还没有已确认记录。</p>'
    )

    return (
        '<div class="margin-head">'
        '<div><span>今日记录</span><strong>今日已确认记录</strong></div>'
        f'<small>{escape(str(summary["summary_date"]))}</small>'
        '</div>'
        f'{records_html}'
        '<section class="margin-note margin-note-lead">'
        '<span>今日饮食热量</span>'
        f'<strong>{float(meal["calories_kcal"]):.0f} '
        '<small>kcal 估算</small></strong>'
        f'<p>{source_count} 顿餐食具有可追溯来源；草稿不会计入。</p>'
        '</section>'
        '<dl class="margin-facts">'
        '<div><dt>饮水</dt>'
        f'<dd>{water_text}</dd></div>'
        '<div><dt>运动</dt>'
        f'<dd>{float(exercise["total_duration_minutes"]):.0f} 分钟</dd></div>'
        '<div><dt>最近体重</dt>'
        f'<dd>{escape(weight_text)}</dd></div>'
        '</dl>'
        '<section class="margin-note">'
        '<span>记录状态</span>'
        f'<strong>{event_count} 条记录已写入</strong>'
        '<p>待确认内容只保留在当前对话，确认后这里会同步更新。</p>'
        '</section>'
        '<section class="margin-boundary">'
        '<strong>你的数据由你决定</strong>'
        '<p>业务数据保存在本机。每次写入、修改和删除都需要确认。</p>'
        '</section>'
    )


def refresh_today_margin() -> str:
    """读取右侧页边笔记所需的今日真实数据。"""

    try:
        today = _today_string()
    except ValueError as exc:
        return _error_text("TIMEZONE_INVALID", str(exc))

    result = get_healthos_daily_summary(
        user_id=LOCAL_USER_ID,
        date=today,
        timezone_name=APP_TIMEZONE,
        store=event_store,
        healthos_store=healthos_store,
    )
    if not result["ok"]:
        error = result["error"]
        return _error_text(error["error_code"], error["message"])

    data = result["data"]
    return _margin_summary_markdown(
        data["summary"],
        data.get("goal_gaps", []),
        data.get("events", []),
    )


def refresh_trends(period_days: int = 7) -> tuple[str, str]:
    """用已确认的每日汇总生成可视化，保留缺失日期。"""

    try:
        days = int(period_days)
    except (TypeError, ValueError):
        days = 7
    if days not in {7, 14, 30}:
        days = 7

    try:
        active_timezone = _timezone()
    except ValueError as exc:
        return "", _error_text("TIMEZONE_INVALID", str(exc))

    final_date = datetime.now(active_timezone).date()
    samples: list[tuple[str, dict[str, Any]]] = []
    total_events = 0
    days_with_data = 0

    for offset in range(days):
        selected_date = final_date - timedelta(days=offset)
        result = get_healthos_daily_summary(
            user_id=LOCAL_USER_ID,
            date=selected_date.isoformat(),
            timezone_name=APP_TIMEZONE,
            store=event_store,
            healthos_store=healthos_store,
        )
        if not result["ok"]:
            error = result["error"]
            return "", _error_text(error["error_code"], error["message"])

        summary = result["data"]["summary"]
        event_count = int(summary["event_count"])
        total_events += event_count
        days_with_data += int(event_count > 0)
        samples.append((selected_date.strftime("%m/%d"), summary))

    return (
        trend_charts(list(reversed(samples))),
        (
            f"最近 {days} 天共有 {total_events} 条已确认记录，"
            f"其中 {days_with_data} 天有数据。未记录项不会按零计算。"
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
    "unknown": "发送结果待核实",
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
    if not profile.get("reminders_enabled", True):
        reminder_text = "关闭"
    elif FEISHU_CONFIG.available:
        reminder_text = "开启"
    else:
        reminder_text = "开启（发送通道未连接）"
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


def _reminder_cards(reminders: list[dict[str, Any]]) -> str:
    """以可换行的飞书提醒列表替代宽表格。"""

    visible_reminders = [
        reminder
        for reminder in reminders
        if reminder.get("delivery_channel") == "feishu"
    ]
    if not visible_reminders:
        return (
            '<section class="reminder-list reminder-list-empty">'
            '<strong>还没有飞书提醒</strong>'
            '<p>创建后会先显示草稿，只有确认后才会安排。</p>'
            '</section>'
        )
    cards: list[str] = []
    for reminder in visible_reminders:
        scheduled = str(reminder.get("scheduled_for", ""))
        try:
            scheduled_text = datetime.fromisoformat(scheduled).astimezone(_timezone()).strftime("%m月%d日 %H:%M")
        except (ValueError, TypeError):
            scheduled_text = scheduled or "时间未知"
        title = (
            "主动健康问候"
            if reminder.get("reminder_type") == "check_in"
            else str(reminder.get("content", "健康提醒"))
        )
        recurrence_label = {
            "once": "单次",
            "daily": "每天",
            "weekdays": "工作日",
        }.get(str(reminder.get("recurrence", "once")), "单次")
        status_label = _REMINDER_STATUS_LABELS.get(
            str(reminder.get("status", "")), "状态未知"
        )
        destination = str(reminder.get("destination_label", "飞书"))
        cards.append(
            '<article class="reminder-item">'
            '<div class="reminder-item-main">'
            f'<strong>{escape(title)}</strong>'
            f'<span>{escape(scheduled_text)} · {escape(recurrence_label)}</span>'
            f'<small>发送到 {escape(destination)}</small>'
            '</div>'
            f'<b class="reminder-state">{escape(status_label)}</b>'
            '</article>'
        )
    return '<section class="reminder-list">' + "".join(cards) + "</section>"


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
    """刷新档案、目标、周期复盘和飞书提醒。"""

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
        return status, "", status, "", status

    profile = profile_result["data"]["profile"]
    goals = goals_result["data"]["goals"]
    period = period_result["data"]
    reminders = reminders_result["data"]["reminders"]
    return (
        _profile_markdown(profile),
        goal_cards(goals, period),
        _checkin_markdown(period, goals),
        _reminder_cards(reminders),
        f"已读取 {sum(item.get('delivery_channel') == 'feishu' for item in reminders)} 条飞书提醒；"
        "修改、取消、延后、暂停和恢复都需要确认。"
        f" {REMINDER_AUTOMATION_STATUS}。",
    )


def refresh_memory_center() -> tuple[Any, Any, str]:
    """读取用户可理解的长期记忆，不向表格暴露内部 ID。"""

    result = list_user_memories(user_id=LOCAL_USER_ID, store=healthos_store)
    if not result["ok"]:
        error = result["error"]
        return [], gr.Dropdown(choices=[], value=None), _error_text(error["error_code"], error["message"])
    memories = result["data"]["memories"]
    rows = [
        [item["label"], item["content"], str(item["confirmed_at"])[:16].replace("T", " ")]
        for item in memories
    ]
    choices = [
        (f"{item['label']} · {item['content']}", item["memory_id"])
        for item in memories
    ]
    status = (
        f"已保存 {len(memories)} 条经过确认的长期记忆。"
        if memories
        else "目前没有长期记忆。健康记录和目标不会因为这里为空而被删除。"
    )
    return rows, gr.Dropdown(choices=choices, value=None), status


def prepare_memory_delete(memory_id: str | None) -> tuple[dict[str, str], Any, Any, Any]:
    if not memory_id:
        return {}, gr.Markdown(value="请先选择要遗忘的内容。", visible=True), gr.Button(visible=False), gr.Button(visible=False)
    result = list_user_memories(user_id=LOCAL_USER_ID, store=healthos_store)
    memory = next(
        (item for item in result.get("data", {}).get("memories", []) if item["memory_id"] == memory_id),
        None,
    )
    if memory is None:
        return {}, gr.Markdown(value="这条记忆已经不存在，请刷新后重试。", visible=True), gr.Button(visible=False), gr.Button(visible=False)
    preview = (
        '<section class="memory-confirmation" role="status" aria-live="polite">'
        '<strong>确认遗忘这条内容？</strong>'
        f'<p>{escape(memory["label"])}：{escape(memory["content"])}</p>'
        '<small>确认后，下一轮对话不会再把这项偏好加入上下文；健康记录和目标不受影响。</small>'
        '</section>'
    )
    return (
        {"action": "delete", "memory_id": memory_id},
        gr.Markdown(value=preview, visible=True, sanitize_html=False, container=False),
        gr.Button(value="确认遗忘", visible=True),
        gr.Button(visible=True),
    )


def prepare_memory_clear() -> tuple[dict[str, str], Any, Any, Any]:
    preview = (
        '<section class="memory-confirmation memory-confirmation-danger" role="alert">'
        '<strong>确认清除全部长期记忆？</strong>'
        '<p>教练风格、饮食偏好、忌口和提醒偏好会恢复默认。</p>'
        '<small>健康事实、目标、提醒任务和对话历史不会被删除。</small>'
        '</section>'
    )
    return (
        {"action": "clear"},
        gr.Markdown(value=preview, visible=True, sanitize_html=False, container=False),
        gr.Button(value="确认全部清除", visible=True),
        gr.Button(visible=True),
    )


def cancel_memory_action() -> tuple[dict[str, str], Any, Any, Any]:
    return {}, gr.Markdown(value="", visible=False), gr.Button(visible=False), gr.Button(visible=False)


def confirm_memory_action(action_state: dict[str, str]) -> tuple[Any, Any, str, dict[str, str], Any, Any, Any]:
    action = action_state.get("action") if isinstance(action_state, dict) else None
    if action == "delete":
        result = delete_user_memory(
            user_id=LOCAL_USER_ID,
            memory_id=action_state.get("memory_id", ""),
            confirmed=True,
            store=healthos_store,
        )
        success = "已遗忘所选内容，下一轮对话将使用更新后的偏好。"
    elif action == "clear":
        result = clear_user_memories(user_id=LOCAL_USER_ID, confirmed=True, store=healthos_store)
        success = "长期记忆已全部清除；健康记录、目标和对话历史仍然保留。"
    else:
        result = {"ok": False, "error": {"error_code": "MEMORY_ACTION_MISSING", "message": "没有等待确认的记忆操作"}}
        success = ""
    rows, selector, status = refresh_memory_center()
    if result["ok"]:
        status = success
    else:
        error = result["error"]
        status = _error_text(error["error_code"], error["message"])
    return rows, selector, status, {}, gr.Markdown(value="", visible=False), gr.Button(visible=False), gr.Button(visible=False)


def export_memory_file() -> tuple[str | None, str]:
    result = export_user_memories(user_id=LOCAL_USER_ID, store=healthos_store)
    if not result["ok"]:
        error = result["error"]
        return None, _error_text(error["error_code"], error["message"])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="healthos-memories-", suffix=".json", delete=False
    ) as target:
        json.dump(result["data"], target, ensure_ascii=False, indent=2)
        target.write("\n")
        path = target.name
    return path, "记忆导出文件已生成。文件不包含健康事件、API Key 或确认令牌。"


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


REMINDER_SETUP_HINT = (
    '<div class="setup-notice-card">'
    "<strong>提醒还发不出去</strong>"
    "<p>提醒只通过飞书发送，现在还没有连接发送通道。"
    "在项目根目录的 .env 里设置 "
    "<code>FEISHU_REMINDER_ENABLED=true</code> 和 "
    "<code>FEISHU_WEBHOOK_URL</code>（群机器人 Webhook 地址），"
    "重启应用后再回到这里。</p>"
    "<p>在那之前你仍然可以正常记录、查看今天和时间线；"
    "下面写好的内容不会丢，连接后直接发送即可。</p>"
    "</div>"
)


def _open_reminder_action(prompt: str) -> tuple[Any, str, dict[str, Any], Any]:
    """提醒相关入口先说明发送通道是否可用，再进入确认链路。"""

    tabs, text, state, context = open_healthos_action(prompt)
    if FEISHU_CONFIG.available:
        return tabs, text, state, context
    return (
        tabs,
        text,
        state,
        gr.Markdown(
            value=REMINDER_SETUP_HINT,
            visible=True,
            sanitize_html=False,
            container=False,
        ),
    )


def open_reminder_creation() -> tuple[Any, str, dict[str, Any], Any]:
    return _open_reminder_action("我想创建一个飞书提醒：")


def open_active_check_in() -> tuple[Any, str, dict[str, Any], Any]:
    """用可编辑自然语言进入确认链路，默认值不会直接启用任务。"""

    return _open_reminder_action(
        "我想每天晚上 9 点通过飞书主动问我，关注饮食、饮水和运动"
    )


def open_reminder_management() -> tuple[Any, str, dict[str, Any], Any]:
    return _open_reminder_action("请列出我的飞书提醒，我想修改、延后、暂停、恢复或取消其中一个。")


def _tool_contract_rows() -> list[list[str]]:
    """为开发者证据页展示 16 个工具的安全契约。"""

    boundaries = {
        "get_user_profile": "最小字段",
        "prepare_profile_update": "确认后写入",
        "get_health_goals": "含版本历史",
        "prepare_goal_change": "确认后追加版本",
        "get_health_events": "仅 committed 事实",
        "prepare_health_event": "缺参追问",
        "prepare_event_change": "展示前后对比",
        "detect_food": "只预填候选词",
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
        "inference": "推理",
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


def open_coach_style_change(style: str) -> tuple[str, Any]:
    """把顶栏表达风格选择转成现有确认式对话流程。"""

    label = {
        "gentle": "温和陪伴",
        "rational": "理性观察",
        "concise": "简洁行动",
        "goal_focused": "目标督促",
    }.get(str(style), "理性观察")
    return f"请把小满的表达风格调整为{label}", gr.Tabs(selected="chat")


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

    for candidate in data["candidates"]:
        choices.append(
            (
                f"{candidate['name']} · "
                f"{candidate['category']}",
                candidate["food_id"],
            )
        )

    selected_value = (
        data["candidates"][0]["food_id"]
        if data["auto_select_allowed"]
        else None
    )

    status = (
        (
            "已为你选中最接近的食物，请确认是否正确。"
            if selected_value
            else "找到了几种相近的食物，请选择最符合图片的一项。"
        )
    )

    return (
        gr.Dropdown(
            choices=choices,
            value=selected_value,
        ),
        status,
        trace,
    )


def refresh_meal_panel(
    image_path: str | None,
    food_query: str,
    selected_food_id: str | None,
    raw_grams: Any,
    detected_query: str = "",
) -> tuple[
    gr.Radio,
    str,
    str,
    dict[str, Any] | None,
    dict[str, Any],
]:
    """名称、候选或份量一变就重新匹配并试算。

    用户不需要点「匹配食物」「查看营养估算」——那两步是系统的内部动作，
    不是用户要做的决定。用户只需要决定两件事：吃的是什么、吃了多少。
    """

    blank_radio = gr.Radio(choices=[], value=None, visible=False)

    if not str(food_query or "").strip():
        return (
            blank_radio,
            "",
            "",
            None,
            {},
        )

    image_result = validate_image(image_path)
    if not image_result.ok:
        return (
            blank_radio,
            _error_text(
                image_result.error_code or "IMAGE_INVALID",
                image_result.message,
            ),
            "",
            None,
            {},
        )

    tool_result = retrieve_nutrition_candidates(
        query=food_query,
        top_k=5,
        repository=repository,
    )

    if not tool_result["ok"]:
        error = tool_result["error"]
        return (
            blank_radio,
            _error_text(
                error["error_code"],
                error["message"],
            ),
            "",
            None,
            {},
        )

    data = tool_result["data"]
    trace = data["trace"]
    candidates = data["candidates"]

    if data["status"] == "not_found" or not candidates:
        return (
            blank_radio,
            (
                f"食物库里没有「{food_query.strip()}」。"
                "换个更常见的说法试试，例如「米饭」「鸡蛋」；"
                "查不到就不会给出热量，不会替你猜。"
            ),
            "",
            None,
            trace,
        )

    choices = _candidate_choices(candidates)
    candidate_ids = {item["food_id"] for item in candidates}

    # 用户已经选过的优先保留，否则精确命中时替他预选。
    if selected_food_id in candidate_ids:
        active_id = selected_food_id
    else:
        active_id = _exact_match_food_id(candidates)

    matched = next(
        (c for c in candidates if c["food_id"] == active_id),
        None,
    )

    if matched is None:
        status = "这个名字对应好几种食物，选一个你吃的："
    elif len(candidates) == 1:
        status = f"用的是食物库里的「{matched['name']}」。"
    else:
        status = (
            f"用的是食物库里的「{matched['name']}」，"
            "不对的话在下面换一个。"
        )

    radio = gr.Radio(
        choices=choices,
        value=active_id,
        visible=True,
        label=None,
    )

    if active_id is None or _grams_is_blank(raw_grams):
        hint = (
            "选好食物后填份量，热量会自动算出来。"
            if active_id is None
            else "填上份量就能看到热量。"
        )
        return (radio, status, hint, None, trace)

    summary, preview_state, _, _ = calculate_meal_preview(
        image_path,
        food_query,
        active_id,
        raw_grams,
        detected_query,
    )

    return (radio, status, summary, preview_state, trace)


def use_portion_preset(grams: int) -> str:
    """份量快捷按钮只是替用户打字。"""

    return str(grams)


def prefill_candidates_after_detection(
    image_path: str | None,
    detected_query: str,
) -> tuple[
    gr.Dropdown,
    str,
    dict[str, Any],
]:
    """检测出名称后顺带预填候选下拉；没识别出来就保持空白。"""

    if not detected_query.strip():
        return (
            gr.Dropdown(choices=[], value=None),
            "",
            {},
        )

    return search_candidates(image_path, detected_query)


def _format_detection_status(
    data: dict[str, Any],
) -> tuple[str, str]:
    """把检测结果转成界面文案和预填检索词。"""

    detections = data.get("detections") or []

    if not detections:
        if DETECTION_CONFIG.mode == "onnx":
            return (
                "没有识别出可靠的食物。"
                "本地权重只认得苹果、香蕉、橙、西兰花、胡萝卜、"
                "披萨、三明治、热狗、甜甜圈和蛋糕，"
                "中餐菜品识别不到。请直接填写食物名称和份量。",
                "",
            )

        return (
            "没有识别出可靠的食物，请直接填写食物名称和份量。",
            "",
        )

    top = detections[0]
    confidence_text = f"{float(top['confidence']) * 100:.0f}%"
    # 同一类别可能有多个检测框（三个番茄 = 三个 apple），
    # 这里按名称去重，否则文案会变成"已识别出苹果……还检测到：苹果"。
    others: list[str] = []
    for detection in detections[1:]:
        name = str(detection["label_zh"])
        if name != top["label_zh"] and name not in others:
            others.append(name)

    status = (
        f"已识别出**{top['label_zh']}**"
        f"（置信度 {confidence_text}），"
        "名称已经替你填好，请核对后填写份量。"
    )

    if others:
        status += (
            "图中还检测到：" + "、".join(others) + "。"
            "如果要记的是其中一项，直接改写上面的名称即可。"
        )

    status += "识别只负责填名称，营养值仍然按你选中的食物数据和份量计算。"

    return status, str(top["suggested_query"])


def open_meal_image_workflow(
    image_path: str | None,
) -> tuple[gr.Column, str | None, str, str, str]:
    """展开餐食核对流程，并尽力用检测结果预填食物名称。

    检测不可用、失败或识别不到时一律回退到手动填写，
    不显示任何识别结论，也不阻断后面的检索、计算和保存。
    """

    workflow = gr.Column(visible=bool(image_path))

    if not image_path:
        return (workflow, image_path, "", "", "")

    if not DETECTION_CONFIG.available:
        return (
            workflow,
            image_path,
            "",
            DETECTION_CONFIG.unavailable_reason() or "",
            "",
        )

    tool_result = detect_food(
        image_path=image_path,
        detector=food_detector,
        config=DETECTION_CONFIG,
    )

    if not tool_result["ok"]:
        error = tool_result["error"]

        return (
            workflow,
            image_path,
            "",
            _error_text(
                error["error_code"],
                error["message"] + "；请直接手动填写食物名称和份量。",
            ),
            "",
        )

    status, suggested_query = _format_detection_status(
        tool_result["data"]
    )

    return (
        workflow,
        image_path,
        suggested_query,
        status,
        suggested_query,
    )


# 界面提供的常用份量，只是替用户省去打字，不是对某种食物的份量断言。
PORTION_PRESETS: tuple[int, ...] = (50, 100, 150, 200, 300)

# 检索层的 auto_select_allowed 恒为 False，含义是"检索不替用户确认"。
# 界面层可以在**精确命中**时预选一个候选，但必须满足两件事：
#   1. 用的是哪条食物数据全程显示在卡片上，可一键更换；
#   2. 仍然要用户点「保存这一餐」才写入。
# 所以"同意"从"点下拉框"变成了"点保存"，没有任何一步是静默的。
EXACT_MATCH_TYPES = frozenset({"standard_name", "alias"})


def _exact_match_food_id(
    candidates: list[dict[str, Any]],
) -> str | None:
    """命中标准名或人工维护的别名时，返回可以预选的 food_id。

    包含匹配和模糊匹配一律返回 None——那些情况系统确实不确定，
    必须让用户自己挑。
    """

    if not candidates:
        return None

    top = candidates[0]

    if top.get("match_type") not in EXACT_MATCH_TYPES:
        return None

    # 第一名和第二名都算精确命中时，说明有歧义（例如"橙"和"柑橘"），
    # 不替用户决定。
    if len(candidates) > 1 and candidates[1].get(
        "match_type"
    ) in EXACT_MATCH_TYPES:
        return None

    return str(top["food_id"])


def _candidate_choices(
    candidates: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """候选的可读标签。"""

    return [
        (
            f"{candidate['name']} · {candidate['category']}",
            candidate["food_id"],
        )
        for candidate in candidates
    ]


def _grams_is_blank(raw_grams: Any) -> bool:
    """份量是否根本没填。

    Gradio 的 Number 在清空时给 None，在某些交互后给 0；
    两者对用户都是"还没填"，不该当成非法数值报错。
    """

    if raw_grams is None:
        return True

    if isinstance(raw_grams, str) and not raw_grams.strip():
        return True

    try:
        return float(raw_grams) == 0
    except (TypeError, ValueError):
        return False


def _missing_input_text(missing: list[str]) -> str:
    """告诉用户还差哪几格，而不是抛一句校验失败。"""

    tail = (
        "营养值只按你选中的那条食物数据和你填的份量换算，不会替你猜。"
    )

    if len(missing) == 1:
        return (
            f"还差一步：请先{missing[0]}，"
            f"再点「查看营养估算」。{tail}"
        )

    steps = "；".join(
        f"{index}. {item}"
        for index, item in enumerate(missing, start=1)
    )

    return (
        f"还差 {len(missing)} 项：{steps}。"
        f"补齐后再点「查看营养估算」。{tail}"
    )


def _resolve_candidate_source(
    food_query: str,
    detected_query: str,
) -> str:
    """名称原样来自检测才算 model，用户改过一个字就回到 manual。"""

    if not detected_query.strip():
        return "manual"

    if food_query.strip() != detected_query.strip():
        return "manual"

    return "model"


def calculate_meal_preview(
    image_path: str | None,
    food_query: str,
    selected_food_id: str | None,
    raw_grams: Any,
    detected_query: str = "",
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

        # 一次把缺的都说清楚，不要用户点一次补一项。
        missing: list[str] = []

        if (
            not selected_food_id
            or selected_food_id
            not in candidate_ids
        ):
            missing.append(
                "在「选择最接近的食物」里选中一个候选"
            )

        if _grams_is_blank(raw_grams):
            missing.append(
                "填写「估计份量（g）」"
            )

        if missing:
            return (
                _missing_input_text(missing),
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
                    _resolve_candidate_source(
                        food_query,
                        detected_query,
                    )
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

    # 先给一眼能看懂的结论，再给明细。用"千卡""克"而不是 kcal/g，
    # 用"大约"而不是"估算值"——后者是内部说法，用户只关心准不准。
    summary = (
        f"### {food.name} {float(grams):g} 克 "
        f"大约 {estimate.calories_kcal:.0f} 千卡\n\n"
        f"蛋白质 {estimate.protein_g:.1f} 克 · "
        f"脂肪 {estimate.fat_g:.1f} 克 · "
        f"碳水 {estimate.carbs_g:.1f} 克\n\n"
        "数值按食物成分表换算，是大概值。"
        "点「保存这一餐」才会记下来，这不是医疗建议。"
    )

    return (
        summary,
        preview_state,
        (
            "计算完成，请核对后"
            "点击确认保存。"
        ),
        "",
    )


def confirm_meal_save(
    preview_state: (
        dict[str, Any]
        | None
    ),
    history: (
        list[dict[str, Any]]
        | None
    ),
) -> tuple[
    str,
    list[list[Any]],
    str,
    str,
    None,
    list[dict[str, Any]],
    None,
    None,
    gr.Column,
    str,
    str,
    gr.Radio,
    str,
    str,
    str,
    str,
]:
    """明确点击后保存饮食草稿。"""

    if not preview_state:
        rows, summary = refresh_today()

        return (
            "还不能保存：请先填好食物名称和份量，"
            "点「匹配食物」，选中一个候选，"
            "再点「查看营养估算」生成待确认记录。",
            rows,
            summary,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
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
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
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

    chat_history = list(history or [])
    chat_history.append(
        {
            "role": "assistant",
            "content": status,
        }
    )

    return (
        "",
        rows,
        summary,
        "",
        None,
        chat_history,
        None,
        None,
        gr.Column(visible=False),
        "",
        "",
        gr.Radio(choices=[], value=None, visible=False),
        "",
        "",
        "",
        "",
    )


def cancel_meal_preview() -> tuple[
    None,
    str,
    str,
    str,
    None,
    None,
    gr.Column,
    str,
    str,
    gr.Radio,
    str,
    str,
    str,
]:
    """取消饮食草稿。"""

    return (
        None,
        "已取消，没有记录任何东西。",
        "",
        "",
        None,
        None,
        gr.Column(visible=False),
        "",
        "",
        gr.Radio(choices=[], value=None, visible=False),
        "",
        "",
        "",
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

    session = _get_agent_session(
        request
    )

    if session is None:
        answer = AGENT_UNAVAILABLE_ANSWER

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            normalized_text,
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
            f"这次没能联系上模型服务：{exc}。\n\n"
            "刚才的内容没有写入健康记录，"
            "输入框里的原话我保留着，"
            "处理完上面这一项后可以直接重发。"
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            chat_history,
            normalized_text,
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

    except Exception:
        answer = (
            "处理时遇到问题，请稍后再试。"
            "刚才的内容没有写入健康记录。"
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
            "对话功能暂时不可用，"
            "当前没有可以保存的内容。"
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

    except Exception:
        answer = (
            "没有保存成功，请稍后再试。"
            "草稿仍然保留，你可以再次确认。"
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
            "对话功能暂时不可用，"
            "当前没有需要取消的内容。"
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

    except Exception:
        answer = (
            "暂时无法取消，请再试一次。"
            "当前内容尚未保存。"
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
    initial_profile = healthos_store.get_profile(LOCAL_USER_ID, APP_TIMEZONE)

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
                <div class="brand-origin" aria-hidden="true"><i></i><b></b><span></span></div>
                <div class="brand-message">
                  <h1>小满</h1>
                  <p>个人身体观察手册</p>
                </div>
              </div>
            </header>
            """,
            sanitize_html=False,
            container=False,
            elem_id="healthos-brand-shell",
        )

        new_conversation_button = gr.Button(
            "＋  创建新对话",
            variant="secondary",
            elem_id="new-conversation-button",
        )
        sidebar_conversations = gr.Radio(
            choices=_conversation_choices(),
            value=None,
            label="最近对话",
            show_label=True,
            elem_id="sidebar-conversations",
        )
        coach_style_selector = gr.Dropdown(
            choices=[
                ("温和陪伴", "gentle"),
                ("理性观察", "rational"),
                ("简洁行动", "concise"),
                ("目标督促", "goal_focused"),
            ],
            value=initial_profile.coach_style.value,
            show_label=False,
            container=False,
            elem_id="topbar-coach-style",
        )

        gr.Markdown(
            f"""
            <header class="app-topbar">
              <div><h1>今日观察</h1><span>{escape(initial_date)} · 记录、确认，再回看真实变化</span></div>
              <p><b>理性观察</b><span>非医疗服务</span></p>
            </header>
            """,
            sanitize_html=False,
            container=False,
            elem_id="healthos-topbar-shell",
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
                "今日观察",
                id="chat",
                elem_id="healthos-record",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    with gr.Row(elem_classes="workspace-heading"):
                        gr.Markdown(
                            """
                            <div class="page-title conversation-title">
                              <h2>先看见今天。</h2>
                              <p>记录不必完整。小满会把事实、来源与尚未确认的部分分别放好。</p>
                            </div>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        view_today_button = gr.Button(
                            "查看完整汇总",
                            variant="secondary",
                            size="md",
                            scale=0,
                            min_width=112,
                        )

                    gr.Markdown(
                        _setup_notice_html(),
                        sanitize_html=False,
                        container=False,
                        visible=agent_model is None,
                        elem_classes="setup-notice",
                    )

                    with gr.Column(
                        scale=0,
                        min_width=0,
                        elem_classes="conversation-starters"
                    ):
                        gr.Markdown(
                            """
                            <div class="starter-heading">
                              <strong>快速记一笔</strong>
                              <span>选择类型后仍会进入同一条真实对话与确认流程。</span>
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

                    agent_status = gr.State(
                        (
                            "对话服务已连接，可以开始记录。"
                            if agent_model is not None
                            else (
                                "对话服务尚未启用。"
                                "你仍可以使用今日概览、"
                                "时间线和饮食记录。"
                            )
                        )
                    )
                    pending_agent_text = gr.State(value="")
                    meal_preview_state = gr.State(value=None)

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

                            with gr.Column(
                                visible=False,
                                elem_classes="chat-meal-workflow",
                            ) as meal_image_workflow:
                                gr.Markdown(
                                    f"""
                                    <div class="chat-meal-heading">
                                      <strong>确认餐食信息</strong>
                                      <span>{MEAL_WORKFLOW_STEPS}</span>
                                    </div>
                                    """,
                                    sanitize_html=False,
                                    container=False,
                                )

                                detection_status = gr.Markdown(
                                    container=False,
                                    elem_classes="meal-inline-status",
                                )
                                detected_query_state = gr.State(value="")

                                with gr.Row(
                                    equal_height=False,
                                    elem_classes="chat-meal-inputs",
                                ):
                                    meal_image_preview = gr.Image(
                                        value=None,
                                        show_label=False,
                                        interactive=False,
                                        height=156,
                                        elem_classes="chat-meal-image-preview",
                                    )
                                    with gr.Column(
                                        elem_classes="chat-meal-fields",
                                    ):
                                        food_query = gr.Textbox(
                                            label="吃的是什么",
                                            placeholder="例如：西红柿炒蛋",
                                        )
                                        # 用 Textbox 而不是 Number：Number 的
                                        # 客户端校验会抢在服务端之前弹英文
                                        # toast，而且会把空值渲染成 0，看着
                                        # 像已经填了。数值交给 parse_grams。
                                        grams_input = gr.Textbox(
                                            label="吃了多少（克）",
                                            value="",
                                            placeholder="例如 150",
                                            lines=1,
                                            max_lines=1,
                                        )
                                        with gr.Row(
                                            elem_classes="meal-portion-presets",
                                        ):
                                            portion_buttons = [
                                                gr.Button(
                                                    f"{grams}g",
                                                    variant="secondary",
                                                    size="sm",
                                                    min_width=48,
                                                )
                                                for grams in PORTION_PRESETS
                                            ]

                                candidate_status = gr.Markdown(
                                    container=False,
                                    elem_classes="meal-inline-status",
                                )
                                selected_food = gr.Radio(
                                    choices=[],
                                    value=None,
                                    visible=False,
                                    interactive=True,
                                    container=False,
                                    elem_classes="meal-candidate-select",
                                )
                                meal_preview = gr.Markdown(
                                    "",
                                    elem_classes="meal-preview",
                                )

                                recompute_evidence = gr.State(value="")

                                with gr.Row(
                                    elem_classes="meal-confirm-actions",
                                ):
                                    meal_cancel_button = gr.Button(
                                        "取消",
                                        variant="secondary",
                                    )
                                    meal_save_button = gr.Button(
                                        "保存这一餐",
                                        variant="primary",
                                    )
                                meal_save_status = gr.Markdown(
                                    container=False,
                                    elem_classes="meal-inline-status",
                                )

                            with gr.Column(elem_classes="composer-shell"):
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

                                with gr.Row(elem_classes="composer-tools"):
                                    image_input = gr.UploadButton(
                                        "添加图片",
                                        file_count="single",
                                        file_types=[
                                            ".jpg",
                                            ".jpeg",
                                            ".png",
                                        ],
                                        type="filepath",
                                        variant="secondary",
                                        size="sm",
                                        min_width=96,
                                        elem_id="chat-meal-upload",
                                    )

            with gr.Tab(
                "对话",
                id="conversations",
                elem_id="healthos-conversations",
            ):
                with gr.Column(elem_classes="page-wrap conversation-library"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>继续一段对话。</h2>
                          <p>选择过去的真实会话，回到原来的上下文继续记录或提问。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )
                    mobile_new_conversation_button = gr.Button(
                        "＋  创建新对话",
                        variant="primary",
                    )
                    mobile_conversations = gr.Radio(
                        choices=_conversation_choices(),
                        value=None,
                        label="历史对话",
                        show_label=True,
                        elem_id="mobile-conversations",
                    )
                    mobile_reset_agent_button = gr.Button(
                        "清除当前对话",
                        variant="secondary",
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
                "趋势与报告",
                id="trends",
                elem_id="healthos-trends",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title">
                          <h2>把身体的变化，看得更清楚。</h2>
                          <p>这里只排列已确认事实，不生成健康分数，也不把未记录解释为零。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    with gr.Row(elem_classes="trend-toolbar"):
                        period_days = gr.Dropdown(
                            choices=[
                                ("最近 7 天", 7),
                                ("最近 14 天", 14),
                                ("最近 30 天", 30),
                            ],
                            value=7,
                            label="观察周期",
                            min_width=160,
                            scale=0,
                        )
                        refresh_trends_button = gr.Button(
                            "刷新趋势",
                            variant="secondary",
                            size="sm",
                            scale=0,
                        )

                    with gr.Column(elem_classes="trend-ledger"):
                        trend_table = gr.HTML(value="", elem_id="health-trend-charts")
                        trend_status = gr.Markdown(
                            container=False,
                            elem_classes="trend-status",
                        )

                    with gr.Column(elem_classes="observation-panel"):
                        gr.Markdown(
                            """
                            <div class="section-heading">这一周期记录了什么</div>
                            <p class="section-copy">事实、数据完整度和下一步分开呈现。记录不足时不会推断原因。</p>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        checkin_summary = gr.Markdown(
                            "正在整理已确认记录……",
                            sanitize_html=False,
                            container=False,
                        )
                        ask_review_button = gr.Button(
                            "在对话中继续复盘",
                            variant="secondary",
                            size="sm",
                        )

            with gr.Tab(
                "目标与教练",
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
                        goals_table = gr.HTML(value="", elem_id="health-goal-cards")

                    refresh_healthos_button = gr.Button(
                        "刷新目标与设置",
                        variant="secondary",
                        size="sm",
                    )

            with gr.Tab(
                "提醒",
                id="reminders",
                elem_id="healthos-reminders",
            ):
                with gr.Column(elem_classes="page-wrap"):
                    gr.Markdown(
                        """
                        <div class="page-title reminder-title">
                          <h2>提醒行动，也由你掌控。</h2>
                          <p>所有提醒都通过飞书发送。小满先展示内容和时间，确认后才安排，之后可以修改、延后、暂停或取消。</p>
                        </div>
                        """,
                        sanitize_html=False,
                        container=False,
                    )
                    with gr.Column(elem_classes="care-card"):
                        with gr.Row(elem_classes="card-heading-row"):
                            gr.Markdown(
                            '<div><div class="section-heading">自动提醒</div>'
                            f'<p class="section-copy">{escape(REMINDER_AUTOMATION_STATUS)}。'
                            '发送设置只保存在这台设备上。</p></div>',
                                sanitize_html=False,
                                container=False,
                            )
                            create_reminder_button = gr.Button(
                                "创建提醒", variant="primary", size="sm", scale=0
                            )
                            create_check_in_button = gr.Button(
                                "设置主动问候", variant="secondary", size="sm", scale=0
                            )
                            manage_reminder_button = gr.Button(
                                "管理提醒", variant="secondary", size="sm", scale=0
                            )
                        gr.Markdown(
                            """
                            <section class="check-in-setup">
                              <div>
                                <strong>主动问候默认关闭</strong>
                                <p>你选择时间和频率并确认后，小满才会通过飞书询问是否需要补记；没看到记录不代表你没有完成。</p>
                              </div>
                              <span>需确认启用</span>
                            </section>
                            """,
                            sanitize_html=False,
                            container=False,
                        )
                        reminders_table = gr.Markdown(
                            value=_reminder_cards([]),
                            sanitize_html=False,
                            container=False,
                            elem_classes="reminders-list-wrap",
                        )
                        reminder_status = gr.Markdown(container=False)

                    gr.Markdown(
                        """
                        <section class="reminder-boundary">
                          <strong>自动发送的能力边界</strong>
                          <p>小满页面运行时才能按时发送。页面关闭期间不会发送；重新启动后，会处理已到期且仍有效的提醒。</p>
                        </section>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

            with gr.Tab(
                "运行证据",
                id="developer",
                elem_id="healthos-evidence",
                visible=SHOW_DEVELOPER_UI,
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
                        "HealthOS · 16 个受控工具契约",
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
                          <article class="privacy-item"><i>本</i><div><b>健康记录留在这台设备</b><span>你的档案、目标、记录、提醒和对话都保存在本机。</span></div></article>
                          <article class="privacy-item"><i>图</i><div><b>餐食图片不会保存</b><span>图片只用于本次核对，不会复制进健康记录。</span></div></article>
                          <article class="privacy-item"><i>会</i><div><b>对话由你管理</b><span>刷新页面可以继续上次对话，重置后会删除本次会话。</span></div></article>
                          <article class="privacy-item"><i>确</i><div><b>改动前先征求你同意</b><span>保存、修改和删除都会先让你核对，再由你确认。</span></div></article>
                        </section>
                        """,
                        sanitize_html=False,
                        container=False,
                    )

                    memory_action_state = gr.State(value={})
                    with gr.Column(elem_classes=["care-card", "memory-center"]):
                        with gr.Row(elem_classes="card-heading-row"):
                            gr.Markdown(
                                '<div><div class="section-heading">小满记住了什么</div>'
                                '<p class="section-copy">这里只显示你确认过的长期偏好。内部标识、健康参数和对话原文不会出现在列表中。</p></div>',
                                sanitize_html=False,
                                container=False,
                            )
                            refresh_memory_button = gr.Button("刷新记忆", variant="secondary", size="sm", scale=0)
                        memory_table = gr.Dataframe(
                            headers=["记忆类型", "记住的内容", "确认时间"],
                            datatype=["str", "str", "str"],
                            value=[],
                            interactive=False,
                            show_label=False,
                            wrap=True,
                            max_height=300,
                            elem_classes="memory-table",
                        )
                        memory_selector = gr.Dropdown(
                            label="选择要遗忘的内容",
                            choices=[],
                            value=None,
                            info="删除后，下一轮对话不会再使用这项偏好。",
                        )
                        memory_status = gr.Markdown(container=False)
                        memory_confirmation = gr.Markdown(
                            value="", visible=False, sanitize_html=False, container=False
                        )
                        with gr.Row(elem_classes="memory-actions"):
                            delete_memory_button = gr.Button("遗忘所选", variant="secondary")
                            export_memory_button = gr.Button("导出记忆", variant="secondary")
                            clear_memories_button = gr.Button("清除全部记忆", variant="stop")
                        with gr.Row(elem_classes="memory-confirm-actions"):
                            cancel_memory_button = gr.Button("取消", variant="secondary", visible=False)
                            confirm_memory_button = gr.Button("确认执行", variant="stop", visible=False)
                        memory_export_file = gr.File(label="记忆导出文件", visible=True)

        right_rail_summary = gr.Markdown(
            value="正在读取今日事实……",
            sanitize_html=False,
            container=False,
            elem_id="healthos-margin",
        )

        conversation_switch_outputs = [
            browser_conversation_id,
            chatbot,
            agent_status,
            pending_agent_card,
            confirm_agent_button,
            cancel_agent_button,
            sidebar_conversations,
            mobile_conversations,
        ]
        for conversation_picker in (sidebar_conversations, mobile_conversations):
            switch_event = conversation_picker.select(
                fn=switch_agent_conversation,
                inputs=[conversation_picker],
                outputs=conversation_switch_outputs,
                show_progress="hidden",
            )
            switch_event.then(
                fn=open_record_workspace,
                outputs=[main_tabs],
                show_progress="hidden",
            )

        new_conversation_outputs = [
            browser_conversation_id,
            chatbot,
            agent_status,
            pending_agent_card,
            confirm_agent_button,
            cancel_agent_button,
            selected_record_state,
            selected_record_context,
            sidebar_conversations,
            mobile_conversations,
        ]
        for create_button in (new_conversation_button, mobile_new_conversation_button):
            create_event = create_button.click(
                fn=create_agent_conversation,
                outputs=new_conversation_outputs,
                show_progress="hidden",
            )
            create_event.then(
                fn=open_record_workspace,
                outputs=[main_tabs],
                show_progress="hidden",
            )

        coach_style_selector.input(
            fn=open_coach_style_change,
            inputs=[coach_style_selector],
            outputs=[chat_input, main_tabs],
                show_progress="hidden",
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

        refresh_today_event = refresh_today_button.click(
            fn=refresh_today,
            outputs=[
                today_table,
                today_summary,
            ],
        )
        refresh_today_event.then(
            fn=refresh_today_margin,
            outputs=[right_rail_summary],
            show_progress="hidden",
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

        refresh_trends_button.click(
            fn=refresh_trends,
            inputs=[period_days],
            outputs=[trend_table, trend_status],
            show_progress="hidden",
        )

        refresh_memory_button.click(
            fn=refresh_memory_center,
            outputs=[memory_table, memory_selector, memory_status],
            show_progress="hidden",
        )

        delete_memory_button.click(
            fn=prepare_memory_delete,
            inputs=[memory_selector],
            outputs=[memory_action_state, memory_confirmation, confirm_memory_button, cancel_memory_button],
            show_progress="hidden",
        )

        clear_memories_button.click(
            fn=prepare_memory_clear,
            outputs=[memory_action_state, memory_confirmation, confirm_memory_button, cancel_memory_button],
            show_progress="hidden",
        )

        cancel_memory_button.click(
            fn=cancel_memory_action,
            outputs=[memory_action_state, memory_confirmation, confirm_memory_button, cancel_memory_button],
            show_progress="hidden",
        )

        confirm_memory_button.click(
            fn=confirm_memory_action,
            inputs=[memory_action_state],
            outputs=[
                memory_table,
                memory_selector,
                memory_status,
                memory_action_state,
                memory_confirmation,
                confirm_memory_button,
                cancel_memory_button,
            ],
            show_progress="hidden",
        )

        export_memory_button.click(
            fn=export_memory_file,
            outputs=[memory_export_file, memory_status],
            show_progress="hidden",
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
        period_days.change(
            fn=refresh_trends,
            inputs=[period_days],
            outputs=[trend_table, trend_status],
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
            (create_check_in_button, open_active_check_in),
            (manage_reminder_button, open_reminder_management),
        ):
            action_button.click(
                fn=action_function,
                outputs=healthos_action_outputs,
                show_progress="hidden",
            )

        upload_event = image_input.upload(
            fn=open_meal_image_workflow,
            inputs=[image_input],
            outputs=[
                meal_image_workflow,
                meal_image_preview,
                food_query,
                detection_status,
                detected_query_state,
            ],
            show_progress="hidden",
        )

        # 匹配和试算都是系统的内部动作，不该让用户点按钮触发。
        # 名称、候选、份量任意一项变化都重跑同一个编排函数。
        meal_panel_inputs = [
            image_input,
            food_query,
            selected_food,
            grams_input,
            detected_query_state,
        ]
        meal_panel_outputs = [
            selected_food,
            candidate_status,
            meal_preview,
            meal_preview_state,
            latest_retrieval_trace,
        ]

        upload_event.then(
            fn=refresh_meal_panel,
            inputs=meal_panel_inputs,
            outputs=meal_panel_outputs,
            show_progress="hidden",
        )

        for meal_trigger in (
            food_query.submit,
            food_query.blur,
            grams_input.submit,
            grams_input.blur,
            selected_food.change,
        ):
            meal_trigger(
                fn=refresh_meal_panel,
                inputs=meal_panel_inputs,
                outputs=meal_panel_outputs,
                show_progress="hidden",
            )

        # 份量快捷按钮：填进输入框后立刻重算。
        for portion_button, portion_grams in zip(
            portion_buttons,
            PORTION_PRESETS,
        ):
            portion_button.click(
                fn=partial(use_portion_preset, portion_grams),
                outputs=[grams_input],
                show_progress="hidden",
            ).then(
                fn=refresh_meal_panel,
                inputs=meal_panel_inputs,
                outputs=meal_panel_outputs,
                show_progress="hidden",
            )

        meal_save_event = meal_save_button.click(
            fn=confirm_meal_save,
            inputs=[
                meal_preview_state,
                chatbot,
            ],
            outputs=[
                meal_save_status,
                today_table,
                today_summary,
                meal_preview,
                meal_preview_state,
                chatbot,
                image_input,
                meal_image_preview,
                meal_image_workflow,
                food_query,
                grams_input,
                selected_food,
                candidate_status,
                recompute_evidence,
                detection_status,
                detected_query_state,
            ],
        )

        meal_save_event.then(
            fn=refresh_healthos_dashboard,
            inputs=[period_days],
            outputs=healthos_outputs,
            show_progress="hidden",
        )
        meal_save_event.then(
            fn=refresh_today_margin,
            outputs=[right_rail_summary],
            show_progress="hidden",
        )
        meal_cancel_button.click(
            fn=cancel_meal_preview,
            outputs=[
                meal_preview_state,
                meal_save_status,
                meal_preview,
                recompute_evidence,
                image_input,
                meal_image_preview,
                meal_image_workflow,
                food_query,
                grams_input,
                selected_food,
                candidate_status,
                detection_status,
                detected_query_state,
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
            staged_event: Any,
        ) -> None:
            """让发送、回车和快捷语句复用同一条 Agent 链路。"""

            activity_event = staged_event.then(
                fn=begin_agent_activity,
                inputs=[
                    pending_agent_text,
                    selected_record_state,
                ],
                outputs=[agent_activity],
                queue=False,
                show_progress="hidden",
            )

            result_event = activity_event.then(
                fn=send_chat_message,
                inputs=[
                    pending_agent_text,
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
                inputs=[
                    agent_status,
                    latest_agent_steps,
                    latest_agent_state,
                ],
                outputs=[agent_activity],
                queue=False,
                show_progress="hidden",
            )
            result_event.then(
                fn=refresh_conversation_pickers,
                inputs=[browser_conversation_id],
                outputs=[sidebar_conversations, mobile_conversations],
                show_progress="hidden",
            )

        send_stage_event = send_button.click(
            fn=stage_chat_message,
            inputs=[
                chat_input,
                chatbot,
            ],
            outputs=[
                chatbot,
                chat_input,
                pending_agent_text,
            ],
            queue=False,
            show_progress="hidden",
        )
        bind_agent_turn(send_stage_event)

        submit_stage_event = chat_input.submit(
            fn=stage_chat_message,
            inputs=[
                chat_input,
                chatbot,
            ],
            outputs=[
                chatbot,
                chat_input,
                pending_agent_text,
            ],
            queue=False,
            show_progress="hidden",
        )
        bind_agent_turn(submit_stage_event)

        for starter_button, starter_message in starter_buttons:
            starter_event = starter_button.click(
                fn=partial(
                    conversation_starter_message,
                    starter_message,
                ),
                outputs=[chat_input],
                show_progress="hidden",
            )
            starter_stage_event = starter_event.then(
                fn=stage_chat_message,
                inputs=[
                    chat_input,
                    chatbot,
                ],
                outputs=[
                    chatbot,
                    chat_input,
                    pending_agent_text,
                ],
                queue=False,
                show_progress="hidden",
            )
            bind_agent_turn(starter_stage_event)

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
        confirm_agent_event.then(
            fn=refresh_today_margin,
            outputs=[right_rail_summary],
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

        reset_conversation_event = mobile_reset_agent_button.click(
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
        reset_conversation_event.then(
            fn=refresh_conversation_pickers,
            inputs=[browser_conversation_id],
            outputs=[sidebar_conversations, mobile_conversations],
            show_progress="hidden",
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
            fn=refresh_today_margin,
            outputs=[right_rail_summary],
            show_progress="hidden",
        )

        demo.load(
            fn=refresh_trends,
            inputs=[period_days],
            outputs=[trend_table, trend_status],
            show_progress="hidden",
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
            fn=refresh_conversation_pickers,
            inputs=[browser_conversation_id],
            outputs=[sidebar_conversations, mobile_conversations],
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

        demo.load(
            fn=refresh_memory_center,
            outputs=[memory_table, memory_selector, memory_status],
            show_progress="hidden",
        )

        demo.unload(
            cleanup_agent_session
        )

    return demo


demo = build_demo()
