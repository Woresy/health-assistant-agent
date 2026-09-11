"""本地持久化 Agent 会话状态。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock

from pydantic import ValidationError

from src.agent.models import SessionState


_SESSION_ID_PATTERN = re.compile(
    r"^conversation-[0-9a-f]{32}$"
)


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """用于会话导航的安全摘要，不暴露工具消息或内部状态。"""

    session_id: str
    user_id: str
    title: str
    updated_at: datetime
    message_count: int


def conversation_title(state: SessionState) -> str:
    """从首条真实用户消息生成稳定标题。"""

    for message in state.messages:
        if message.role != "user":
            continue
        content = message.content.strip()
        marker = "用户原始请求："
        if content.startswith("[内部选中记录]") and marker in content:
            content = content.split(marker, maxsplit=1)[1].strip()
        if content:
            return content if len(content) <= 24 else f"{content[:24]}…"
    return "新对话"


class ConversationStore:
    """使用独立 JSON 文件保存每个浏览器会话。"""

    def __init__(
        self,
        directory: Path,
    ) -> None:
        self._directory = directory
        self._lock = RLock()

    def _path(
        self,
        session_id: str,
    ) -> Path:
        if not _SESSION_ID_PATTERN.fullmatch(
            session_id
        ):
            raise ValueError(
                "无效的持久化会话标识"
            )

        return self._directory / f"{session_id}.json"

    def load(
        self,
        session_id: str,
    ) -> SessionState | None:
        """读取会话。文件缺失或损坏时安全返回空。"""

        path = self._path(session_id)

        with self._lock:
            try:
                raw_data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
                return SessionState.model_validate(
                    raw_data
                )
            except (
                FileNotFoundError,
                OSError,
                json.JSONDecodeError,
                ValidationError,
            ):
                return None

    def save(
        self,
        state: SessionState,
    ) -> None:
        """以原子替换方式保存完整会话状态。"""

        path = self._path(
            state.session_id
        )
        temporary_path = path.with_suffix(
            ".tmp"
        )
        payload = json.dumps(
            state.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )

        with self._lock:
            self._directory.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o700,
            )
            temporary_path.write_text(
                payload,
                encoding="utf-8",
            )
            os.chmod(
                temporary_path,
                0o600,
            )
            temporary_path.replace(path)

    def delete(
        self,
        session_id: str,
    ) -> None:
        """删除用户明确重置的会话历史。"""

        path = self._path(session_id)

        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return

    def list_summaries(
        self,
        user_id: str,
        *,
        limit: int = 20,
    ) -> list[ConversationSummary]:
        """按最近更新时间列出指定用户的可继续会话。"""

        normalized_user_id = user_id.strip()
        if not normalized_user_id or limit <= 0:
            return []

        summaries: list[ConversationSummary] = []
        with self._lock:
            try:
                paths = list(self._directory.glob("conversation-*.json"))
            except OSError:
                return []
            for path in paths:
                try:
                    state = SessionState.model_validate_json(path.read_text(encoding="utf-8"))
                    if state.user_id != normalized_user_id:
                        continue
                    updated_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                except (OSError, ValueError, ValidationError):
                    continue
                summaries.append(
                    ConversationSummary(
                        session_id=state.session_id,
                        user_id=state.user_id,
                        title=conversation_title(state),
                        updated_at=updated_at,
                        message_count=sum(
                            message.role in {"user", "assistant"}
                            for message in state.messages
                        ),
                    )
                )

        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        return summaries[:limit]
