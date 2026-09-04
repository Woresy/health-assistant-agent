"""本地持久化 Agent 会话状态。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import RLock

from pydantic import ValidationError

from src.agent.models import SessionState


_SESSION_ID_PATTERN = re.compile(
    r"^conversation-[0-9a-f]{32}$"
)


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

