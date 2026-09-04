"""HealthOS P1 档案、目标与提醒的原子本地存储。"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import UUID

from pydantic import ValidationError

from src.healthos.models import HealthOSState, UserProfile


class HealthOSStoreError(Exception):
    """P1 本地状态无法读取或写入。"""


class HealthOSStore:
    """以单个严格 JSON 快照保存 P1 状态并执行原子替换。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def read(self) -> HealthOSState:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> HealthOSState:
        if not self.path.exists():
            return HealthOSState()
        try:
            return HealthOSState.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise HealthOSStoreError(f"读取 HealthOS 状态失败：{exc}") from exc

    def update(self, mutation: Callable[[HealthOSState], Any]) -> Any:
        """在锁内读取、变更和原子保存；mutation 返回业务结果。"""

        with self._lock:
            state = self._read_unlocked()
            result = mutation(state)
            self._write_unlocked(state)
            return result

    def _write_unlocked(self, state: HealthOSState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            file_descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(raw_path)
            os.fchmod(file_descriptor, 0o600)
            content = state.model_dump_json(indent=2).encode("utf-8")
            written = os.write(file_descriptor, content)
            if written != len(content):
                raise OSError(f"仅写入 {written} 字节，预期 {len(content)} 字节")
            os.fsync(file_descriptor)
            os.close(file_descriptor)
            file_descriptor = None
            os.replace(temporary_path, self.path)
            temporary_path = None
        except OSError as exc:
            raise HealthOSStoreError(f"写入 HealthOS 状态失败：{exc}") from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_profile(self, user_id: str, timezone_name: str) -> UserProfile:
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id 不能为空")
        state = self.read()
        existing = state.profiles.get(normalized_user_id)
        if existing is not None:
            return existing
        return UserProfile(
            user_id=normalized_user_id,
            timezone_name=timezone_name,
            updated_at=datetime.now(timezone.utc),
        )

    def idempotent_result(self, key: str) -> dict[str, Any] | None:
        return self.read().idempotency_results.get(key)

    @staticmethod
    def find_goal_index(state: HealthOSState, user_id: str, goal_id: UUID) -> int:
        for index, goal in enumerate(state.goals):
            if goal.user_id == user_id and goal.goal_id == goal_id:
                return index
        raise ValueError("健康目标不存在或已删除")

    @staticmethod
    def find_reminder_index(
        state: HealthOSState, user_id: str, reminder_id: UUID
    ) -> int:
        for index, reminder in enumerate(state.reminders):
            if reminder.user_id == user_id and reminder.reminder_id == reminder_id:
                return index
        raise ValueError("提醒不存在或已删除")
