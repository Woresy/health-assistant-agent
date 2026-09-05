"""HealthOS 本地 SQLite 主存储与旧文件迁移。

SQLite 保存需要查询、更新和事务一致性的业务状态；JSONL 继续用于只追加的
Agent/RAG Trace 和显式导出。三个适配器保持现有 Store 的外部协议，使工具层无需
感知存储切换。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator
from uuid import UUID

from src.agent.models import SessionState
from src.health.models import EventType, HealthEvent
from src.healthos.models import HealthOSState, UserProfile
from src.storage.conversation_store import _SESSION_ID_PATTERN
from src.storage.healthos_store import HealthOSStore, HealthOSStoreError
from src.storage.jsonl_store import (
    HealthEventConflictError,
    HealthEventNotFoundError,
    HealthEventStore,
    JsonlReadError,
    JsonlWriteError,
    _require_aware_datetime,
)


SCHEMA_VERSION = 1


class SQLiteStoreError(Exception):
    """SQLite 初始化、读取或事务写入失败。"""


class SQLiteDatabase:
    """共享数据库连接策略和可重复执行的 Schema migration。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with self._lock, self.connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS health_events (
                        event_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        event_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_health_events_user_time
                        ON health_events(user_id, occurred_at);
                    CREATE INDEX IF NOT EXISTS idx_health_events_user_type_time
                        ON health_events(user_id, event_type, occurred_at);
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS health_goals (
                        goal_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        goal_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_health_goals_user
                        ON health_goals(user_id);
                    CREATE TABLE IF NOT EXISTS reminders (
                        reminder_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        reminder_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_reminders_user_time
                        ON reminders(user_id, updated_at);
                    CREATE TABLE IF NOT EXISTS idempotency_results (
                        idempotency_key TEXT PRIMARY KEY,
                        result_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_conversations_user_time
                        ON conversation_sessions(user_id, updated_at);
                    CREATE TABLE IF NOT EXISTS legacy_imports (
                        source_key TEXT PRIMARY KEY,
                        source_digest TEXT NOT NULL,
                        imported_at TEXT NOT NULL,
                        row_count INTEGER NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version, applied_at) "
                    "VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now().astimezone().isoformat()),
                )
                connection.commit()
            os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise SQLiteStoreError(f"初始化 SQLite 失败：{exc}") from exc

    def integrity_check(self) -> dict[str, Any]:
        """返回可用于启动验收和迁移后验证的最小证据。"""

        with self._lock, self.connect() as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            tables = {
                name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in (
                    "health_events",
                    "user_profiles",
                    "health_goals",
                    "reminders",
                    "conversation_sessions",
                )
            }
        return {"ok": integrity == "ok", "integrity": integrity, "counts": tables}


class SQLiteHealthEventStore(HealthEventStore):
    """与 JSONL HealthEventStore 等价的事务型 SQLite 适配器。"""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        self.path = self.database.path
        self._lock = self.database._lock

    @staticmethod
    def _serialize(event: HealthEvent) -> tuple[str, ...]:
        return (
            str(event.event_id),
            event.user_id,
            event.event_type.value,
            event.occurred_at.isoformat(),
            event.created_at.isoformat(),
            event.updated_at.isoformat(),
            event.model_dump_json(),
        )

    def append(self, event: HealthEvent) -> None:
        try:
            with self._lock, self.database.connect() as connection:
                connection.execute(
                    "INSERT INTO health_events(event_id,user_id,event_type,occurred_at,"
                    "created_at,updated_at,event_json) VALUES (?,?,?,?,?,?,?)",
                    self._serialize(event),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise JsonlWriteError(f"健康事件已存在：{event.event_id}") from exc
        except sqlite3.Error as exc:
            raise JsonlWriteError(f"写入健康事件失败：{exc}") from exc

    def read_all(self) -> list[HealthEvent]:
        try:
            with self._lock, self.database.connect() as connection:
                rows = connection.execute(
                    "SELECT event_json FROM health_events ORDER BY rowid"
                ).fetchall()
            return [HealthEvent.model_validate_json(row["event_json"]) for row in rows]
        except (sqlite3.Error, ValueError) as exc:
            raise JsonlReadError(0, f"读取 SQLite 健康事件失败：{exc}") from exc

    def find_by_event_id(self, event_id: UUID) -> HealthEvent | None:
        try:
            with self._lock, self.database.connect() as connection:
                row = connection.execute(
                    "SELECT event_json FROM health_events WHERE event_id = ?", (str(event_id),)
                ).fetchone()
            return HealthEvent.model_validate_json(row["event_json"]) if row else None
        except (sqlite3.Error, ValueError) as exc:
            raise JsonlReadError(0, f"读取 SQLite 健康事件失败：{exc}") from exc

    def query(
        self,
        *,
        user_id: str,
        event_type: EventType | str | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
        newest_first: bool = False,
    ) -> list[HealthEvent]:
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id 不能为空")
        normalized_type: EventType | None = None
        if event_type is not None:
            try:
                normalized_type = EventType(event_type)
            except ValueError as exc:
                raise ValueError("event_type 必须是 meal、water、weight 或 exercise") from exc
        if occurred_from is not None:
            _require_aware_datetime(occurred_from, "occurred_from")
        if occurred_to is not None:
            _require_aware_datetime(occurred_to, "occurred_to")
        if occurred_from is not None and occurred_to is not None and occurred_to <= occurred_from:
            raise ValueError("occurred_to 必须晚于 occurred_from")

        clauses = ["user_id = ?"]
        parameters: list[Any] = [normalized_user_id]
        if normalized_type is not None:
            clauses.append("event_type = ?")
            parameters.append(normalized_type.value)
        sql = "SELECT event_json FROM health_events WHERE " + " AND ".join(clauses)
        try:
            with self._lock, self.database.connect() as connection:
                rows = connection.execute(sql, parameters).fetchall()
            events = [HealthEvent.model_validate_json(row["event_json"]) for row in rows]
        except (sqlite3.Error, ValueError) as exc:
            raise JsonlReadError(0, f"查询 SQLite 健康事件失败：{exc}") from exc
        # 带不同时区偏移的 ISO 字符串不能直接按文本比较；在模型验证后使用
        # aware datetime 过滤，保证跨时区边界正确。
        events = [
            event
            for event in events
            if (occurred_from is None or event.occurred_at >= occurred_from)
            and (occurred_to is None or event.occurred_at < occurred_to)
        ]
        return sorted(
            events,
            key=lambda item: (item.occurred_at, item.created_at, str(item.event_id)),
            reverse=newest_first,
        )

    def update(
        self,
        *,
        event_id: UUID,
        replacement: HealthEvent,
        expected_updated_at: datetime,
    ) -> HealthEvent:
        _require_aware_datetime(expected_updated_at, "expected_updated_at")
        with self._lock, self.database.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT event_json FROM health_events WHERE event_id = ?", (str(event_id),)
                ).fetchone()
                if row is None:
                    raise HealthEventNotFoundError(event_id)
                current = HealthEvent.model_validate_json(row["event_json"])
                if current.updated_at != expected_updated_at:
                    raise HealthEventConflictError(event_id)
                if replacement.event_id != current.event_id:
                    raise ValueError("更新时不得修改 event_id")
                if replacement.user_id != current.user_id:
                    raise ValueError("更新时不得修改 user_id")
                if replacement.created_at != current.created_at:
                    raise ValueError("更新时不得修改 created_at")
                if replacement.updated_at <= current.updated_at:
                    raise ValueError("replacement.updated_at 必须晚于旧版本 updated_at")
                values = self._serialize(replacement)
                connection.execute(
                    "UPDATE health_events SET user_id=?,event_type=?,occurred_at=?,created_at=?,"
                    "updated_at=?,event_json=? WHERE event_id=?",
                    (*values[1:], values[0]),
                )
                connection.commit()
                return replacement
            except (HealthEventNotFoundError, HealthEventConflictError, ValueError):
                connection.rollback()
                raise
            except Exception as exc:
                connection.rollback()
                raise JsonlWriteError(f"更新 SQLite 健康事件失败：{exc}") from exc

    def delete(self, *, event_id: UUID, expected_updated_at: datetime) -> HealthEvent:
        _require_aware_datetime(expected_updated_at, "expected_updated_at")
        with self._lock, self.database.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT event_json FROM health_events WHERE event_id = ?", (str(event_id),)
                ).fetchone()
                if row is None:
                    raise HealthEventNotFoundError(event_id)
                current = HealthEvent.model_validate_json(row["event_json"])
                if current.updated_at != expected_updated_at:
                    raise HealthEventConflictError(event_id)
                connection.execute("DELETE FROM health_events WHERE event_id = ?", (str(event_id),))
                connection.commit()
                return current
            except (HealthEventNotFoundError, HealthEventConflictError):
                connection.rollback()
                raise
            except Exception as exc:
                connection.rollback()
                raise JsonlWriteError(f"删除 SQLite 健康事件失败：{exc}") from exc


class SQLiteHealthOSStore(HealthOSStore):
    """按档案、目标、提醒和幂等结果分表保存 HealthOS 状态。"""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        self.path = self.database.path
        self._lock = self.database._lock

    @staticmethod
    def _read_connection(connection: sqlite3.Connection) -> HealthOSState:
        profiles = {
            row["user_id"]: json.loads(row["profile_json"])
            for row in connection.execute("SELECT user_id, profile_json FROM user_profiles")
        }
        goals = [
            json.loads(row["goal_json"])
            for row in connection.execute("SELECT goal_json FROM health_goals ORDER BY rowid")
        ]
        reminders = [
            json.loads(row["reminder_json"])
            for row in connection.execute("SELECT reminder_json FROM reminders ORDER BY rowid")
        ]
        idempotency = {
            row["idempotency_key"]: json.loads(row["result_json"])
            for row in connection.execute(
                "SELECT idempotency_key, result_json FROM idempotency_results"
            )
        }
        return HealthOSState.model_validate(
            {
                "profiles": profiles,
                "goals": goals,
                "reminders": reminders,
                "idempotency_results": idempotency,
            }
        )

    def read(self) -> HealthOSState:
        try:
            with self._lock, self.database.connect() as connection:
                return self._read_connection(connection)
        except (sqlite3.Error, ValueError) as exc:
            raise HealthOSStoreError(f"读取 HealthOS SQLite 状态失败：{exc}") from exc

    @staticmethod
    def _write_connection(connection: sqlite3.Connection, state: HealthOSState) -> None:
        connection.execute("DELETE FROM user_profiles")
        connection.execute("DELETE FROM health_goals")
        connection.execute("DELETE FROM reminders")
        connection.execute("DELETE FROM idempotency_results")
        connection.executemany(
            "INSERT INTO user_profiles(user_id, profile_json, updated_at) VALUES (?,?,?)",
            [
                (user_id, profile.model_dump_json(), profile.updated_at.isoformat())
                for user_id, profile in state.profiles.items()
            ],
        )
        connection.executemany(
            "INSERT INTO health_goals(goal_id, user_id, goal_json, updated_at) VALUES (?,?,?,?)",
            [
                (
                    str(goal.goal_id),
                    goal.user_id,
                    goal.model_dump_json(),
                    goal.current.created_at.isoformat(),
                )
                for goal in state.goals
            ],
        )
        connection.executemany(
            "INSERT INTO reminders(reminder_id, user_id, reminder_json, updated_at) VALUES (?,?,?,?)",
            [
                (
                    str(reminder.reminder_id),
                    reminder.user_id,
                    reminder.model_dump_json(),
                    reminder.updated_at.isoformat(),
                )
                for reminder in state.reminders
            ],
        )
        connection.executemany(
            "INSERT INTO idempotency_results(idempotency_key, result_json) VALUES (?,?)",
            [
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in state.idempotency_results.items()
            ],
        )

    def update(self, mutation: Callable[[HealthOSState], Any]) -> Any:
        with self._lock, self.database.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                state = self._read_connection(connection)
                result = mutation(state)
                self._write_connection(connection, state)
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    def get_profile(self, user_id: str, timezone_name: str) -> UserProfile:
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id 不能为空")
        existing = self.read().profiles.get(normalized_user_id)
        if existing is not None:
            return existing
        return UserProfile(
            user_id=normalized_user_id,
            timezone_name=timezone_name,
            updated_at=datetime.now().astimezone(),
        )


class SQLiteConversationStore:
    """持久化完整会话快照，刷新和进程重启后均可恢复。"""

    def __init__(self, database: SQLiteDatabase | str | Path) -> None:
        self.database = database if isinstance(database, SQLiteDatabase) else SQLiteDatabase(database)
        self.path = self.database.path
        self._lock = self.database._lock

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("无效的持久化会话标识")

    def load(self, session_id: str) -> SessionState | None:
        self._validate_session_id(session_id)
        try:
            with self._lock, self.database.connect() as connection:
                row = connection.execute(
                    "SELECT state_json FROM conversation_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            return SessionState.model_validate_json(row["state_json"]) if row else None
        except (sqlite3.Error, ValueError):
            return None

    def save(self, state: SessionState) -> None:
        self._validate_session_id(state.session_id)
        with self._lock, self.database.connect() as connection:
            connection.execute(
                "INSERT INTO conversation_sessions(session_id,user_id,state_json,updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                "user_id=excluded.user_id,state_json=excluded.state_json,updated_at=excluded.updated_at",
                (
                    state.session_id,
                    state.user_id,
                    state.model_dump_json(indent=2),
                    datetime.now().astimezone().isoformat(),
                ),
            )
            connection.commit()

    def delete(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        with self._lock, self.database.connect() as connection:
            connection.execute(
                "DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,)
            )
            connection.commit()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_legacy_storage(
    database: SQLiteDatabase,
    *,
    events_path: Path,
    healthos_path: Path,
    conversations_path: Path,
) -> dict[str, int]:
    """非破坏、可重复地导入旧文件；源文件不删除，因而可直接回滚。"""

    imported = {"health_events": 0, "healthos_entities": 0, "conversations": 0}
    with database._lock, database.connect() as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")

            def already_imported(source_key: str, source_path: Path) -> bool:
                if not source_path.exists():
                    return True
                digest = _digest(source_path)
                row = connection.execute(
                    "SELECT source_digest FROM legacy_imports WHERE source_key = ?",
                    (source_key,),
                ).fetchone()
                return bool(row and row["source_digest"] == digest)

            if events_path.exists() and not already_imported("health_events", events_path):
                events = HealthEventStore(events_path).read_all()
                for event in events:
                    connection.execute(
                        "INSERT OR IGNORE INTO health_events(event_id,user_id,event_type,occurred_at,"
                        "created_at,updated_at,event_json) VALUES (?,?,?,?,?,?,?)",
                        SQLiteHealthEventStore._serialize(event),
                    )
                imported["health_events"] = len(events)
                connection.execute(
                    "INSERT OR REPLACE INTO legacy_imports VALUES (?,?,?,?)",
                    ("health_events", _digest(events_path), datetime.now().astimezone().isoformat(), len(events)),
                )

            if healthos_path.exists() and not already_imported("healthos_state", healthos_path):
                state = HealthOSStore(healthos_path).read()
                SQLiteHealthOSStore._write_connection(connection, state)
                count = len(state.profiles) + len(state.goals) + len(state.reminders)
                imported["healthos_entities"] = count
                connection.execute(
                    "INSERT OR REPLACE INTO legacy_imports VALUES (?,?,?,?)",
                    ("healthos_state", _digest(healthos_path), datetime.now().astimezone().isoformat(), count),
                )

            if conversations_path.exists():
                for path in sorted(conversations_path.glob("conversation-*.json")):
                    source_key = f"conversation:{path.name}"
                    if already_imported(source_key, path):
                        continue
                    if not re.fullmatch(r"conversation-[0-9a-f]{32}\.json", path.name):
                        continue
                    try:
                        state = SessionState.model_validate_json(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    connection.execute(
                        "INSERT INTO conversation_sessions(session_id,user_id,state_json,updated_at) "
                        "VALUES (?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                        "user_id=excluded.user_id,state_json=excluded.state_json,updated_at=excluded.updated_at",
                        (
                            state.session_id,
                            state.user_id,
                            state.model_dump_json(indent=2),
                            datetime.now().astimezone().isoformat(),
                        ),
                    )
                    imported["conversations"] += 1
                    connection.execute(
                        "INSERT OR REPLACE INTO legacy_imports VALUES (?,?,?,?)",
                        (source_key, _digest(path), datetime.now().astimezone().isoformat(), 1),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return imported
