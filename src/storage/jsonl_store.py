"""单进程 HealthEvent JSONL 存储。"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import UUID

from pydantic import ValidationError

from src.health.models import (
    EventType,
    HealthEvent,
)


class JsonlReadError(Exception):
    """读取 JSONL 时报告精确坏行位置。"""

    def __init__(
        self,
        line_number: int,
        reason: str,
    ) -> None:
        self.line_number = line_number
        self.reason = reason

        super().__init__(
            f"JSONL 第 {line_number} 行损坏：{reason}"
        )


class JsonlWriteError(Exception):
    """JSONL 写入或原子替换失败。"""


class HealthEventNotFoundError(Exception):
    """指定的健康事件不存在。"""

    def __init__(
        self,
        event_id: UUID,
    ) -> None:
        self.event_id = event_id

        super().__init__(
            f"健康事件不存在：{event_id}"
        )


class HealthEventConflictError(Exception):
    """事件已经被其他操作修改。"""

    def __init__(
        self,
        event_id: UUID,
    ) -> None:
        self.event_id = event_id

        super().__init__(
            "健康事件版本冲突，请重新读取最新记录后再操作："
            f"{event_id}"
        )


def _require_aware_datetime(
    value: datetime,
    field_name: str,
) -> None:
    """查询边界必须包含时区。"""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} 必须包含时区"
        )


class HealthEventStore:
    """
    单进程 HealthEvent JSONL 存储。

    append 使用单行追加和失败回滚。

    update/delete 会先在内存中生成完整的新快照，
    再通过同目录临时文件和 os.replace 原子替换原文件。
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def append(
        self,
        event: HealthEvent,
    ) -> None:
        """追加一个经过校验的完整健康事件。"""

        with self._lock:
            self._append_unlocked(event)

    def _append_unlocked(
        self,
        event: HealthEvent,
    ) -> None:
        """
        一次写入一整行。

        发生异常时尽力恢复到写入前的文件长度，
        避免留下半行数据。
        """

        line = (
            event.model_dump_json()
            + "\n"
        ).encode("utf-8")

        existed_before = False
        original_size = 0
        file_descriptor: int | None = None

        try:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            existed_before = (
                self.path.exists()
            )
            original_size = (
                self.path.stat().st_size
                if existed_before
                else 0
            )

            file_descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND,
                0o600,
            )

            written = os.write(
                file_descriptor,
                line,
            )

            if written != len(line):
                raise OSError(
                    f"仅写入 {written} 字节，"
                    f"预期写入 {len(line)} 字节"
                )

            os.fsync(file_descriptor)
        except OSError as exc:
            if file_descriptor is not None:
                try:
                    os.close(
                        file_descriptor
                    )
                finally:
                    file_descriptor = None

            try:
                if self.path.exists():
                    if existed_before:
                        with self.path.open(
                            "r+b"
                        ) as rollback_file:
                            rollback_file.truncate(
                                original_size
                            )
                            rollback_file.flush()
                            os.fsync(
                                rollback_file.fileno()
                            )
                    else:
                        self.path.unlink()
            except OSError:
                pass

            raise JsonlWriteError(
                f"写入健康事件失败：{exc}"
            ) from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)

    def read_all(
        self,
    ) -> list[HealthEvent]:
        """
        读取全部事件。

        文件不存在时返回空列表。
        遇到空行、非法 JSON 或 Schema 错误时立即失败。
        """

        with self._lock:
            return self._read_all_unlocked()

    def _read_all_unlocked(
        self,
    ) -> list[HealthEvent]:
        """锁内部使用的读取方法。"""

        if not self.path.exists():
            return []

        events: list[HealthEvent] = []

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as source_file:
                for (
                    line_number,
                    raw_line,
                ) in enumerate(
                    source_file,
                    start=1,
                ):
                    if not raw_line.strip():
                        raise JsonlReadError(
                            line_number,
                            "空行不是有效健康事件",
                        )

                    try:
                        event = (
                            HealthEvent
                            .model_validate_json(
                                raw_line
                            )
                        )
                    except ValidationError as exc:
                        raise JsonlReadError(
                            line_number,
                            str(exc),
                        ) from exc

                    events.append(event)
        except JsonlReadError:
            raise
        except OSError as exc:
            raise JsonlReadError(
                0,
                f"读取文件失败：{exc}",
            ) from exc

        return events

    def find_by_event_id(
        self,
        event_id: UUID,
    ) -> HealthEvent | None:
        """通过稳定事件 ID 查找记录。"""

        with self._lock:
            for event in (
                self._read_all_unlocked()
            ):
                if (
                    event.event_id
                    == event_id
                ):
                    return event

        return None

    def query(
        self,
        *,
        user_id: str,
        event_type: (
            EventType
            | str
            | None
        ) = None,
        occurred_from: (
            datetime
            | None
        ) = None,
        occurred_to: (
            datetime
            | None
        ) = None,
        newest_first: bool = False,
    ) -> list[HealthEvent]:
        """
        查询已经保存的健康事件。

        时间范围使用：
        occurred_from <= occurred_at < occurred_to

        默认按发生时间从早到晚排序。
        """

        normalized_user_id = (
            user_id.strip()
        )

        if not normalized_user_id:
            raise ValueError(
                "user_id 不能为空"
            )

        normalized_event_type: (
            EventType
            | None
        ) = None

        if event_type is not None:
            try:
                normalized_event_type = (
                    EventType(event_type)
                )
            except ValueError as exc:
                raise ValueError(
                    "event_type 必须是 "
                    "meal、water、weight "
                    "或 exercise"
                ) from exc

        if occurred_from is not None:
            _require_aware_datetime(
                occurred_from,
                "occurred_from",
            )

        if occurred_to is not None:
            _require_aware_datetime(
                occurred_to,
                "occurred_to",
            )

        if (
            occurred_from is not None
            and occurred_to is not None
            and occurred_to
            <= occurred_from
        ):
            raise ValueError(
                "occurred_to 必须晚于 "
                "occurred_from"
            )

        with self._lock:
            events = (
                self._read_all_unlocked()
            )

        matched_events: list[
            HealthEvent
        ] = []

        for event in events:
            if (
                event.user_id
                != normalized_user_id
            ):
                continue

            if (
                normalized_event_type
                is not None
                and event.event_type
                != normalized_event_type
            ):
                continue

            if (
                occurred_from
                is not None
                and event.occurred_at
                < occurred_from
            ):
                continue

            if (
                occurred_to
                is not None
                and event.occurred_at
                >= occurred_to
            ):
                continue

            matched_events.append(event)

        return sorted(
            matched_events,
            key=lambda event: (
                event.occurred_at,
                event.created_at,
                str(event.event_id),
            ),
            reverse=newest_first,
        )

    def update(
        self,
        *,
        event_id: UUID,
        replacement: HealthEvent,
        expected_updated_at: datetime,
    ) -> HealthEvent:
        """
        原子更新一个事件。

        expected_updated_at 是调用方读取到的旧版本时间。
        如果文件中的事件已经变化，则拒绝覆盖。
        """

        _require_aware_datetime(
            expected_updated_at,
            "expected_updated_at",
        )

        with self._lock:
            events = (
                self._read_all_unlocked()
            )

            target_index: int | None = (
                None
            )

            for index, current in enumerate(
                events
            ):
                if (
                    current.event_id
                    == event_id
                ):
                    target_index = index
                    break

            if target_index is None:
                raise (
                    HealthEventNotFoundError(
                        event_id
                    )
                )

            current = events[
                target_index
            ]

            if (
                current.updated_at
                != expected_updated_at
            ):
                raise (
                    HealthEventConflictError(
                        event_id
                    )
                )

            if (
                replacement.event_id
                != current.event_id
            ):
                raise ValueError(
                    "更新时不得修改 event_id"
                )

            if (
                replacement.user_id
                != current.user_id
            ):
                raise ValueError(
                    "更新时不得修改 user_id"
                )

            if (
                replacement.created_at
                != current.created_at
            ):
                raise ValueError(
                    "更新时不得修改 created_at"
                )

            if (
                replacement.updated_at
                <= current.updated_at
            ):
                raise ValueError(
                    "replacement.updated_at "
                    "必须晚于旧版本 updated_at"
                )

            events[
                target_index
            ] = replacement

            self._atomic_rewrite_unlocked(
                events
            )

            return replacement

    def delete(
        self,
        *,
        event_id: UUID,
        expected_updated_at: datetime,
    ) -> HealthEvent:
        """
        原子删除一个事件。

        返回被删除的完整事件，方便调用方展示审计信息。
        """

        _require_aware_datetime(
            expected_updated_at,
            "expected_updated_at",
        )

        with self._lock:
            events = (
                self._read_all_unlocked()
            )

            target_index: int | None = (
                None
            )

            for index, current in enumerate(
                events
            ):
                if (
                    current.event_id
                    == event_id
                ):
                    target_index = index
                    break

            if target_index is None:
                raise (
                    HealthEventNotFoundError(
                        event_id
                    )
                )

            current = events[
                target_index
            ]

            if (
                current.updated_at
                != expected_updated_at
            ):
                raise (
                    HealthEventConflictError(
                        event_id
                    )
                )

            remaining_events = (
                events[:target_index]
                + events[
                    target_index + 1:
                ]
            )

            self._atomic_rewrite_unlocked(
                remaining_events
            )

            return current

    def _atomic_rewrite_unlocked(
        self,
        events: list[HealthEvent],
    ) -> None:
        """
        通过同目录临时文件原子替换 JSONL。

        在 os.replace 成功前，原文件保持不变。
        """

        serialized = "".join(
            event.model_dump_json()
            + "\n"
            for event in events
        ).encode("utf-8")

        temporary_path: (
            Path
            | None
        ) = None

        try:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with (
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=(
                        f".{self.path.name}."
                    ),
                    suffix=".tmp",
                    delete=False,
                )
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                os.chmod(
                    temporary_path,
                    0o600,
                )

                written = (
                    temporary_file.write(
                        serialized
                    )
                )

                if written != len(
                    serialized
                ):
                    raise OSError(
                        f"仅写入 {written} 字节，"
                        f"预期写入 "
                        f"{len(serialized)} 字节"
                    )

                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.replace(
                temporary_path,
                self.path,
            )

            temporary_path = None
        except OSError as exc:
            raise JsonlWriteError(
                "原子更新健康事件文件失败："
                f"{exc}"
            ) from exc
        finally:
            if (
                temporary_path
                is not None
                and temporary_path.exists()
            ):
                try:
                    temporary_path.unlink()
                except OSError:
                    pass