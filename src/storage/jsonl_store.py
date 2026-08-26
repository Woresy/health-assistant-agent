"""单进程 HealthEvent JSONL 存储。"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from src.health.models import HealthEvent


class JsonlReadError(Exception):
    """读取 JSONL 时报告精确坏行位置。"""

    def __init__(self, line_number: int, reason: str) -> None:
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"JSONL 第 {line_number} 行损坏：{reason}")


class JsonlWriteError(Exception):
    """JSONL 追加写入失败。"""


class HealthEventStore:
    """适用于当天单进程演示范围的追加式事件存储。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: HealthEvent) -> None:
        """
        一次写入一整行。

        写入前事件已经通过 Pydantic 校验。发生异常时尽力恢复到原长度，
        避免留下半行或错误的成功状态。
        """

        line = (event.model_dump_json() + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        existed_before = self.path.exists()
        original_size = self.path.stat().st_size if existed_before else 0
        file_descriptor: int | None = None

        try:
            file_descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            written = os.write(file_descriptor, line)
            if written != len(line):
                raise OSError(
                    f"仅写入 {written} 字节，预期写入 {len(line)} 字节"
                )
            os.fsync(file_descriptor)
        except OSError as exc:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                finally:
                    file_descriptor = None

            try:
                if self.path.exists():
                    if existed_before:
                        with self.path.open("r+b") as rollback_file:
                            rollback_file.truncate(original_size)
                            rollback_file.flush()
                            os.fsync(rollback_file.fileno())
                    else:
                        self.path.unlink()
            except OSError:
                # 仍然抛出原始写入失败，调用方不得展示成功。
                pass

            raise JsonlWriteError(f"写入健康事件失败：{exc}") from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)

    def read_all(self) -> list[HealthEvent]:
        """读取全部事件；空文件返回空列表，坏行立即报告行号。"""

        if not self.path.exists():
            return []

        events: list[HealthEvent] = []

        try:
            with self.path.open("r", encoding="utf-8") as source_file:
                for line_number, raw_line in enumerate(source_file, start=1):
                    if not raw_line.strip():
                        raise JsonlReadError(line_number, "空行不是有效健康事件")

                    try:
                        event = HealthEvent.model_validate_json(raw_line)
                    except ValidationError as exc:
                        raise JsonlReadError(
                            line_number,
                            str(exc),
                        ) from exc

                    events.append(event)
        except JsonlReadError:
            raise
        except OSError as exc:
            raise JsonlReadError(0, f"读取文件失败：{exc}") from exc

        return events

    def find_by_event_id(self, event_id: UUID) -> HealthEvent | None:
        """通过稳定事件 ID 检查幂等提交。"""

        for event in self.read_all():
            if event.event_id == event_id:
                return event
        return None