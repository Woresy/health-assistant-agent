"""检索 Trace 的单进程 JSONL 追加存储。"""

from __future__ import annotations

import os
from pathlib import Path

from src.nutrition.retrieval_trace import RetrievalTrace


class TraceWriteError(Exception):
    """Trace 写入失败，携带稳定错误码。"""

    error_code = "TRACE_WRITE_FAILED"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class TraceStore:
    """追加完整 JSON 行；失败时尽力回滚半行。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trace: RetrievalTrace) -> None:
        """将一条已校验 Trace 原子式追加到本地文件。"""

        line = (trace.model_dump_json() + "\n").encode("utf-8")
        existed_before = self.path.exists()
        original_size = self.path.stat().st_size if existed_before else 0
        file_descriptor: int | None = None

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
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
                pass
            raise TraceWriteError(f"写入检索 Trace 失败：{exc}") from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
