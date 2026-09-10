#!/usr/bin/env python3
"""Create a transactionally consistent SQLite backup while HealthOS is running."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error(f"数据库不存在：{args.database}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"healthos-{timestamp}.db"

    with sqlite3.connect(args.database) as source:
        with sqlite3.connect(output) as destination:
            source.backup(destination)
            result = destination.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        output.unlink(missing_ok=True)
        raise RuntimeError("备份完整性检查失败")

    output.chmod(0o600)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
