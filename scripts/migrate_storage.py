#!/usr/bin/env python3
"""把旧 JSON/JSONL 数据非破坏地迁移到本地 SQLite。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.sqlite_store import SQLiteDatabase, migrate_legacy_storage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移并验证 HealthOS 本地存储")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "healthos.db",
        help="目标 SQLite 文件",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只执行完整性与记录数检查",
    )
    args = parser.parse_args()

    database = SQLiteDatabase(args.database)
    migrated = {"health_events": 0, "healthos_entities": 0, "conversations": 0}
    if not args.verify_only:
        migrated = migrate_legacy_storage(
            database,
            events_path=PROJECT_ROOT / "data" / "health_events.jsonl",
            healthos_path=PROJECT_ROOT / "data" / "healthos_state.json",
            conversations_path=PROJECT_ROOT / "data" / "conversations",
        )
    report = {
        "database": str(args.database),
        "migrated": migrated,
        "verification": database.integrity_check(),
        "rollback": "设置 STORAGE_BACKEND=json；旧 JSON/JSONL 未被修改或删除。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verification"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
