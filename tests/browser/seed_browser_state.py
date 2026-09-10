"""为浏览器 E2E 创建隔离且可预测的本地状态。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.health.models import HealthEvent
from src.healthos.memory_control import sync_profile_memories
from src.healthos.models import CoachStyle, UserProfile
from src.storage.sqlite_store import (
    SQLiteDatabase,
    SQLiteHealthEventStore,
    SQLiteHealthOSStore,
)


USER_ID = "local-demo-user"


def seed(database_path: Path) -> None:
    database = SQLiteDatabase(database_path)
    event_store = SQLiteHealthEventStore(database)
    healthos_store = SQLiteHealthOSStore(database)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0)

    event_store.append(
        HealthEvent.model_validate(
            {
                "schema_version": "1.1",
                "event_id": str(UUID("22222222-2222-2222-2222-222222222222")),
                "user_id": USER_ID,
                "event_type": "water",
                "occurred_at": now,
                "payload": {"amount_ml": 350, "beverage": "水", "note": ""},
                "source_refs": [],
                "input_source": "chat",
                "created_at": now,
                "updated_at": now,
            }
        )
    )

    def add_profile(state: object) -> None:
        profile = UserProfile(
            user_id=USER_ID,
            timezone_name="Asia/Shanghai",
            coach_style=CoachStyle.RATIONAL,
            dietary_preferences=["少油"],
            exclusions=["花生"],
            updated_at=now,
        )
        state.profiles[USER_ID] = profile  # type: ignore[attr-defined]
        sync_profile_memories(state, profile)

    healthos_store.update(add_profile)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: seed_browser_state.py <sqlite-path>")
    seed(Path(sys.argv[1]))
