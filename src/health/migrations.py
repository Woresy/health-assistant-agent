"""HealthEvent 字典数据的版本迁移。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


CURRENT_HEALTH_EVENT_SCHEMA_VERSION = (
    "1.1"
)
LEGACY_HEALTH_EVENT_SCHEMA_VERSIONS = {
    "1.0",
}


class HealthEventMigrationError(
    ValueError
):
    """HealthEvent 版本无法迁移。"""


def migrate_health_event_data(
    raw_data: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将旧 HealthEvent 字典迁移到当前版本。

    当前支持：
    1.0 → 1.1

    迁移不会修改调用方传入的原始字典。
    """

    if not isinstance(
        raw_data,
        Mapping,
    ):
        raise HealthEventMigrationError(
            "HealthEvent 原始数据必须是对象"
        )

    migrated = deepcopy(
        dict(raw_data)
    )

    raw_version = migrated.get(
        "schema_version"
    )

    if raw_version is None:
        migrated[
            "schema_version"
        ] = (
            CURRENT_HEALTH_EVENT_SCHEMA_VERSION
        )
        return migrated

    version = str(raw_version).strip()

    if (
        version
        == CURRENT_HEALTH_EVENT_SCHEMA_VERSION
    ):
        return migrated

    if (
        version
        in LEGACY_HEALTH_EVENT_SCHEMA_VERSIONS
    ):
        migrated[
            "schema_version"
        ] = (
            CURRENT_HEALTH_EVENT_SCHEMA_VERSION
        )

        if (
            migrated.get(
                "input_source"
            )
            == "image_manual"
        ):
            migrated[
                "input_source"
            ] = "image"

        return migrated

    raise HealthEventMigrationError(
        "不支持的 HealthEvent "
        f"schema_version：{version}"
    )