"""准备健康事件更新或删除草稿，不执行写入。"""

from __future__ import annotations

from copy import deepcopy
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any
from uuid import (
    UUID,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from src.health.models import (
    EventType,
    HealthEvent,
    MealPayload,
)
from src.storage.jsonl_store import (
    HealthEventStore,
    JsonlReadError,
)
from src.tools.confirmation import (
    issue_delete_confirmation_token,
    issue_update_confirmation_token,
)


class UpdateHealthEventPatch(
    BaseModel
):
    """
    用户允许修改的健康事件字段。

    event_id、user_id、event_type、
    created_at 和 schema_version 不允许修改。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    occurred_at: (
        datetime
        | None
    ) = None
    payload: (
        dict[str, Any]
        | None
    ) = None
    source_refs: (
        list[str]
        | None
    ) = None

    @field_validator(
        "occurred_at",
    )
    @classmethod
    def validate_occurred_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None

        if (
            value.tzinfo is None
            or value.utcoffset()
            is None
        ):
            raise ValueError(
                "occurred_at 必须包含时区"
            )

        return value

    @model_validator(mode="after")
    def require_at_least_one_field(
        self,
    ) -> "UpdateHealthEventPatch":
        if not self.model_fields_set:
            raise ValueError(
                "更新 patch 至少包含一个字段"
            )

        return self


def _error(
    error_code: str,
    message: str,
) -> dict[str, Any]:
    """构造统一工具错误。"""

    return {
        "ok": False,
        "data": None,
        "error": {
            "error_code": error_code,
            "message": message,
        },
    }


def _parse_event_id(
    event_id: str | UUID,
) -> tuple[
    UUID | None,
    dict[str, Any] | None,
]:
    """把字符串事件 ID 转换为 UUID。"""

    try:
        parsed = (
            event_id
            if isinstance(
                event_id,
                UUID,
            )
            else UUID(
                str(event_id)
            )
        )
    except (
        ValueError,
        TypeError,
        AttributeError,
    ):
        return (
            None,
            _error(
                "EVENT_ID_INVALID",
                "event_id 必须是有效 UUID",
            ),
        )

    return (
        parsed,
        None,
    )


def _normalize_user_id(
    user_id: str,
) -> tuple[
    str | None,
    dict[str, Any] | None,
]:
    """清理并验证用户 ID。"""

    if not isinstance(
        user_id,
        str,
    ):
        return (
            None,
            _error(
                "USER_ID_REQUIRED",
                "user_id 必须是字符串",
            ),
        )

    normalized = user_id.strip()

    if not normalized:
        return (
            None,
            _error(
                "USER_ID_REQUIRED",
                "user_id 不能为空",
            ),
        )

    return (
        normalized,
        None,
    )


def _normalize_idempotency_key(
    idempotency_key: str,
) -> tuple[
    str | None,
    dict[str, Any] | None,
]:
    """验证幂等键。"""

    if not isinstance(
        idempotency_key,
        str,
    ):
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_REQUIRED",
                "idempotency_key "
                "必须是字符串",
            ),
        )

    normalized = (
        idempotency_key.strip()
    )

    if not normalized:
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_REQUIRED",
                "idempotency_key 不能为空",
            ),
        )

    if len(normalized) > 128:
        return (
            None,
            _error(
                "IDEMPOTENCY_KEY_INVALID",
                "idempotency_key "
                "长度不得超过 128",
            ),
        )

    return (
        normalized,
        None,
    )


def _deep_merge(
    original: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """
    递归合并字典。

    例如：
    原 payload:
    {"amount_ml": 300, "unit": "ml"}

    patch:
    {"amount_ml": 600}

    合并后:
    {"amount_ml": 600, "unit": "ml"}
    """

    merged = deepcopy(
        original
    )

    for key, patch_value in (
        patch.items()
    ):
        original_value = (
            merged.get(key)
        )

        if (
            isinstance(
                original_value,
                dict,
            )
            and isinstance(
                patch_value,
                dict,
            )
        ):
            merged[key] = _deep_merge(
                original_value,
                patch_value,
            )
        else:
            merged[key] = deepcopy(
                patch_value
            )

    return merged


def _logical_event_data(
    event: HealthEvent,
) -> dict[str, Any]:
    """
    获取不包含 updated_at 的业务数据。

    用于判断用户是否真的修改了内容。
    """

    data = event.model_dump(
        mode="json"
    )
    data.pop(
        "updated_at",
        None,
    )

    return data


def prepare_update_health_event(
    *,
    event_id: str | UUID,
    user_id: str,
    patch: (
        UpdateHealthEventPatch
        | dict[str, Any]
    ),
    idempotency_key: str,
    store: HealthEventStore,
    now: datetime | None = None,
    confirmation_ttl_seconds: int = 900,
) -> dict[str, Any]:
    """
    生成更新前后对比和确认令牌。

    本函数只读取 JSONL，不执行更新。
    """

    (
        parsed_event_id,
        event_id_error,
    ) = _parse_event_id(
        event_id
    )

    if event_id_error is not None:
        return event_id_error

    assert parsed_event_id is not None

    (
        normalized_user_id,
        user_id_error,
    ) = _normalize_user_id(
        user_id
    )

    if user_id_error is not None:
        return user_id_error

    assert (
        normalized_user_id
        is not None
    )

    (
        normalized_key,
        key_error,
    ) = _normalize_idempotency_key(
        idempotency_key
    )

    if key_error is not None:
        return key_error

    assert normalized_key is not None

    try:
        validated_patch = (
            patch
            if isinstance(
                patch,
                UpdateHealthEventPatch,
            )
            else (
                UpdateHealthEventPatch
                .model_validate(
                    patch
                )
            )
        )
    except ValidationError as exc:
        return _error(
            "PATCH_VALIDATION_ERROR",
            f"更新字段校验失败：{exc}",
        )

    try:
        current_event = (
            store.find_by_event_id(
                parsed_event_id
            )
        )
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )

    if (
        current_event is None
        or current_event.user_id
        != normalized_user_id
    ):
        return _error(
            "EVENT_NOT_FOUND",
            "没有找到可更新的健康事件",
        )

    patch_fields = (
        validated_patch
        .model_fields_set
    )

    if (
        current_event.event_type
        == EventType.MEAL
        and "payload"
        in patch_fields
    ):
        incoming_payload = (
            validated_patch.payload
        )

        required_meal_fields = set(
            MealPayload.model_fields
        )

        if (
            not isinstance(
                incoming_payload,
                dict,
            )
            or not (
                required_meal_fields
                .issubset(
                    incoming_payload
                    .keys()
                )
            )
            or "source_refs"
            not in patch_fields
        ):
            return _error(
                "MEAL_RECALCULATION_REQUIRED",
                "修改饮食的食物或份量时，"
                "必须先重新执行营养检索和"
                "确定性计算，然后提交完整的 "
                "meal payload 与 source_refs",
            )

    replacement_data = (
        current_event.model_dump(
            mode="python"
        )
    )

    if "occurred_at" in patch_fields:
        replacement_data[
            "occurred_at"
        ] = (
            validated_patch
            .occurred_at
        )

    if "payload" in patch_fields:
        if (
            validated_patch.payload
            is None
        ):
            replacement_data[
                "payload"
            ] = None
        else:
            current_payload = (
                replacement_data[
                    "payload"
                ]
            )

            if not isinstance(
                current_payload,
                dict,
            ):
                return _error(
                    "CURRENT_EVENT_INVALID",
                    "原事件 payload "
                    "不是有效对象",
                )

            replacement_data[
                "payload"
            ] = _deep_merge(
                current_payload,
                validated_patch.payload,
            )

    if "source_refs" in patch_fields:
        replacement_data[
            "source_refs"
        ] = (
            validated_patch
            .source_refs
        )

    effective_now = (
        now
        if now is not None
        else datetime.now(
            timezone.utc
        )
    )

    if (
        effective_now.tzinfo
        is None
        or effective_now.utcoffset()
        is None
    ):
        return _error(
            "TIME_INVALID",
            "now 必须包含时区",
        )

    if (
        effective_now
        <= current_event.updated_at
    ):
        effective_now = (
            current_event.updated_at
            + timedelta(
                microseconds=1
            )
        )

    replacement_data[
        "updated_at"
    ] = effective_now

    try:
        replacement_event = (
            HealthEvent.model_validate(
                replacement_data
            )
        )
    except ValidationError as exc:
        return _error(
            "VALIDATION_ERROR",
            "更新后的健康事件"
            f"校验失败：{exc}",
        )

    if (
        _logical_event_data(
            replacement_event
        )
        == _logical_event_data(
            current_event
        )
    ):
        return _error(
            "NO_CHANGES",
            "更新内容与当前记录相同",
        )

    try:
        confirmation_token = (
            issue_update_confirmation_token(
                current_event=(
                    current_event
                ),
                replacement_event=(
                    replacement_event
                ),
                idempotency_key=(
                    normalized_key
                ),
                ttl_seconds=(
                    confirmation_ttl_seconds
                ),
            )
        )
    except ValueError as exc:
        return _error(
            "CONFIRMATION_DRAFT_INVALID",
            str(exc),
        )

    return {
        "ok": True,
        "data": {
            "action": "update",
            "event_id": str(
                current_event.event_id
            ),
            "current_event": (
                current_event.model_dump(
                    mode="json"
                )
            ),
            "proposed_event": (
                replacement_event
                .model_dump(
                    mode="json"
                )
            ),
            "confirmation_token": (
                confirmation_token
            ),
            "idempotency_key": (
                normalized_key
            ),
            "confirmation_ttl_seconds": (
                confirmation_ttl_seconds
            ),
        },
        "error": None,
    }


def prepare_delete_health_event(
    *,
    event_id: str | UUID,
    user_id: str,
    idempotency_key: str,
    store: HealthEventStore,
    confirmation_ttl_seconds: int = 900,
) -> dict[str, Any]:
    """
    生成待删除事件预览和确认令牌。

    本函数只读取事件，不执行删除。
    """

    (
        parsed_event_id,
        event_id_error,
    ) = _parse_event_id(
        event_id
    )

    if event_id_error is not None:
        return event_id_error

    assert parsed_event_id is not None

    (
        normalized_user_id,
        user_id_error,
    ) = _normalize_user_id(
        user_id
    )

    if user_id_error is not None:
        return user_id_error

    assert (
        normalized_user_id
        is not None
    )

    (
        normalized_key,
        key_error,
    ) = _normalize_idempotency_key(
        idempotency_key
    )

    if key_error is not None:
        return key_error

    assert normalized_key is not None

    try:
        current_event = (
            store.find_by_event_id(
                parsed_event_id
            )
        )
    except JsonlReadError as exc:
        return _error(
            "STORAGE_READ_FAILED",
            str(exc),
        )

    if (
        current_event is None
        or current_event.user_id
        != normalized_user_id
    ):
        return _error(
            "EVENT_NOT_FOUND",
            "没有找到可删除的健康事件",
        )

    try:
        confirmation_token = (
            issue_delete_confirmation_token(
                current_event=(
                    current_event
                ),
                idempotency_key=(
                    normalized_key
                ),
                ttl_seconds=(
                    confirmation_ttl_seconds
                ),
            )
        )
    except ValueError as exc:
        return _error(
            "CONFIRMATION_DRAFT_INVALID",
            str(exc),
        )

    return {
        "ok": True,
        "data": {
            "action": "delete",
            "event_id": str(
                current_event.event_id
            ),
            "target_event": (
                current_event.model_dump(
                    mode="json"
                )
            ),
            "confirmation_token": (
                confirmation_token
            ),
            "idempotency_key": (
                normalized_key
            ),
            "confirmation_ttl_seconds": (
                confirmation_ttl_seconds
            ),
        },
        "error": None,
    }