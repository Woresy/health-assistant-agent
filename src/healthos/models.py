"""HealthOS P1 档案、目标与提醒领域模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthOSModel(BaseModel):
    """P1 领域模型的严格公共配置。"""

    model_config = ConfigDict(extra="forbid")


class CoachStyle(str, Enum):
    GENTLE = "gentle"
    RATIONAL = "rational"
    CONCISE = "concise"
    GOAL_FOCUSED = "goal_focused"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class ReminderStatus(str, Enum):
    SCHEDULED = "scheduled"
    FIRED = "fired"
    COMPLETED = "completed"
    SNOOZED = "snoozed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"


class UserProfile(HealthOSModel):
    """只保存用户明确确认的最小档案。"""

    user_id: str = Field(min_length=1, max_length=128)
    timezone_name: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)
    unit_system: Literal["metric"] = "metric"
    coach_style: CoachStyle = CoachStyle.GENTLE
    dietary_preferences: list[str] = Field(default_factory=list, max_length=20)
    exclusions: list[str] = Field(default_factory=list, max_length=20)
    reminders_enabled: bool = True
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    version: int = Field(default=1, ge=1)
    updated_at: datetime

    @field_validator("user_id", "timezone_name", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("coach_style", mode="before")
    @classmethod
    def normalize_coach_style(cls, value: Any) -> Any:
        """兼容模型和用户常用的中英文风格表达。"""

        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "encouraging": "gentle",
            "warm": "gentle",
            "supportive": "gentle",
            "温和": "gentle",
            "温暖": "gentle",
            "鼓励": "gentle",
            "温和陪伴": "gentle",
            "analytical": "rational",
            "logical": "rational",
            "理性": "rational",
            "理性复盘": "rational",
            "brief": "concise",
            "simple": "concise",
            "简洁": "concise",
            "简洁提醒": "concise",
            "accountability": "goal_focused",
            "strict": "goal_focused",
            "目标督促": "goal_focused",
        }
        return aliases.get(normalized, normalized)

    @field_validator("dietary_preferences", "exclusions", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("偏好列表只能包含非空文本")
            text = item.strip()
            if len(text) > 80:
                raise ValueError("单条偏好不得超过 80 个字符")
            if text not in normalized:
                normalized.append(text)
        return normalized

    @field_validator("quiet_hours_start", "quiet_hours_end")
    @classmethod
    def validate_clock_time(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        try:
            datetime.strptime(normalized, "%H:%M")
        except ValueError as exc:
            raise ValueError("免打扰时间必须使用 HH:MM") from exc
        return normalized

    @field_validator("updated_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at 必须包含时区")
        return value

    @model_validator(mode="after")
    def validate_quiet_hours_pair(self) -> "UserProfile":
        if (self.quiet_hours_start is None) != (self.quiet_hours_end is None):
            raise ValueError("免打扰开始和结束时间必须同时提供")
        return self


class GoalVersion(HealthOSModel):
    """健康目标的一个不可变版本。"""

    version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=120)
    goal_type: Literal["water", "exercise", "weight", "nutrition", "custom"]
    target_value: float = Field(gt=0, le=1_000_000)
    unit: str = Field(min_length=1, max_length=30)
    period: Literal["daily", "weekly", "monthly", "8_weeks"]
    status: GoalStatus = GoalStatus.ACTIVE
    reason: str = Field(min_length=1, max_length=500)
    created_at: datetime

    @field_validator("title", "unit", "reason", mode="before")
    @classmethod
    def strip_goal_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("created_at")
    @classmethod
    def require_goal_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at 必须包含时区")
        return value


class HealthGoal(HealthOSModel):
    """带完整版本历史的用户健康目标。"""

    goal_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    versions: list[GoalVersion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_versions(self) -> "HealthGoal":
        expected = list(range(1, len(self.versions) + 1))
        actual = [version.version for version in self.versions]
        if actual != expected:
            raise ValueError("目标版本必须从 1 连续递增")
        return self

    @property
    def current(self) -> GoalVersion:
        return self.versions[-1]


class ReminderTransition(HealthOSModel):
    """提醒状态的一次可追溯变化。"""

    status: ReminderStatus
    occurred_at: datetime
    reason: str = Field(min_length=1, max_length=500)


class Reminder(HealthOSModel):
    """本地模拟 Provider 中的一条提醒。"""

    reminder_id: UUID
    user_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=300)
    scheduled_for: datetime
    timezone_name: str = Field(min_length=1, max_length=100)
    status: ReminderStatus = ReminderStatus.SCHEDULED
    created_at: datetime
    updated_at: datetime
    transitions: list[ReminderTransition] = Field(min_length=1)

    @field_validator("content", "timezone_name", mode="before")
    @classmethod
    def strip_reminder_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("scheduled_for", "created_at", "updated_at")
    @classmethod
    def require_reminder_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("提醒时间必须包含时区")
        return value


class HealthOSState(HealthOSModel):
    """HealthOS P1 本地持久化快照。"""

    schema_version: Literal["1.0"] = "1.0"
    profiles: dict[str, UserProfile] = Field(default_factory=dict)
    goals: list[HealthGoal] = Field(default_factory=list)
    reminders: list[Reminder] = Field(default_factory=list)
    idempotency_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
