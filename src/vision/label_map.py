"""COCO 食物类到中文检索词的固定映射。

COCO 80 类里只有 10 个食物类，这就是本期识别能力的全部边界。
映射表是数据文件，不写死在代码里，便于换权重时一起替换。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.vision.detector import DetectionError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MAP_PATH = (
    PROJECT_ROOT / "data" / "detection_label_map.json"
)


class DetectionLabel(BaseModel):
    """一个可识别类别。"""

    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0)
    coco_label: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_food_id: str = Field(min_length=1)


class DetectionLabelMapFile(BaseModel):
    """映射表文件。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    note: str = ""
    labels: list[DetectionLabel] = Field(min_length=1)


class DetectionLabelMap:
    """按 class_id 查中文检索词。"""

    def __init__(
        self,
        labels: list[DetectionLabel],
    ) -> None:
        self._by_class_id = {
            label.class_id: label for label in labels
        }

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
    ) -> "DetectionLabelMap":
        map_path = Path(path or DEFAULT_LABEL_MAP_PATH)

        if not map_path.exists():
            raise DetectionError(
                "DETECTION_LABEL_MAP_MISSING",
                f"找不到检测类别映射表：{map_path}",
            )

        try:
            payload = json.loads(
                map_path.read_text(encoding="utf-8")
            )
            parsed = DetectionLabelMapFile.model_validate(payload)
        except (
            OSError,
            ValueError,
            ValidationError,
        ) as exc:
            raise DetectionError(
                "DETECTION_LABEL_MAP_INVALID",
                f"检测类别映射表无效：{exc}",
            ) from exc

        return cls(parsed.labels)

    @property
    def class_ids(self) -> frozenset[int]:
        """可识别的 class_id；其余 COCO 类一律丢弃。"""

        return frozenset(self._by_class_id)

    @property
    def labels(self) -> tuple[DetectionLabel, ...]:
        return tuple(
            self._by_class_id[class_id]
            for class_id in sorted(self._by_class_id)
        )

    def get(
        self,
        class_id: int,
    ) -> DetectionLabel | None:
        return self._by_class_id.get(class_id)
