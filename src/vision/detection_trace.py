"""餐食检测的证据 Trace。

只记录图片的摘要和尺寸，不记录图片内容、文件名或路径——
原图默认不保存，Trace 也不该成为绕过这条规则的后门。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.vision.models import DetectionResult


class DetectionTrace(BaseModel):
    """一次检测的可序列化证据。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: UUID
    created_at: datetime
    status: Literal["detected", "no_detection"]
    mode: Literal["model", "manual_fallback"]
    model_version: str
    image_sha256: str = Field(min_length=64, max_length=64)
    image_bytes: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    labels: list[str]
    confidences: list[float]
    min_confidence: float = Field(ge=0, le=1)
    elapsed_ms: float = Field(ge=0)
    strategies_used: list[str]


def hash_image_file(image_path: str | Path) -> tuple[str, int]:
    """返回图片内容摘要和字节数。"""

    path = Path(image_path)
    digest = hashlib.sha256()
    total = 0

    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)

    return digest.hexdigest(), total


def build_detection_trace(
    result: DetectionResult,
    *,
    image_sha256: str,
    image_bytes: int,
    strategies_used: list[str] | None = None,
) -> DetectionTrace:
    """从检测结果构建独立 Trace，不写入 HealthEvent。"""

    return DetectionTrace(
        trace_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        status=result.status,
        mode=result.mode,
        model_version=result.model_version,
        image_sha256=image_sha256,
        image_bytes=image_bytes,
        detection_count=len(result.detections),
        labels=[
            detection.label for detection in result.detections
        ],
        confidences=[
            round(detection.confidence, 4)
            for detection in result.detections
        ],
        min_confidence=result.min_confidence,
        elapsed_ms=result.elapsed_ms,
        strategies_used=strategies_used
        or ["yolov8n_onnx", "coco_food_class_filter", "nms"],
    )
