"""餐食图片检测工具协议封装。

检测只产生"建议检索词"。候选确认、RAG 检索、营养计算和保存一步都不跳过，
识别失败时全部回退到手工填写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.storage.trace_store import TraceStore, TraceWriteError
from src.ui.image_input import validate_image
from src.vision.config import FoodDetectionConfig
from src.vision.detection_trace import (
    build_detection_trace,
    hash_image_file,
)
from src.vision.detector import DetectionError, FoodDetector
from src.vision.factory import build_food_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = (
    PROJECT_ROOT / "data" / "detection_traces.jsonl"
)

MAX_SUGGESTIONS = 5


def _failure(error_code: str, message: str) -> dict[str, Any]:
    """构造稳定的失败协议。"""

    return {
        "ok": False,
        "data": None,
        "error": {"error_code": error_code, "message": message},
    }


def detect_food(
    image_path: Any,
    detector: FoodDetector | None = None,
    config: FoodDetectionConfig | None = None,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """校验图片、运行检测并尽力追加 Trace。"""

    if image_path is not None and not isinstance(
        image_path, (str, Path)
    ):
        return _failure(
            "IMAGE_PATH_INVALID",
            "image_path 必须是字符串路径",
        )

    image_result = validate_image(image_path)

    if not image_result.ok:
        return _failure(
            image_result.error_code or "IMAGE_INVALID",
            image_result.message,
        )

    active_config = (
        config or FoodDetectionConfig.from_environment()
    )
    active_detector = detector or build_food_detector(
        active_config
    )

    if active_detector is None:
        return _failure(
            "DETECTION_UNAVAILABLE",
            active_config.unavailable_reason()
            or "图片识别当前不可用，请手动填写食物名称",
        )

    try:
        result = active_detector.detect(image_path)
    except DetectionError as exc:
        return _failure(exc.error_code, exc.message)

    try:
        image_sha256, image_bytes = hash_image_file(image_path)
    except OSError as exc:
        return _failure(
            "IMAGE_READ_FAILED",
            f"无法读取图片内容：{exc}",
        )

    trace = build_detection_trace(
        result,
        image_sha256=image_sha256,
        image_bytes=image_bytes,
    )
    trace_warning: dict[str, str] | None = None
    try:
        (
            trace_store or TraceStore(DEFAULT_TRACE_PATH)
        ).append(trace)
    except TraceWriteError as exc:
        trace_warning = {
            "error_code": exc.error_code,
            "message": exc.message,
        }

    detections = sorted(
        result.detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )[:MAX_SUGGESTIONS]

    top = detections[0] if detections else None

    return {
        "ok": True,
        "data": {
            "status": result.status,
            "mode": result.mode,
            "model_version": result.model_version,
            "candidate_source": "model",
            "min_confidence": result.min_confidence,
            "elapsed_ms": round(result.elapsed_ms, 2),
            "suggested_query": (
                top.suggested_query if top else None
            ),
            "detections": [
                detection.model_dump(mode="json")
                for detection in detections
            ],
            "trace": trace.model_dump(mode="json"),
            "trace_warning": trace_warning,
        },
        "error": None,
    }
