"""餐食检测的纯逻辑测试。

全部不需要真实权重、不需要 onnxruntime：解码和 NMS 用合成张量验证，
可用性用配置对象验证，映射表用食物库交叉验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.nutrition.repository import FoodRepository
from src.tools.detect_food import detect_food
from src.vision.config import FoodDetectionConfig
from src.vision.detection_trace import (
    build_detection_trace,
    hash_image_file,
)
from src.vision.detector import DetectionError
from src.vision.label_map import DetectionLabelMap
from src.vision.models import (
    BoundingBox,
    DetectionResult,
    FoodDetection,
)
from src.vision.postprocess import (
    decode_predictions,
    letterbox,
    non_max_suppression,
    to_model_input,
    xywh_to_xyxy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOODS_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "detection_label_map.json"
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "meal.png"

APPLE_CLASS_ID = 47
BANANA_CLASS_ID = 46
PERSON_CLASS_ID = 0
COCO_CLASS_COUNT = 80


def build_raw_output(
    boxes: list[tuple[int, float, tuple[float, float, float, float]]],
    anchors: int = 8,
) -> np.ndarray:
    """构造一个形状与 YOLOv8 导出一致的假输出 (1, 84, anchors)。"""

    output = np.zeros((1, 4 + COCO_CLASS_COUNT, anchors), dtype=np.float32)

    for index, (class_id, score, box) in enumerate(boxes):
        output[0, 0, index] = box[0]
        output[0, 1, index] = box[1]
        output[0, 2, index] = box[2]
        output[0, 3, index] = box[3]
        output[0, 4 + class_id, index] = score

    return output


def test_letterbox_keeps_aspect_ratio_and_centers_padding() -> None:
    image = np.full((200, 400, 3), 7, dtype=np.uint8)

    canvas, scale, pad_x, pad_y = letterbox(image, 640)

    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(1.6)
    # 宽度铺满，高度上下各留一半填充。
    assert pad_x == 0.0
    assert pad_y == 160.0
    # 填充区是 114，不是原图内容。
    assert canvas[0, 0, 0] == 114
    assert canvas[320, 320, 0] == 7


def test_to_model_input_produces_normalized_nchw_tensor() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    tensor, scale, _, _ = to_model_input(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert scale == pytest.approx(6.4)


def test_xywh_to_xyxy_converts_center_boxes() -> None:
    boxes = np.array([[50.0, 60.0, 20.0, 40.0]], dtype=np.float32)

    converted = xywh_to_xyxy(boxes)

    assert converted.tolist() == [[40.0, 40.0, 60.0, 80.0]]


def test_nms_drops_overlapping_boxes_and_keeps_distinct_ones() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [100.0, 100.0, 110.0, 110.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    kept = non_max_suppression(boxes, scores, iou_threshold=0.45)

    assert kept == [0, 2]


def test_decode_restores_original_image_coordinates() -> None:
    # 400x200 的原图会被缩放 1.6 倍并在上下各填充 160 像素。
    raw = build_raw_output(
        [(APPLE_CLASS_ID, 0.9, (320.0, 320.0, 160.0, 160.0))]
    )

    decoded = decode_predictions(
        raw,
        keep_class_ids=frozenset({APPLE_CLASS_ID}),
        min_confidence=0.35,
        scale=1.6,
        pad_x=0.0,
        pad_y=160.0,
        image_width=400,
        image_height=200,
    )

    assert len(decoded) == 1
    class_id, confidence, box = decoded[0]
    assert class_id == APPLE_CLASS_ID
    assert confidence == pytest.approx(0.9)
    assert box == pytest.approx((150.0, 50.0, 250.0, 150.0))


def test_decode_discards_non_food_classes() -> None:
    raw = build_raw_output(
        [
            (PERSON_CLASS_ID, 0.99, (320.0, 320.0, 100.0, 100.0)),
            (APPLE_CLASS_ID, 0.60, (100.0, 100.0, 50.0, 50.0)),
        ]
    )

    decoded = decode_predictions(
        raw,
        keep_class_ids=frozenset({APPLE_CLASS_ID, BANANA_CLASS_ID}),
        min_confidence=0.35,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        image_width=640,
        image_height=640,
    )

    # 置信度 0.99 的 person 不能因为分高就被上报。
    assert [item[0] for item in decoded] == [APPLE_CLASS_ID]


def test_decode_respects_min_confidence() -> None:
    raw = build_raw_output(
        [(APPLE_CLASS_ID, 0.20, (320.0, 320.0, 100.0, 100.0))]
    )

    decoded = decode_predictions(
        raw,
        keep_class_ids=frozenset({APPLE_CLASS_ID}),
        min_confidence=0.35,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        image_width=640,
        image_height=640,
    )

    assert decoded == []


def test_decode_sorts_by_confidence_descending() -> None:
    raw = build_raw_output(
        [
            (APPLE_CLASS_ID, 0.50, (100.0, 100.0, 40.0, 40.0)),
            (BANANA_CLASS_ID, 0.80, (300.0, 300.0, 40.0, 40.0)),
        ]
    )

    decoded = decode_predictions(
        raw,
        keep_class_ids=frozenset({APPLE_CLASS_ID, BANANA_CLASS_ID}),
        min_confidence=0.35,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        image_width=640,
        image_height=640,
    )

    assert [item[0] for item in decoded] == [
        BANANA_CLASS_ID,
        APPLE_CLASS_ID,
    ]


def test_every_mapped_label_resolves_to_a_real_food_row() -> None:
    """守卫：映射表不得指向食物库里不存在的 food_id。"""

    label_map = DetectionLabelMap.load(LABEL_MAP_PATH)
    repository = FoodRepository(FOODS_PATH, rag_mode="lexical")
    known_ids = {food.food_id for food in repository._foods}

    assert len(label_map.labels) == 10

    for label in label_map.labels:
        assert label.expected_food_id in known_ids, label.coco_label

        result = repository.search(label.query, top_k=3)
        assert result.status == "ok", label.query
        assert result.candidates[0].food_id == label.expected_food_id, (
            label.query,
            result.candidates[0].food_id,
        )


def test_label_map_class_ids_match_coco_food_classes() -> None:
    label_map = DetectionLabelMap.load(LABEL_MAP_PATH)

    assert label_map.class_ids == frozenset(range(46, 56))


def test_label_map_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(DetectionError) as missing:
        DetectionLabelMap.load(tmp_path / "absent.json")
    assert missing.value.error_code == "DETECTION_LABEL_MAP_MISSING"

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(DetectionError) as invalid:
        DetectionLabelMap.load(broken)
    assert invalid.value.error_code == "DETECTION_LABEL_MAP_INVALID"


def test_config_defaults_to_disabled_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MEAL_DETECTION_MODE",
        "MEAL_DETECTION_MODEL_PATH",
        "MEAL_DETECTION_MIN_CONFIDENCE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = FoodDetectionConfig.from_environment()

    assert config.mode == "disabled"
    assert config.enabled is False
    assert config.available is False
    assert "MEAL_DETECTION_MODE=vlm" in (config.unavailable_reason() or "")
    # 缺项说明必须告诉用户现在还能做什么。
    assert "手动填写" in (config.unavailable_reason() or "")


def test_config_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEAL_DETECTION_MODE", "tensorrt")

    assert FoodDetectionConfig.from_environment().mode == "disabled"


def test_config_clamps_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEAL_DETECTION_MIN_CONFIDENCE", "9.9")
    assert FoodDetectionConfig.from_environment().min_confidence == 0.95

    monkeypatch.setenv("MEAL_DETECTION_MIN_CONFIDENCE", "not-a-number")
    assert FoodDetectionConfig.from_environment().min_confidence == 0.35


def test_available_is_false_when_weights_are_missing(
    tmp_path: Path,
) -> None:
    config = FoodDetectionConfig(
        mode="onnx",
        model_path=tmp_path / "absent.onnx",
    )

    assert config.enabled is True
    assert config.weights_present is False
    assert config.available is False
    assert "找不到权重文件" in (config.unavailable_reason() or "")


def test_available_is_false_when_runtime_is_missing(
    tmp_path: Path,
) -> None:
    """装了权重但缺 onnxruntime 时不得报告为可用。"""

    weights = tmp_path / "yolov8n.onnx"
    weights.write_bytes(b"not-a-real-model")

    class RuntimeMissingConfig(FoodDetectionConfig):
        @property
        def runtime_importable(self) -> bool:
            return False

    config = RuntimeMissingConfig(mode="onnx", model_path=weights)

    assert config.weights_present is True
    assert config.available is False
    assert "缺少运行依赖" in (config.unavailable_reason() or "")
    assert "onnxruntime" in (config.unavailable_reason() or "")


def test_detect_food_rejects_bad_image_before_touching_the_model() -> None:
    assert detect_food(None)["error"]["error_code"] == "IMAGE_REQUIRED"
    assert detect_food(123)["error"]["error_code"] == "IMAGE_PATH_INVALID"
    assert (
        detect_food("/nonexistent/meal.png")["error"]["error_code"]
        == "IMAGE_FILE_MISSING"
    )


def test_detect_food_reports_unavailable_without_weights(
    tmp_path: Path,
) -> None:
    config = FoodDetectionConfig(
        mode="onnx",
        model_path=tmp_path / "absent.onnx",
    )

    result = detect_food(FIXTURE_IMAGE, config=config)

    assert result["ok"] is False
    assert result["error"]["error_code"] == "DETECTION_UNAVAILABLE"
    assert "手动填写" in result["error"]["message"]


def test_detection_trace_records_evidence_without_image_content(
    tmp_path: Path,
) -> None:
    result = DetectionResult(
        status="detected",
        mode="model",
        model_version="yolov8n@test",
        min_confidence=0.35,
        detections=[
            FoodDetection(
                class_id=APPLE_CLASS_ID,
                label="apple",
                label_zh="苹果",
                suggested_query="苹果",
                confidence=0.87,
                bbox=BoundingBox(x1=1, y1=2, x2=3, y2=4),
            )
        ],
        elapsed_ms=12.5,
    )
    image_sha256, image_bytes = hash_image_file(FIXTURE_IMAGE)

    trace = build_detection_trace(
        result,
        image_sha256=image_sha256,
        image_bytes=image_bytes,
    )
    payload = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)

    assert trace.detection_count == 1
    assert trace.labels == ["apple"]
    assert trace.confidences == [0.87]
    assert len(trace.image_sha256) == 64
    assert trace.image_bytes > 0
    # 图片路径和文件名都不得进入 Trace。
    assert "meal.png" not in payload
    assert str(FIXTURE_IMAGE) not in payload


def test_top_detection_picks_highest_confidence() -> None:
    result = DetectionResult(
        status="detected",
        mode="model",
        model_version="yolov8n@test",
        min_confidence=0.35,
        detections=[
            FoodDetection(
                class_id=APPLE_CLASS_ID,
                label="apple",
                label_zh="苹果",
                suggested_query="苹果",
                confidence=0.51,
                bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            ),
            FoodDetection(
                class_id=BANANA_CLASS_ID,
                label="banana",
                label_zh="香蕉",
                suggested_query="香蕉",
                confidence=0.77,
                bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            ),
        ],
        elapsed_ms=1.0,
    )

    top = result.top_detection
    assert top is not None
    assert top.label == "banana"
