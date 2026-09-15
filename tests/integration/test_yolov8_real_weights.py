"""真实权重冒烟：加载导出的 ONNX，跑一次真实推理。

PRD 要求"必须保留一次真实权重冒烟"。权重不随仓库分发，所以没有
runtime/models/yolov8n.onnx 时整个模块跳过；CI 与本地都用同一条命令：

    python scripts/prepare_food_detection_model.py
    MEAL_DETECTION_MODE=onnx python -m pytest tests/integration -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tools.detect_food import detect_food
from src.vision.config import FoodDetectionConfig
from src.vision.label_map import DetectionLabelMap
from src.vision.onnx_detector import OnnxFoodDetector
from src.vision.postprocess import decode_predictions, to_model_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "runtime" / "models" / "yolov8n.onnx"
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "meal.png"
REAL_PHOTO_DIR = PROJECT_ROOT / "tests" / "fixtures" / "detection"

COCO_CLASS_COUNT = 80
YOLOV8_ANCHORS = 8400

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.is_file(),
    reason=(
        "没有 runtime/models/yolov8n.onnx；"
        "先运行 scripts/prepare_food_detection_model.py"
    ),
)


@pytest.fixture(scope="module")
def detector() -> OnnxFoodDetector:
    return OnnxFoodDetector(WEIGHTS_PATH, min_confidence=0.35)


def test_manifest_pins_the_exported_weights() -> None:
    manifest_path = WEIGHTS_PATH.parent / "detection_manifest.json"

    assert manifest_path.is_file(), "导出脚本必须同时写出清单"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["input_size"] == 640
    assert manifest["class_count"] == COCO_CLASS_COUNT
    assert len(manifest["weights_sha256"]) == 64
    assert manifest["source_url"].startswith(
        "https://github.com/ultralytics/assets/releases/download/"
    )


def test_model_version_is_stable_across_calls(
    detector: OnnxFoodDetector,
) -> None:
    assert detector.model_version == detector.model_version
    assert "yolov8n@" in detector.model_version


def test_real_graph_has_the_expected_output_shape(
    detector: OnnxFoodDetector,
) -> None:
    """(1, 4+80, 8400)：输入张量和图的契约对得上。"""

    session = detector._get_session()
    image = np.asarray(Image.open(FIXTURE_IMAGE).convert("RGB"))
    tensor, _, _, _ = to_model_input(image, 640)

    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    assert raw.shape == (1, 4 + COCO_CLASS_COUNT, YOLOV8_ANCHORS)


def test_decode_produces_well_formed_boxes_from_real_output(
    detector: OnnxFoodDetector,
) -> None:
    """把阈值放到 0，真实张量必须能解码出结构合法的框。

    这一条把"权重坏了"和"这张图里确实没有食物"区分开：
    解码路径本身在真实输出上必须工作。
    """

    session = detector._get_session()
    image = np.asarray(Image.open(FIXTURE_IMAGE).convert("RGB"))
    height, width = image.shape[:2]
    tensor, scale, pad_x, pad_y = to_model_input(image, 640)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    all_classes = decode_predictions(
        raw,
        keep_class_ids=frozenset(range(COCO_CLASS_COUNT)),
        min_confidence=0.0,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        image_width=width,
        image_height=height,
    )

    assert all_classes, "真实输出在零阈值下必须能解码出候选框"

    for class_id, confidence, box in all_classes:
        assert 0 <= class_id < COCO_CLASS_COUNT
        assert 0.0 <= confidence <= 1.0
        assert 0 <= box[0] <= box[2] <= width
        assert 0 <= box[1] <= box[3] <= height

    # 置信度必须单调不增。
    scores = [item[1] for item in all_classes]
    assert scores == sorted(scores, reverse=True)


def test_food_class_filter_narrows_real_output(
    detector: OnnxFoodDetector,
) -> None:
    """同一张真实输出，限定食物类后只能是全部类别的子集。"""

    session = detector._get_session()
    image = np.asarray(Image.open(FIXTURE_IMAGE).convert("RGB"))
    height, width = image.shape[:2]
    tensor, scale, pad_x, pad_y = to_model_input(image, 640)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    shared = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "image_width": width,
        "image_height": height,
        "min_confidence": 0.0,
    }
    food_class_ids = DetectionLabelMap.load().class_ids

    everything = decode_predictions(
        raw,
        keep_class_ids=frozenset(range(COCO_CLASS_COUNT)),
        **shared,
    )
    food_only = decode_predictions(
        raw,
        keep_class_ids=food_class_ids,
        **shared,
    )

    assert len(food_only) <= len(everything)
    assert {item[0] for item in food_only} <= food_class_ids


def test_real_detection_returns_a_valid_result(
    detector: OnnxFoodDetector,
) -> None:
    """合成夹具里没有真实食物，因此这里只校验契约，不断言识别出什么。"""

    result = detector.detect(FIXTURE_IMAGE)

    assert result.mode == "model"
    assert result.status in {"detected", "no_detection"}
    assert result.elapsed_ms > 0
    assert result.min_confidence == 0.35

    for detection in result.detections:
        assert detection.class_id in DetectionLabelMap.load().class_ids
        assert detection.confidence >= 0.35
        assert detection.suggested_query


def test_tool_contract_holds_against_real_weights(tmp_path: Path) -> None:
    from src.storage.trace_store import TraceStore

    config = FoodDetectionConfig(mode="onnx", model_path=WEIGHTS_PATH)

    assert config.available is True

    result = detect_food(
        FIXTURE_IMAGE,
        config=config,
        trace_store=TraceStore(tmp_path / "detection_traces.jsonl"),
    )

    assert result["ok"] is True
    data = result["data"]
    assert data["mode"] == "model"
    assert data["candidate_source"] == "model"
    assert "yolov8n@" in data["model_version"]
    assert data["trace"]["model_version"] == data["model_version"]


def _real_photos() -> list[Path]:
    if not REAL_PHOTO_DIR.is_dir():
        return []
    return sorted(
        path
        for path in REAL_PHOTO_DIR.iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
    )


@pytest.mark.skipif(
    not _real_photos(),
    reason=(
        "tests/fixtures/detection/ 里没有真实照片；"
        "放入以 COCO 食物类英文名命名的照片（如 banana.jpg）即可启用"
    ),
)
@pytest.mark.parametrize(
    "photo",
    _real_photos(),
    ids=lambda path: path.stem,
)
def test_real_photos_detect_their_expected_label(
    detector: OnnxFoodDetector,
    photo: Path,
) -> None:
    """可选的识别质量检查：文件名即期望的 COCO 标签。"""

    expected = photo.stem.split("-")[0].replace("_", " ")
    result = detector.detect(photo)

    assert result.status == "detected", f"{photo.name} 没有识别出任何食物"
    assert expected in {
        detection.label for detection in result.detections
    }, (photo.name, [d.label for d in result.detections])
