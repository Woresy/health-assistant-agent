"""识别预填的纵向链路 E2E；不启动浏览器，不需要真实权重。

检测器通过构造函数注入，和 tests/unit/test_hybrid_retrieval.py 里
FakeDenseRetriever / FailingDenseRetriever 的做法一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage.trace_store import TraceStore
from src.tools.detect_food import detect_food
from src.ui.app import (
    _format_detection_status,
    _resolve_candidate_source,
    calculate_meal_preview,
)
from src.vision.config import FoodDetectionConfig
from src.vision.detector import DetectionError
from src.vision.models import (
    BoundingBox,
    DetectionResult,
    FoodDetection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "meal.png"

ENABLED_CONFIG = FoodDetectionConfig(
    mode="onnx",
    model_path=FIXTURE_IMAGE,  # 只需存在，注入检测器后不会真的加载
    min_confidence=0.35,
)


class FakeFoodDetector:
    """返回固定检测结果的替身，只用于测试夹具。"""

    def __init__(
        self,
        detections: list[tuple[int, str, str, float]],
    ) -> None:
        self._detections = detections

    @property
    def model_version(self) -> str:
        return "yolov8n@fake"

    def detect(self, image_path: str | Path) -> DetectionResult:
        detections = [
            FoodDetection(
                class_id=class_id,
                label=label,
                label_zh=label_zh,
                suggested_query=label_zh,
                confidence=confidence,
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )
            for class_id, label, label_zh, confidence in self._detections
        ]

        return DetectionResult(
            status="detected" if detections else "no_detection",
            mode="model",
            model_version=self.model_version,
            min_confidence=0.35,
            detections=detections,
            elapsed_ms=8.0,
        )


class FailingFoodDetector:
    """模拟权重损坏；上层必须回退手填而不是崩溃。"""

    @property
    def model_version(self) -> str:
        return "yolov8n@broken"

    def detect(self, image_path: str | Path) -> DetectionResult:
        raise DetectionError(
            "DETECTION_INFERENCE_FAILED",
            "模拟推理失败",
        )


def apple_detector() -> FakeFoodDetector:
    return FakeFoodDetector([(47, "apple", "苹果", 0.91)])


def test_detection_prefills_name_and_saves_as_model_source(
    tmp_path: Path,
) -> None:
    """识别 → 预填名称 → 用户填份量 → 计算 → candidate_source=model。"""

    result = detect_food(
        FIXTURE_IMAGE,
        detector=apple_detector(),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(tmp_path / "detection_traces.jsonl"),
    )

    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == "detected"
    assert data["mode"] == "model"
    assert data["candidate_source"] == "model"
    assert data["model_version"] == "yolov8n@fake"
    assert data["suggested_query"] == "苹果"

    status, prefilled_query = _format_detection_status(data)

    assert prefilled_query == "苹果"
    assert "已识别出**苹果**" in status
    assert "91%" in status
    assert "请核对后填写份量" in status

    # 份量不会被预填，必须由用户提供。
    summary, preview_state, calc_status, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        prefilled_query,
        "FOOD_005",
        180,
        prefilled_query,
    )

    assert preview_state is not None
    assert "大约" in summary and "千卡" in summary
    # 保存前必须说清还没记下来，以及这不是医疗建议。
    assert "点「保存这一餐」才会记下来" in summary
    assert "不是医疗建议" in summary
    assert "计算完成" in calc_status

    payload = preview_state["event"]["payload"]
    assert payload["candidate_source"] == "model"
    assert payload["food"]["food_id"] == "FOOD_005"
    assert payload["portion"]["grams"] == 180.0
    # 营养值仍然来自食物数据行，不来自检测。
    assert payload["nutrition"]["calories_kcal"] == pytest.approx(95.4)


def test_editing_the_detected_name_falls_back_to_manual_source() -> None:
    """用户改写识别结果后，这条记录不能再算作模型候选。"""

    assert _resolve_candidate_source("苹果", "苹果") == "model"
    assert _resolve_candidate_source(" 苹果 ", "苹果") == "model"
    assert _resolve_candidate_source("香蕉", "苹果") == "manual"
    assert _resolve_candidate_source("苹果", "") == "manual"

    _, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "香蕉",
        "FOOD_006",
        100,
        "苹果",
    )

    assert preview_state is not None
    payload = preview_state["event"]["payload"]
    assert payload["candidate_source"] == "manual"
    assert payload["food"]["food_id"] == "FOOD_006"


def test_manual_flow_without_detection_keeps_manual_source() -> None:
    """没有识别时 detected_query 为空，链路与今天完全一致。"""

    _, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "西红柿",
        "FOOD_001",
        150,
    )

    assert preview_state is not None
    assert (
        preview_state["event"]["payload"]["candidate_source"] == "manual"
    )


def test_multiple_detections_prefill_top_one_and_list_the_rest(
    tmp_path: Path,
) -> None:
    result = detect_food(
        FIXTURE_IMAGE,
        detector=FakeFoodDetector(
            [
                (47, "apple", "苹果", 0.55),
                (46, "banana", "香蕉", 0.88),
                (50, "broccoli", "西兰花", 0.41),
            ]
        ),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(tmp_path / "detection_traces.jsonl"),
    )

    status, prefilled_query = _format_detection_status(result["data"])

    # 置信度最高的进输入框，其余只在文案里列出供用户改写。
    assert prefilled_query == "香蕉"
    assert "已识别出**香蕉**" in status
    assert "图中还检测到：苹果、西兰花" in status


def test_no_detection_tells_the_user_to_fill_it_in(
    tmp_path: Path,
) -> None:
    result = detect_food(
        FIXTURE_IMAGE,
        detector=FakeFoodDetector([]),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(tmp_path / "detection_traces.jsonl"),
    )

    assert result["ok"] is True
    assert result["data"]["status"] == "no_detection"
    assert result["data"]["detections"] == []
    assert result["data"]["suggested_query"] is None

    status, prefilled_query = _format_detection_status(result["data"])

    assert prefilled_query == ""
    assert "没有识别出可靠的食物" in status
    assert "请直接填写食物名称和份量" in status


def test_broken_weights_fall_back_to_manual_entry(
    tmp_path: Path,
) -> None:
    result = detect_food(
        FIXTURE_IMAGE,
        detector=FailingFoodDetector(),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(tmp_path / "detection_traces.jsonl"),
    )

    assert result["ok"] is False
    assert result["error"]["error_code"] == "DETECTION_INFERENCE_FAILED"

    # 识别失败不影响手填链路。
    _, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "西红柿",
        "FOOD_001",
        150,
        "",
    )

    assert preview_state is not None
    assert (
        preview_state["event"]["payload"]["candidate_source"] == "manual"
    )


def test_detection_appends_one_trace_line_without_image_content(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "detection_traces.jsonl"

    detect_food(
        FIXTURE_IMAGE,
        detector=apple_detector(),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(trace_path),
    )

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "detected"
    assert record["model_version"] == "yolov8n@fake"
    assert record["labels"] == ["apple"]
    assert len(record["image_sha256"]) == 64
    assert "meal.png" not in lines[0]
    assert str(FIXTURE_IMAGE) not in lines[0]


def test_trace_write_failure_does_not_fail_detection(
    tmp_path: Path,
) -> None:
    """Trace 写不进去只是警告，不能让识别结果作废。"""

    blocked = tmp_path / "blocked"
    blocked.mkdir()

    result = detect_food(
        FIXTURE_IMAGE,
        detector=apple_detector(),
        config=ENABLED_CONFIG,
        trace_store=TraceStore(blocked),
    )

    assert result["ok"] is True
    assert result["data"]["suggested_query"] == "苹果"
    assert result["data"]["trace_warning"] is not None


def test_missing_portion_gives_chinese_guidance_not_a_validation_error() -> None:
    """份量没填时要告诉用户填哪一格，而不是抛校验错误。

    Gradio 的 Number 组件如果设了 minimum，会抢在服务端之前弹出英文的
    "Value 0 is less than minimum value 0.01."，用户看不懂也不知道该填哪。
    """

    for blank in (0, 0.0, None, ""):
        summary, preview_state, _, _ = calculate_meal_preview(
            str(FIXTURE_IMAGE),
            "豆腐",
            "FOOD_010",
            blank,
        )

        assert preview_state is None, blank
        assert "还差一步" in summary
        assert "估计份量" in summary
        # 不能把内部校验错误码甩给用户。
        assert "PORTION_INVALID" not in summary
        assert "minimum value" not in summary


def test_missing_candidate_and_portion_are_reported_together() -> None:
    """两项都缺时一次说清，不要点一次补一项。"""

    summary, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "豆腐",
        None,
        0,
    )

    assert preview_state is None
    assert "还差 2 项" in summary
    assert "选择最接近的食物" in summary
    assert "估计份量" in summary


def test_valid_portion_still_produces_a_draft() -> None:
    """放宽组件校验之后，正常输入必须照常生成草稿。"""

    summary, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "豆腐",
        "FOOD_010",
        120,
    )

    assert preview_state is not None
    assert "千卡" in summary
    assert preview_state["event"]["payload"]["portion"]["grams"] == 120.0


def test_out_of_range_portion_is_still_rejected_by_the_backend() -> None:
    """去掉组件的 maximum 之后，上限仍然要由后端把住。"""

    summary, preview_state, _, _ = calculate_meal_preview(
        str(FIXTURE_IMAGE),
        "豆腐",
        "FOOD_010",
        99999,
    )

    assert preview_state is None
    assert "10000" in summary


def test_workflow_heading_does_not_carry_the_privacy_paragraph() -> None:
    """图片外发的说明放在隐私页和 README，不占对话区版面。"""

    from src.ui.app import MEAL_WORKFLOW_STEPS

    assert "MEAL_DETECTION_MODE" not in MEAL_WORKFLOW_STEPS
    assert "离开了这台机器" not in MEAL_WORKFLOW_STEPS
    assert len(MEAL_WORKFLOW_STEPS) < 120
