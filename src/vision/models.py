"""餐食检测的结果模型。

检测按餐盒或成品菜给出名称、份量及明确标注的模型营养估算。
"""

from __future__ import annotations

from typing import Literal

from src.nutrition.model_estimate import ModelNutritionEstimate

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


DetectionStatus = Literal[
    "detected",
    "no_detection",
]

DetectionMode = Literal[
    "model",
    "manual_fallback",
]

# 产生候选的引擎；写进 Trace，便于事后区分是哪条链路给的建议。
DetectionEngine = Literal[
    "yolov8n_onnx",
    "vlm",
]


class BoundingBox(BaseModel):
    """还原到原图像素坐标的检测框。"""

    model_config = ConfigDict(extra="forbid")

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class FoodDetection(BaseModel):
    """单个检测结果。

    `suggested_query` 是喂给现有 Hybrid RAG 的检索词，不是选中的食物。

    `class_id` 和 `bbox` 只有目标检测器（YOLO）能给。VLM 整图描述菜名，
    给不出框和类别号，这两个字段为 None——不编造坐标。
    """

    model_config = ConfigDict(extra="forbid")

    class_id: int | None = Field(default=None, ge=0)
    label: str = Field(min_length=1)
    label_zh: str = Field(min_length=1)
    suggested_query: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox | None = None
    estimated_grams: float | None = Field(default=None, gt=0, le=10000, allow_inf_nan=False)
    portion_description: str = Field(default="", max_length=120)
    group_id: str | None = Field(default=None, max_length=40)
    components: list[str] = Field(default_factory=list, max_length=12)
    estimated_nutrition: ModelNutritionEstimate | None = None


class DetectionResult(BaseModel):
    """一次检测的完整结果。

    `mode="manual_fallback"` 表示这次没有真实推理，界面必须回到手填模式，
    不得显示任何识别结论。
    """

    model_config = ConfigDict(extra="forbid")

    status: DetectionStatus
    mode: DetectionMode
    engine: DetectionEngine = "yolov8n_onnx"
    model_version: str
    min_confidence: float = Field(ge=0, le=1)
    detections: list[FoodDetection] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    incomplete: bool = False

    @property
    def top_detection(self) -> FoodDetection | None:
        """置信度最高的一个；界面只把它预填进食物名称输入框。"""

        if not self.detections:
            return None
        return max(
            self.detections,
            key=lambda detection: detection.confidence,
        )
