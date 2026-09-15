"""按本机配置构建检测器。

配置不可用时返回 None，而不是抛错——识别只是增强项，
缺它不该让上传图片这件事失败。
"""

from __future__ import annotations

from src.vision.config import FoodDetectionConfig
from src.vision.detector import FoodDetector


def build_food_detector(
    config: FoodDetectionConfig | None = None,
) -> FoodDetector | None:
    """检测可用就返回检测器，否则返回 None。"""

    active_config = config or FoodDetectionConfig.from_environment()

    if not active_config.available:
        return None

    if active_config.mode == "vlm":
        from src.vision.detector import DetectionError
        from src.vision.vlm_detector import build_vlm_detector

        try:
            return build_vlm_detector(
                min_confidence=active_config.min_confidence,
            )
        except DetectionError:
            # 配置不全时退回手填，不让应用起不来。
            return None

    from src.vision.onnx_detector import OnnxFoodDetector

    return OnnxFoodDetector(
        active_config.model_path,
        min_confidence=active_config.min_confidence,
    )
