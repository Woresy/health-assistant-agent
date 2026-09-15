"""餐食检测的本机配置。

按 `docs/DECISIONS.md` §9.14 的准备条件分级，这是第三档：默认关闭，
缺失时不挡任何核心任务，只是少了预填。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = "runtime/models/yolov8n.onnx"
DEFAULT_MIN_CONFIDENCE = 0.35

SUPPORTED_MODES = frozenset({"disabled", "onnx", "vlm"})


@dataclass(frozen=True)
class FoodDetectionConfig:
    """检测开关、权重位置和置信度下限。"""

    mode: str = "disabled"
    model_path: Path = PROJECT_ROOT / DEFAULT_MODEL_PATH
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    @classmethod
    def from_environment(cls) -> "FoodDetectionConfig":
        mode = (
            os.getenv("MEAL_DETECTION_MODE", "disabled")
            .strip()
            .casefold()
        )
        if mode not in SUPPORTED_MODES:
            mode = "disabled"

        raw_path = os.getenv(
            "MEAL_DETECTION_MODEL_PATH", ""
        ).strip()
        model_path = (
            Path(raw_path)
            if raw_path
            else PROJECT_ROOT / DEFAULT_MODEL_PATH
        )
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        confidence_text = os.getenv(
            "MEAL_DETECTION_MIN_CONFIDENCE", ""
        ).strip()
        try:
            min_confidence = float(
                confidence_text or DEFAULT_MIN_CONFIDENCE
            )
        except ValueError:
            min_confidence = DEFAULT_MIN_CONFIDENCE

        return cls(
            mode=mode,
            model_path=model_path,
            min_confidence=min(max(min_confidence, 0.05), 0.95),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @property
    def uses_remote_provider(self) -> bool:
        """图片是否会离开本机。界面必须据此显式告知用户。"""

        return self.mode == "vlm"

    @property
    def weights_present(self) -> bool:
        # VLM 不用本地权重，这一项不适用。
        if self.mode == "vlm":
            return True
        return self.model_path.is_file()

    @property
    def vlm_configured(self) -> bool:
        """VLM 模式需要模型名和密钥，可回退到 AGENT_* 配置。"""

        model = (
            os.getenv("MEAL_DETECTION_VLM_MODEL", "").strip()
            or os.getenv("AGENT_MODEL", "").strip()
        )
        api_key = (
            os.getenv("MEAL_DETECTION_VLM_API_KEY", "").strip()
            or os.getenv("AGENT_API_KEY", "").strip()
        )
        base_url = (
            os.getenv("MEAL_DETECTION_VLM_BASE_URL", "").strip()
            or os.getenv("AGENT_BASE_URL", "").strip()
        )
        return bool(model and api_key and base_url)

    @property
    def runtime_importable(self) -> bool:
        """只查能否导入，不真的导入，避免为了探测而加载运行时。"""

        package = "openai" if self.mode == "vlm" else "onnxruntime"

        return importlib_util.find_spec(package) is not None

    @property
    def available(self) -> bool:
        """开关、权重和推理运行时必须同时具备。

        只查权重文件会把"装了权重但没装 onnxruntime"报告成可用，
        参考 `src/knowledge/repository.py` 的 `retrieval_receipt`：
        少查一个条件就会把降级状态说成正常状态。
        """

        if not (
            self.enabled
            and self.weights_present
            and self.runtime_importable
        ):
            return False

        if self.mode == "vlm":
            return self.vlm_configured

        return True

    def unavailable_reason(self) -> str | None:
        """给界面用的四段式缺项说明：缺什么、为什么、怎么补、现在能做什么。"""

        if not self.enabled:
            return (
                "图片识别未开启，下面手动填写即可完整记一餐。"
                "想自动预填名称：在 .env 设置 MEAL_DETECTION_MODE=vlm "
                "后重启（说明见 README）。"
            )

        if self.mode == "vlm" and not self.vlm_configured:
            return (
                "图片识别已开启，但没有可用的多模态模型配置，"
                "这次上传只能手动填写。"
                "在 .env 设置 MEAL_DETECTION_VLM_MODEL、"
                "MEAL_DETECTION_VLM_API_KEY 和 MEAL_DETECTION_VLM_BASE_URL"
                "（不设置则分别回退用 AGENT_MODEL、AGENT_API_KEY 和 "
                "AGENT_BASE_URL），确认该模型支持图片输入后重启应用。"
                "在那之前，下面手动填写食物名称和份量仍然可以完整记一餐。"
            )

        if not self.weights_present:
            return (
                "图片识别已开启但找不到权重文件"
                f"（{self.model_path}），这次上传只能手动填写。"
                "运行 python scripts/prepare_food_detection_model.py "
                "下载并导出权重后重启应用即可。"
                "在那之前，下面手动填写食物名称和份量仍然可以完整记一餐。"
            )

        if not self.runtime_importable:
            package = "openai" if self.mode == "vlm" else "onnxruntime"
            return (
                "图片识别已开启但缺少运行依赖，这次上传只能手动填写。"
                f"运行 pip install -r requirements.txt 安装 {package} "
                "后重启应用即可。"
                "在那之前，下面手动填写食物名称和份量仍然可以完整记一餐。"
            )

        return None
