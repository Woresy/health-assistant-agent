"""检测器接口。

PRD 要求视觉推理通过单一可替换适配器接入，本期不设计多模型路由。
测试用 Fake 实现这个 Protocol，不 patch 模块全局变量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.vision.models import DetectionResult


class DetectionError(Exception):
    """检测失败的稳定错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


@runtime_checkable
class FoodDetector(Protocol):
    """餐食检测适配器。"""

    @property
    def model_version(self) -> str:
        """稳定的权重标识，会写进 Trace 和事件来源。"""

    def detect(
        self,
        image_path: str | Path,
    ) -> DetectionResult:
        """检测一张已经通过格式校验的图片。

        实现必须在无法推理时抛 `DetectionError`，不得返回空结果冒充
        "没检测到"——两者对界面的含义完全不同。
        """
