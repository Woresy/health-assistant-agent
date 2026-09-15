"""基于 ONNX Runtime 的 YOLOv8 餐食检测适配器。

onnxruntime 在方法内导入，缺失时抛类型化错误让上层回退手填，
与 `src/nutrition/dense_retriever.py` 处理 sentence-transformers 的做法一致。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.vision.detector import DetectionError
from src.vision.label_map import DetectionLabelMap
from src.vision.models import (
    BoundingBox,
    DetectionResult,
    FoodDetection,
)
from src.vision.postprocess import (
    DEFAULT_IOU_THRESHOLD,
    decode_predictions,
    to_model_input,
)


DEFAULT_INPUT_SIZE = 640


class DetectionManifest(BaseModel):
    """权重导出清单，由 prepare_food_detection_model.py 写出。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    created_at: str
    model_name: str
    model_version: str
    source_url: str
    weights_sha256: str = Field(min_length=64, max_length=64)
    input_size: int = Field(ge=32)
    class_count: int = Field(ge=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class OnnxFoodDetector:
    """加载本地 ONNX 权重，只回报映射表里的食物类别。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_confidence: float = 0.35,
        label_map: DetectionLabelMap | None = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold

        self._label_map = label_map
        self._session: Any | None = None
        self._manifest: DetectionManifest | None = None
        self._input_name: str | None = None
        self._input_size: int = DEFAULT_INPUT_SIZE
        self._model_version: str | None = None

    @property
    def label_map(self) -> DetectionLabelMap:
        if self._label_map is None:
            self._label_map = DetectionLabelMap.load()
        return self._label_map

    @property
    def model_version(self) -> str:
        """优先用清单里的版本号，没有清单就退回权重摘要。"""

        if self._model_version is not None:
            return self._model_version

        manifest = self._load_manifest()
        if manifest is not None:
            self._model_version = manifest.model_version
            self._input_size = manifest.input_size
            return self._model_version

        if not self.model_path.is_file():
            raise DetectionError(
                "DETECTION_WEIGHTS_MISSING",
                f"找不到检测权重：{self.model_path}",
            )

        digest = _sha256_file(self.model_path)
        self._model_version = (
            f"{self.model_path.stem}@sha256:{digest[:12]}"
        )

        return self._model_version

    def _load_manifest(self) -> DetectionManifest | None:
        if self._manifest is not None:
            return self._manifest

        manifest_path = (
            self.model_path.parent / "detection_manifest.json"
        )
        if not manifest_path.exists():
            return None

        try:
            self._manifest = DetectionManifest.model_validate(
                json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            )
        except (
            OSError,
            ValueError,
            ValidationError,
        ) as exc:
            raise DetectionError(
                "DETECTION_MANIFEST_INVALID",
                f"检测权重清单无效：{exc}",
            ) from exc

        return self._manifest

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session

        if not self.model_path.is_file():
            raise DetectionError(
                "DETECTION_WEIGHTS_MISSING",
                (
                    f"找不到检测权重：{self.model_path}；"
                    "请先运行 scripts/prepare_food_detection_model.py"
                ),
            )

        try:
            import onnxruntime
        except Exception as exc:
            raise DetectionError(
                "DETECTION_RUNTIME_MISSING",
                (
                    "本机没有可用的 onnxruntime；"
                    f"请运行 pip install -r requirements.txt：{exc}"
                ),
            ) from exc

        try:
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise DetectionError(
                "DETECTION_WEIGHTS_INVALID",
                f"无法加载检测权重：{exc}",
            ) from exc

        model_input = session.get_inputs()[0]
        self._input_name = model_input.name

        # 导出时固定了输入尺寸就用它，动态维度回退到默认 640。
        shape = list(model_input.shape or [])
        if len(shape) == 4 and isinstance(shape[2], int):
            self._input_size = int(shape[2])

        self._session = session

        return session

    def detect(
        self,
        image_path: str | Path,
    ) -> DetectionResult:
        """检测一张已经通过 `validate_image` 的图片。"""

        session = self._get_session()
        label_map = self.label_map
        started_at = time.perf_counter()

        try:
            with Image.open(Path(image_path)) as opened:
                image = np.asarray(opened.convert("RGB"))
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise DetectionError(
                "DETECTION_IMAGE_UNREADABLE",
                f"检测前无法读取图片：{exc}",
            ) from exc

        image_height, image_width = image.shape[:2]

        try:
            tensor, scale, pad_x, pad_y = to_model_input(
                image,
                self._input_size,
            )
            outputs = session.run(
                None,
                {self._input_name: tensor},
            )
            decoded = decode_predictions(
                outputs[0],
                keep_class_ids=label_map.class_ids,
                min_confidence=self.min_confidence,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                image_width=image_width,
                image_height=image_height,
                iou_threshold=self.iou_threshold,
            )
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(
                "DETECTION_INFERENCE_FAILED",
                f"检测推理失败：{type(exc).__name__}",
            ) from exc

        detections: list[FoodDetection] = []

        for class_id, confidence, box in decoded:
            label = label_map.get(class_id)
            if label is None:
                continue

            detections.append(
                FoodDetection(
                    class_id=class_id,
                    label=label.coco_label,
                    label_zh=label.query,
                    suggested_query=label.query,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                    ),
                )
            )

        elapsed_ms = (
            time.perf_counter() - started_at
        ) * 1000

        return DetectionResult(
            status=(
                "detected" if detections else "no_detection"
            ),
            mode="model",
            model_version=self.model_version,
            min_confidence=self.min_confidence,
            detections=detections,
            elapsed_ms=elapsed_ms,
        )
