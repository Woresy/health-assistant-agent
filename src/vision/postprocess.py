"""YOLOv8 的纯 NumPy 前后处理。

单独成文件，好处是不装 onnxruntime、不下载权重也能完整单测这段逻辑。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


# 与 letterbox 填充色一致的常量，来自 YOLOv8 默认导出设置。
LETTERBOX_FILL = 114

DEFAULT_IOU_THRESHOLD = 0.45

# YOLOv8 导出的单图输出是 (1, 4 + 类别数, anchors)。
BOX_CHANNELS = 4


def letterbox(
    image: np.ndarray[Any, Any],
    target_size: int,
) -> tuple[np.ndarray[Any, Any], float, float, float]:
    """等比缩放并居中填充到正方形，返回还原所需的比例和偏移。"""

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("图片尺寸无效")

    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # 必须用双线性缩放：YOLOv8 的训练和官方推理都走 cv2.INTER_LINEAR，
    # 换成最近邻会让分数偏移到足以让阈值附近的检测忽有忽无。
    # Pillow 已经是本项目依赖，不必为此引入 OpenCV。
    resized = np.asarray(
        Image.fromarray(image).resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )
    )

    canvas = np.full(
        (target_size, target_size, image.shape[2]),
        LETTERBOX_FILL,
        dtype=image.dtype,
    )
    pad_x = (target_size - new_width) / 2
    pad_y = (target_size - new_height) / 2
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[
        top : top + new_height,
        left : left + new_width,
    ] = resized

    return canvas, scale, float(left), float(top)


def to_model_input(
    image: np.ndarray[Any, Any],
    target_size: int,
) -> tuple[np.ndarray[Any, Any], float, float, float]:
    """把 HWC uint8 RGB 图片变成 (1, 3, S, S) float32 输入。"""

    canvas, scale, pad_x, pad_y = letterbox(image, target_size)
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def xywh_to_xyxy(
    boxes: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """中心点宽高转左上右下。"""

    result = np.empty_like(boxes)
    half_width = boxes[:, 2] / 2
    half_height = boxes[:, 3] / 2
    result[:, 0] = boxes[:, 0] - half_width
    result[:, 1] = boxes[:, 1] - half_height
    result[:, 2] = boxes[:, 0] + half_width
    result[:, 3] = boxes[:, 1] + half_height

    return result


def non_max_suppression(
    boxes: np.ndarray[Any, Any],
    scores: np.ndarray[Any, Any],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[int]:
    """标准贪心 NMS，返回保留下来的下标。"""

    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]

    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        rest = order[1:]
        intersect_x1 = np.maximum(x1[current], x1[rest])
        intersect_y1 = np.maximum(y1[current], y1[rest])
        intersect_x2 = np.minimum(x2[current], x2[rest])
        intersect_y2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(
            intersect_x2 - intersect_x1, 0
        ) * np.maximum(intersect_y2 - intersect_y1, 0)
        union = areas[current] + areas[rest] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)
        order = rest[iou <= iou_threshold]

    return keep


def decode_predictions(
    raw_output: np.ndarray[Any, Any],
    *,
    keep_class_ids: frozenset[int] | set[int],
    min_confidence: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    image_width: int,
    image_height: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """把 YOLOv8 原始输出解码成原图坐标下的 (class_id, 置信度, 框)。

    只保留 `keep_class_ids` 里的类别——本项目只认 COCO 的 10 个食物类，
    别的类别（餐具、桌子、人）一律丢弃，不参与 NMS 也不上报。
    """

    predictions = np.asarray(raw_output, dtype=np.float32)

    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError("检测输出维度不是 (通道, anchors)")
    if predictions.shape[0] <= BOX_CHANNELS:
        raise ValueError("检测输出通道数不足以包含类别分数")

    # (通道, anchors) -> (anchors, 通道)
    predictions = predictions.transpose()

    boxes = predictions[:, :BOX_CHANNELS]
    class_scores = predictions[:, BOX_CHANNELS:]
    class_count = class_scores.shape[1]

    wanted = {
        class_id
        for class_id in keep_class_ids
        if 0 <= class_id < class_count
    }
    if not wanted:
        return []

    # 先在全部 80 类里取 argmax，再筛食物类——顺序不能反。
    # 只在食物列里取 argmax，会把一个高分的 person 强行改判成它
    # 最像的食物类，凭空造出"蛋糕 0.4"这种候选。
    best_class = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)

    selected = (best_score >= min_confidence) & np.isin(
        best_class,
        np.asarray(sorted(wanted), dtype=np.int64),
    )
    if not bool(selected.any()):
        return []

    boxes = xywh_to_xyxy(boxes[selected])
    best_score = best_score[selected]
    class_ids = best_class[selected]

    # 还原 letterbox：先去掉填充偏移，再除以缩放比例。
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_height)

    detections: list[
        tuple[int, float, tuple[float, float, float, float]]
    ] = []

    for class_id in np.unique(class_ids):
        mask = class_ids == class_id
        class_boxes = boxes[mask]
        class_confidences = best_score[mask]

        for index in non_max_suppression(
            class_boxes,
            class_confidences,
            iou_threshold=iou_threshold,
        ):
            box = class_boxes[index]
            detections.append(
                (
                    int(class_id),
                    float(
                        min(
                            max(
                                float(class_confidences[index]),
                                0.0,
                            ),
                            1.0,
                        )
                    ),
                    (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                )
            )

    detections.sort(key=lambda item: item[1], reverse=True)

    return detections
