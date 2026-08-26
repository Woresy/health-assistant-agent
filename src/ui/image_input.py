"""上传图片的格式、大小和完整性校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict


MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}


class ImageValidationResult(BaseModel):
    """图片校验统一结果。"""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error_code: str | None
    message: str
    image_format: Literal["JPEG", "PNG"] | None = None
    size_bytes: int | None = None


def validate_image(
    image_path: str | Path | None,
) -> ImageValidationResult:
    """验证单张 JPG、JPEG 或 PNG，不保存、不复制原图。"""

    if image_path is None or not str(image_path).strip():
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_REQUIRED",
            message="请上传一张 JPG、JPEG 或 PNG 图片",
        )

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_FILE_MISSING",
            message="上传的临时图片文件不存在",
        )

    suffix = path.suffix.casefold()
    if suffix not in ALLOWED_SUFFIXES:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_UNSUPPORTED_FORMAT",
            message="仅支持 JPG、JPEG、PNG 格式",
        )

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_READ_FAILED",
            message=f"无法读取图片大小：{exc}",
        )

    if size_bytes == 0:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_EMPTY",
            message="图片文件为空",
            size_bytes=0,
        )

    if size_bytes > MAX_IMAGE_BYTES:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_TOO_LARGE",
            message="图片不得超过 5MB",
            size_bytes=size_bytes,
        )

    try:
        with Image.open(path) as image:
            detected_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_CORRUPTED",
            message=f"图片已损坏或无法解析：{exc}",
            size_bytes=size_bytes,
        )

    if detected_format not in ALLOWED_PIL_FORMATS:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_UNSUPPORTED_FORMAT",
            message="图片内容不是 JPG、JPEG 或 PNG",
            size_bytes=size_bytes,
        )

    expected_format = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    if detected_format != expected_format:
        return ImageValidationResult(
            ok=False,
            error_code="IMAGE_FORMAT_MISMATCH",
            message="图片扩展名与实际内容格式不一致",
            size_bytes=size_bytes,
        )

    return ImageValidationResult(
        ok=True,
        error_code=None,
        message="图片校验通过，可继续手动填写食物",
        image_format=detected_format,
        size_bytes=size_bytes,
    )