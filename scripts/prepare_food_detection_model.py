"""下载并导出餐食检测所需的 YOLOv8n ONNX 权重。

权重不随仓库提交：本项目是 MIT，而 Ultralytics YOLOv8 的代码和预训练权重是
AGPL-3.0。本脚本按固定版本从官方 Release 下载 `yolov8n.pt`，用 ultralytics
导出成 ONNX，写到 `runtime/models/`（已在 .gitignore 里）。

ultralytics 和 torch 只在导出时需要，不进 requirements.txt；运行时只用
onnxruntime 加载导出结果。

用法：

    pip install ultralytics                                  # 仅导出时需要
    python scripts/prepare_food_detection_model.py           # 下载并导出
    python scripts/prepare_food_detection_model.py --emit-pin  # 只打印固定值

`--emit-pin` 只依赖标准库，可以在安装依赖之前运行，用于 CI 缓存键。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 固定到具体 Release，避免"最新"在不同机器上导出不同权重。
WEIGHTS_VERSION = "v8.3.0"
WEIGHTS_NAME = "yolov8n.pt"
WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    f"{WEIGHTS_VERSION}/{WEIGHTS_NAME}"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runtime" / "models"
DEFAULT_ONNX_NAME = "yolov8n.onnx"
INPUT_SIZE = 640
COCO_CLASS_COUNT = 80

DOWNLOAD_TIMEOUT_SECONDS = 120


def cache_key() -> str:
    """构造只含缓存键安全字符的固定值标识。"""

    identity = f"{WEIGHTS_NAME}-{WEIGHTS_VERSION}-onnx{INPUT_SIZE}"
    return re.sub(r"[^A-Za-z0-9._-]", "-", identity)


def emit_pin() -> None:
    """以 GitHub Actions 的 key=value 形式打印固定值。"""

    sys.stdout.write(
        f"weights_name={WEIGHTS_NAME}\n"
        f"weights_version={WEIGHTS_VERSION}\n"
        f"weights_url={WEIGHTS_URL}\n"
        f"input_size={INPUT_SIZE}\n"
        f"cache_key={cache_key()}\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download_weights(target: Path) -> Path:
    """下载 .pt 权重；已存在就直接复用。"""

    if target.exists() and target.stat().st_size > 0:
        print(f"reusing cached weights: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")

    print(f"downloading {WEIGHTS_URL}")
    try:
        with urllib.request.urlopen(
            WEIGHTS_URL,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response, temporary.open("wb") as sink:
            shutil.copyfileobj(response, sink)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"下载 YOLOv8n 权重失败：{exc}\n"
            f"可以手动下载 {WEIGHTS_URL} 后放到 {target}，再重新运行本脚本。"
        ) from exc

    os.replace(temporary, target)

    return target


def export_onnx(
    weights_path: Path,
    output_path: Path,
) -> Path:
    """用 ultralytics 把 .pt 导出成固定输入尺寸的 ONNX。"""

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "导出 ONNX 需要 ultralytics（会一并安装 torch）：\n"
            "    pip install ultralytics\n"
            "导出完成后运行时只需要 onnxruntime，可以再把它卸载。\n"
            f"原始错误：{exc}"
        ) from exc

    print(f"exporting {weights_path.name} -> ONNX (imgsz={INPUT_SIZE})")
    model = YOLO(str(weights_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=INPUT_SIZE,
            dynamic=False,
            simplify=False,
            opset=12,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), str(output_path))

    return output_path


def write_manifest(
    output_path: Path,
    weights_sha256: str,
) -> Path:
    """写出运行时读取的权重清单。"""

    manifest_path = output_path.parent / "detection_manifest.json"
    digest = _sha256_file(output_path)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": WEIGHTS_NAME,
        "model_version": f"yolov8n@{WEIGHTS_VERSION}+sha256:{digest[:12]}",
        "source_url": WEIGHTS_URL,
        "weights_sha256": weights_sha256,
        "input_size": INPUT_SIZE,
        "class_count": COCO_CLASS_COUNT,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--emit-pin", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.emit_pin:
        emit_pin()
        return 0

    output_dir = Path(args.output_dir)
    onnx_path = output_dir / DEFAULT_ONNX_NAME

    weights_path = download_weights(output_dir / WEIGHTS_NAME)
    weights_sha256 = _sha256_file(weights_path)

    export_onnx(weights_path, onnx_path)
    manifest_path = write_manifest(onnx_path, weights_sha256)

    print(
        f"food detection model ready: {onnx_path}\n"
        f"manifest: {manifest_path}\n"
        "在 .env 设置 MEAL_DETECTION_MODE=onnx 后重启应用即可启用预填。\n"
        "注意：YOLOv8 权重为 Ultralytics AGPL-3.0 授权，不随本仓库分发。"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
