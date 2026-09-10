"""生成可公开、确定性的人工餐食闭环验收图片。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "meals"

CASES = (
    ("01-tomato", (218, 86, 72), "TOMATO"),
    ("02-rice", (238, 226, 190), "RICE"),
    ("03-egg", (244, 190, 66), "EGG"),
    ("04-chicken", (205, 158, 112), "CHICKEN"),
    ("05-apple", (185, 54, 60), "APPLE"),
    ("06-banana", (239, 207, 72), "BANANA"),
    ("07-potato", (176, 133, 77), "POTATO"),
    ("08-cucumber", (71, 145, 91), "CUCUMBER"),
    ("09-milk", (224, 235, 236), "MILK"),
    ("10-tofu", (232, 218, 184), "TOFU"),
)


def generate() -> list[Path]:
    """创建十张内容不同但不宣称真实识别能力的 JPEG。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for index, (slug, color, label) in enumerate(CASES, start=1):
        image = Image.new("RGB", (360, 240), (246, 248, 243))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((18, 18, 342, 222), radius=24, fill=(255, 253, 247))
        draw.ellipse((95, 38, 265, 208), fill=(223, 232, 220), outline=(47, 118, 94), width=4)
        inset = 22 + index % 4 * 3
        draw.ellipse((95 + inset, 38 + inset, 265 - inset, 208 - inset), fill=color)
        draw.text((28, 28), f"MANUAL MEAL {index:02d}", fill=(23, 56, 47))
        draw.text((144 - len(label) * 3, 211), label, fill=(23, 56, 47))
        path = OUTPUT_DIR / f"{slug}.jpg"
        image.save(path, format="JPEG", quality=92, optimize=False)
        generated.append(path)
    return generated


if __name__ == "__main__":
    for generated_path in generate():
        print(generated_path.relative_to(PROJECT_ROOT))
