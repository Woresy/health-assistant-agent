"""通过 OpenAI-compatible 多模态 Provider 识别餐食。

和 YOLO 适配器实现同一个 `FoodDetector` 协议：只产出"建议检索词"，
交给现有 Hybrid RAG 检索、由用户确认，营养值仍然按选中的食物数据行计算。

与 YOLO 的关键差别，必须让用户知道：**图片会离开本机，发送给第三方 Provider。**
因此这条链路默认关闭，开启后界面必须显式告知。

刻意不把本地食物库的名称列表塞进 Prompt：那会退化成和 COCO 一样的闭集问题，
逼模型从已知名字里挑一个最像的。这里让它自由说出中文菜名，再由检索的
拒答门控决定认不认——认不出来就诚实回退手填。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.vision.detector import DetectionError
from src.vision.models import DetectionResult, FoodDetection


MAX_SUGGESTIONS = 5

# 整图描述，不是逐框检测，所以一次请求就够，不需要多轮。
SYSTEM_PROMPT = (
    "你是餐食图片识别助手。只做一件事：说出图里的食物叫什么。\n"
    "规则：\n"
    "1. 用中文说出最常见的菜名或食材名，例如「番茄炒蛋」「米饭」「苹果」。\n"
    "2. 名称要简短，不超过 12 个字，不要加份量、不要加形容词、不要加标点。\n"
    "3. 按你的把握从高到低排列，最多 5 个。\n"
    "4. confidence 用 0 到 1 的小数，表示你对这个名字的把握。\n"
    "5. 如果图里没有食物，或者你认不出来，返回空列表。"
    "不要猜一个最像的答案充数。\n"
    "6. 只输出 JSON，不要输出任何解释或代码块标记。\n"
    "输出格式：{\"foods\": [{\"name\": \"番茄炒蛋\", \"confidence\": 0.9}]}"
)

USER_PROMPT = "这张图里的食物是什么？按上面的格式只回 JSON。"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}

# 图片体积上限与 validate_image 一致；再大不该走到这里。
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class VlmFood(BaseModel):
    """模型返回的一个候选。"""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=0.0, ge=0, le=1)


class VlmResponse(BaseModel):
    """模型返回的完整结构。"""

    model_config = ConfigDict(extra="ignore")

    foods: list[VlmFood] = Field(default_factory=list)


def _strip_code_fence(text: str) -> str:
    """去掉模型偶尔加上的 ```json 包裹。"""

    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2:
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return stripped


def encode_image_data_url(image_path: str | Path) -> str:
    """把本地图片编码成 data URL。"""

    path = Path(image_path)

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DetectionError(
            "DETECTION_IMAGE_UNREADABLE",
            f"识别前无法读取图片：{exc}",
        ) from exc

    if not payload:
        raise DetectionError(
            "DETECTION_IMAGE_UNREADABLE",
            "图片内容为空",
        )

    if len(payload) > MAX_IMAGE_BYTES:
        raise DetectionError(
            "DETECTION_IMAGE_TOO_LARGE",
            "图片超过 5MB，不发送给 Provider",
        )

    media_type = (
        mimetypes.guess_type(path.name)[0] or "image/jpeg"
    )
    if media_type not in ALLOWED_IMAGE_TYPES:
        media_type = "image/jpeg"

    encoded = base64.b64encode(payload).decode("ascii")

    return f"data:{media_type};base64,{encoded}"


class VlmFoodDetector:
    """用多模态 Provider 给出菜名建议。"""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        min_confidence: float = 0.35,
        max_tokens: int = 256,
    ) -> None:
        self._client = client
        self._model = model
        self.min_confidence = min_confidence
        self.max_tokens = max_tokens

    @property
    def model_version(self) -> str:
        return f"vlm:{self._model}"

    def _request(self, data_url: str) -> str:
        """调用 Provider，返回原始文本。"""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self.max_tokens,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": USER_PROMPT,
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            # 只回错误类型，不把 Provider 的原始报文（可能含密钥片段）带出去。
            raise DetectionError(
                "DETECTION_PROVIDER_FAILED",
                (
                    "图片识别 Provider 调用失败"
                    f"（{type(exc).__name__}）；"
                    "请检查网络、密钥和该模型是否支持图片输入"
                ),
            ) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise DetectionError(
                "DETECTION_PROVIDER_PROTOCOL",
                "图片识别 Provider 返回了无法解析的结构",
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise DetectionError(
                "DETECTION_PROVIDER_PROTOCOL",
                "图片识别 Provider 返回了空内容",
            )

        return content

    def _parse(self, content: str) -> list[VlmFood]:
        """解析模型返回的 JSON；解析不了就当作识别失败，不猜。"""

        try:
            payload = json.loads(_strip_code_fence(content))
            parsed = VlmResponse.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise DetectionError(
                "DETECTION_PROVIDER_PROTOCOL",
                f"图片识别结果不是预期的 JSON：{exc}",
            ) from exc

        return parsed.foods

    def detect(
        self,
        image_path: str | Path,
    ) -> DetectionResult:
        """识别一张已经通过 `validate_image` 的图片。"""

        started_at = time.perf_counter()
        data_url = encode_image_data_url(image_path)
        foods = self._parse(self._request(data_url))

        detections: list[FoodDetection] = []
        seen: set[str] = set()

        for food in foods:
            name = food.name.strip()
            if not name or name in seen:
                continue
            if food.confidence < self.min_confidence:
                continue

            seen.add(name)
            detections.append(
                FoodDetection(
                    class_id=None,
                    label=name,
                    label_zh=name,
                    suggested_query=name,
                    confidence=food.confidence,
                    bbox=None,
                )
            )

            if len(detections) >= MAX_SUGGESTIONS:
                break

        detections.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        elapsed_ms = (time.perf_counter() - started_at) * 1000

        return DetectionResult(
            status="detected" if detections else "no_detection",
            mode="model",
            engine="vlm",
            model_version=self.model_version,
            min_confidence=self.min_confidence,
            detections=detections,
            elapsed_ms=elapsed_ms,
        )


def build_vlm_detector(
    *,
    min_confidence: float = 0.35,
) -> VlmFoodDetector:
    """按环境变量构建 VLM 检测器。

    视觉 Provider 独立配置：文本模型不支持图片时（例如纯文本的 Provider），
    可以只把视觉这一路指向别家，不动 `AGENT_*`。未单独设置时回退到 `AGENT_*`。
    """

    from openai import OpenAI

    from src.agent.openai_model import _validate_base_url

    model = (
        os.getenv("MEAL_DETECTION_VLM_MODEL", "").strip()
        or os.getenv("AGENT_MODEL", "").strip()
    )
    if not model:
        raise DetectionError(
            "DETECTION_VLM_NOT_CONFIGURED",
            "未设置 MEAL_DETECTION_VLM_MODEL，也没有可回退的 AGENT_MODEL",
        )

    api_key = (
        os.getenv("MEAL_DETECTION_VLM_API_KEY", "").strip()
        or os.getenv("AGENT_API_KEY", "").strip()
    )
    if not api_key:
        raise DetectionError(
            "DETECTION_VLM_NOT_CONFIGURED",
            "未设置 MEAL_DETECTION_VLM_API_KEY，也没有可回退的 AGENT_API_KEY",
        )

    raw_base_url = (
        os.getenv("MEAL_DETECTION_VLM_BASE_URL", "").strip()
        or os.getenv("AGENT_BASE_URL", "").strip()
    )
    try:
        base_url = _validate_base_url(raw_base_url)
    except Exception as exc:
        raise DetectionError(
            "DETECTION_VLM_NOT_CONFIGURED",
            f"图片识别 Provider 地址无效：{exc}",
        ) from exc

    try:
        timeout = float(
            os.getenv("MEAL_DETECTION_VLM_TIMEOUT", "20").strip() or 20
        )
    except ValueError:
        timeout = 20.0

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=min(max(timeout, 1.0), 60.0),
        max_retries=0,
    )

    return VlmFoodDetector(
        client,
        model,
        min_confidence=min_confidence,
    )
