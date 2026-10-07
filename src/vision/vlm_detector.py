"""通过 OpenAI-compatible 多模态 Provider 识别餐食。

和 YOLO 适配器实现同一个 `FoodDetector` 协议：产出"建议检索词"和估算份量，
优先用本地食物数据；缺失时可使用显式标注的模型营养估算，用户确认后保存。

与 YOLO 的关键差别，必须让用户知道：**图片会离开本机，发送给第三方 Provider。**
因此这条链路默认关闭，开启后界面必须显式告知。

刻意不把本地食物库的名称列表塞进 Prompt：那会退化成和 COCO 一样的闭集问题，
逼模型从已知名字里挑一个最像的。这里按餐盒或成品菜给出保守名称和显式粗估，
由用户核对后确认；不可辨认的图片不会伪造结果。
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

from src.nutrition.model_estimate import ModelNutritionEstimate
from src.vision.detector import DetectionError
from src.vision.models import DetectionResult, FoodDetection


MAX_SUGGESTIONS = 20

SYSTEM_PROMPT = (
    "你是日常餐食记录助手。用户只想确认整餐估算，不会填写克重。\n"
    "1. 以独立餐盒或成品菜为单位识别：一只餐盒里的主食、配菜、面包、酱料合为一项；"
    "一盘成品菜为一项，独立汤碗或饮料可单列。不要把同一餐盒拆成原料清单。\n"
    "2. name 用简洁、保守的中文名称，如「三明治餐盒」「蔬菜意面」「汤品」。"
    "看不清肉类、口味或馅料时不要硬猜，如无法辨明鱼种就说「煎鱼」。"
    "components 只列有把握的主要组成，不能在 foods 中再次计算这些配料。\n"
    "3. 每个独立餐盒/菜品用唯一 group_id，例如 box_left、box_middle、soup_right；"
    "不同餐盒即使同名也保持独立。portion_description 写位置与自然份量，例如「左侧约一盒」。\n"
    "4. estimated_grams 为这一整盒/整道菜可食部分的总克重，包含配菜和酱料，"
    "不含容器。根据可见大小与常见份量估算，不要全部填100克。范围大于0且不超过10000。\n"
    "5. 每项同时给出 estimated_nutrition：根据可见主要组成、常见烹调油和酱料，"
    "估算混合熟食每100克的热量、蛋白质、脂肪、碳水。数值只是模型粗估，"
    "不是营养数据库结果；不要编造来源网址或声称已称量。不要把整份热量填到每100克。\n"
    "6. confidence 表示对餐盒或成品菜类别的把握，0到1。认不出来或没有食物返回空列表，"
    "不要猜一个最像的答案充数。确实无法估量时相关字段返回null。\n"
    "7. 覆盖图中全部餐盒/成品菜，不重复、不遗漏；最多20项。只输出JSON。\n"
    '格式：{"foods":[{"group_id":"box_left","name":"三明治餐盒",'
    '"components":["三明治","蔬菜配菜"],"confidence":0.85,"estimated_grams":350,'
    '"portion_description":"左侧约一盒","estimated_nutrition":{'
    '"calories_per_100g":200,"protein_per_100g":12,"fat_per_100g":8,"carbs_per_100g":20}}]}'
)

USER_PROMPT = "按餐盒或成品菜汇总这张照片，估算整盒份量和混合熟食营养，不逐个拆出配料。只回JSON。"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}

# 图片体积上限与 validate_image 一致；再大不该走到这里。
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class VlmFood(BaseModel):
    """模型返回的一个候选。"""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=0.0, ge=0, le=1)
    estimated_grams: float | None = Field(default=None, gt=0, le=10000, allow_inf_nan=False)
    portion_description: str = Field(default="", max_length=120)
    group_id: str | None = Field(default=None, max_length=40)
    components: list[str] = Field(default_factory=list, max_length=12)
    estimated_nutrition: ModelNutritionEstimate | None = None


class VlmResponse(BaseModel):
    """模型返回的完整结构。"""

    model_config = ConfigDict(extra="ignore")

    foods: list[VlmFood] = Field(max_length=MAX_SUGGESTIONS)


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
        max_tokens: int = 4096,
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
            if not name or food.confidence < self.min_confidence:
                continue
            key = f"group:{food.group_id}" if food.group_id else f"name:{name}"
            if key in seen:
                # A repeated container is the same visual object, not another portion.
                if food.group_id:
                    continue
                previous = next(item for item in detections if item.label == name and not item.group_id)
                if previous.estimated_grams is not None and food.estimated_grams is not None:
                    old_grams = previous.estimated_grams
                    total = old_grams + food.estimated_grams
                    if total > 10000:
                        raise DetectionError("DETECTION_PROVIDER_PROTOCOL", "合并后的份量超出上限")
                    if previous.estimated_nutrition is not None and food.estimated_nutrition is not None:
                        previous.estimated_nutrition = ModelNutritionEstimate(**{
                            field: (getattr(previous.estimated_nutrition, field) * old_grams
                                    + getattr(food.estimated_nutrition, field) * food.estimated_grams) / total
                            for field in ModelNutritionEstimate.model_fields
                        })
                    else:
                        previous.estimated_nutrition = None
                    previous.estimated_grams = total
                    previous.portion_description = "同类食物合计"
                else:
                    previous.estimated_grams = None
                    previous.estimated_nutrition = None
                continue

            seen.add(key)
            detections.append(
                FoodDetection(
                    class_id=None,
                    label=name,
                    label_zh=name,
                    suggested_query=name,
                    confidence=food.confidence,
                    bbox=None,
                    estimated_grams=food.estimated_grams,
                    portion_description=food.portion_description,
                    group_id=food.group_id,
                    components=food.components,
                    estimated_nutrition=food.estimated_nutrition,
                )
            )

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
            incomplete=any(not food.name.strip() or food.confidence < self.min_confidence for food in foods),
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
