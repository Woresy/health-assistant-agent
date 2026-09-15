"""VLM 识别适配器测试。

全部用假 client，不发网络请求、不消耗任何 Provider 额度。
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.vision.detector import DetectionError, FoodDetector
from src.vision.vlm_detector import (
    VlmFoodDetector,
    encode_image_data_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "meal.png"


class FakeClient:
    """按脚本返回内容的假 Provider。"""

    def __init__(self, content: str | None = None, error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content)
                )
            ]
        )


def detector(content: str | None = None, error: Exception | None = None):
    return VlmFoodDetector(
        FakeClient(content, error),
        "fake-vl",
        min_confidence=0.35,
    )


def test_satisfies_the_detector_protocol() -> None:
    assert isinstance(detector('{"foods": []}'), FoodDetector)


def test_recognises_a_chinese_dish() -> None:
    """COCO 做不到的事：说出中餐菜名。"""

    d = detector(
        '{"foods": [{"name": "番茄炒蛋", "confidence": 0.92},'
        ' {"name": "米饭", "confidence": 0.71}]}'
    )
    result = d.detect(FIXTURE_IMAGE)

    assert result.status == "detected"
    assert result.engine == "vlm"
    assert result.model_version == "vlm:fake-vl"
    assert [x.label_zh for x in result.detections] == ["番茄炒蛋", "米饭"]
    assert result.detections[0].suggested_query == "番茄炒蛋"
    # VLM 给不出框和类别号，不得编造。
    assert result.detections[0].bbox is None
    assert result.detections[0].class_id is None


def test_sends_the_image_as_a_data_url() -> None:
    d = detector('{"foods": []}')
    d.detect(FIXTURE_IMAGE)

    request = d._client.calls[0]
    image_part = request["messages"][1]["content"][0]

    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # 确实是这张图的字节。
    assert base64.b64decode(url.split(",", 1)[1]) == FIXTURE_IMAGE.read_bytes()
    assert request["temperature"] == 0


def test_low_confidence_answers_are_dropped() -> None:
    d = detector(
        '{"foods": [{"name": "米饭", "confidence": 0.9},'
        ' {"name": "面条", "confidence": 0.1}]}'
    )
    result = d.detect(FIXTURE_IMAGE)

    assert [x.label_zh for x in result.detections] == ["米饭"]


def test_duplicate_names_are_collapsed() -> None:
    d = detector(
        '{"foods": [{"name": "苹果", "confidence": 0.9},'
        ' {"name": "苹果", "confidence": 0.8}]}'
    )
    result = d.detect(FIXTURE_IMAGE)

    assert len(result.detections) == 1


def test_empty_answer_is_no_detection_not_a_guess() -> None:
    """模型说认不出来时，必须如实返回，不能退而求其次猜一个。"""

    d = detector('{"foods": []}')
    result = d.detect(FIXTURE_IMAGE)

    assert result.status == "no_detection"
    assert result.detections == []


def test_code_fenced_json_is_accepted() -> None:
    d = detector('```json\n{"foods": [{"name": "宫保鸡丁", "confidence": 0.8}]}\n```')
    result = d.detect(FIXTURE_IMAGE)

    assert result.detections[0].label_zh == "宫保鸡丁"


def test_unparseable_answer_raises_instead_of_guessing() -> None:
    d = detector("我觉得这可能是米饭吧")

    with pytest.raises(DetectionError) as exc:
        d.detect(FIXTURE_IMAGE)

    assert exc.value.error_code == "DETECTION_PROVIDER_PROTOCOL"


def test_empty_content_is_a_protocol_error() -> None:
    d = detector("   ")

    with pytest.raises(DetectionError) as exc:
        d.detect(FIXTURE_IMAGE)

    assert exc.value.error_code == "DETECTION_PROVIDER_PROTOCOL"


def test_provider_failure_does_not_leak_raw_error_text() -> None:
    """Provider 报文可能带密钥片段，只回错误类型。"""

    secret = "sk-live-abcdef123456"
    d = detector(error=RuntimeError(f"401 unauthorized key={secret}"))

    with pytest.raises(DetectionError) as exc:
        d.detect(FIXTURE_IMAGE)

    assert exc.value.error_code == "DETECTION_PROVIDER_FAILED"
    assert secret not in exc.value.message
    assert "RuntimeError" in exc.value.message


def test_oversized_image_is_not_sent(tmp_path: Path) -> None:
    big = tmp_path / "big.jpg"
    big.write_bytes(b"\xff" * (5 * 1024 * 1024 + 1))

    with pytest.raises(DetectionError) as exc:
        encode_image_data_url(big)

    assert exc.value.error_code == "DETECTION_IMAGE_TOO_LARGE"


def test_unreadable_image_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DetectionError) as exc:
        encode_image_data_url(tmp_path / "absent.png")

    assert exc.value.error_code == "DETECTION_IMAGE_UNREADABLE"


def test_prompt_does_not_pin_the_model_to_the_local_food_list() -> None:
    """不能把 33 条食物名塞进 Prompt——那会重演 COCO 的闭集错误。"""

    from src.vision.vlm_detector import SYSTEM_PROMPT

    assert "FOOD_0" not in SYSTEM_PROMPT
    # 允许"认不出来"是关键：闭集模型没有这个选项，才会把番茄说成苹果。
    assert "认不出来" in SYSTEM_PROMPT
    assert "不要猜一个最像的答案充数" in SYSTEM_PROMPT
