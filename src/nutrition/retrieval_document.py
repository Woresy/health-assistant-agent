"""把结构化食物记录转换为向量检索文档。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from src.nutrition.repository import FoodRecord
from src.nutrition.text_normalize import (
    normalize_text,
)


class RetrievalDocument(BaseModel):
    """一条可被 embedding 模型编码的食物检索文档。"""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    food_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    canonical_name: str = Field(min_length=1)
    aliases: list[str]
    category: str = Field(min_length=1)
    semantic_hints: list[str]

    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)


class RetrievalHintFile(BaseModel):
    """人工审核的检索语义提示文件。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    hints: dict[str, list[str]]


class RetrievalHintError(Exception):
    """检索提示文件错误。"""


def load_retrieval_hints(
    path: str | Path,
) -> dict[str, list[str]]:
    """加载、去重并清洗语义提示。"""

    source_path = Path(path)

    try:
        raw_text = source_path.read_text(
            encoding="utf-8"
        )
        hint_file = (
            RetrievalHintFile
            .model_validate_json(raw_text)
        )
    except FileNotFoundError as exc:
        raise RetrievalHintError(
            f"找不到检索提示文件：{source_path}"
        ) from exc
    except (
        OSError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        raise RetrievalHintError(
            f"检索提示文件无效：{exc}"
        ) from exc

    if hint_file.schema_version != "1.0":
        raise RetrievalHintError(
            "检索提示文件 schema_version 必须是 1.0"
        )

    cleaned_mapping: dict[
        str,
        list[str],
    ] = {}

    for food_id, hints in hint_file.hints.items():
        cleaned_food_id = food_id.strip()

        if not cleaned_food_id:
            raise RetrievalHintError(
                "检索提示中的 food_id 不能为空"
            )

        seen: set[str] = set()
        cleaned_hints: list[str] = []

        for hint in hints:
            cleaned_hint = hint.strip()
            normalized_hint = normalize_text(
                cleaned_hint
            )

            if (
                cleaned_hint
                and normalized_hint
                and normalized_hint not in seen
            ):
                seen.add(normalized_hint)
                cleaned_hints.append(
                    cleaned_hint
                )

        cleaned_mapping[
            cleaned_food_id
        ] = cleaned_hints

    return cleaned_mapping


def build_retrieval_document(
    food: FoodRecord,
    *,
    semantic_hints: list[str] | None = None,
) -> RetrievalDocument:
    """
    将 FoodRecord 转换成用于食物识别的文档。

    文档包含人工审核的语义描述，但不包含营养数值。
    """

    aliases_text = (
        "、".join(food.aliases)
        if food.aliases
        else "无"
    )

    cleaned_hints = [
        hint.strip()
        for hint in (
            semantic_hints or []
        )
        if hint.strip()
    ]

    hints_text = (
        "；".join(cleaned_hints)
        if cleaned_hints
        else "无额外语义提示"
    )

    text = (
        f"标准名称：{food.name}。"
        f"别名：{aliases_text}。"
        f"食物类别：{food.category}。"
        f"语义描述：{hints_text}。"
        "该记录用于识别食物名称、类别、外观、"
        "食用方式和日常自然语言描述。"
    )

    return RetrievalDocument(
        document_id=f"food:{food.food_id}",
        food_id=food.food_id,
        text=text,
        canonical_name=food.name,
        aliases=list(food.aliases),
        category=food.category,
        semantic_hints=cleaned_hints,
        source=food.source,
        source_version=food.source_version,
    )


def build_retrieval_documents(
    foods: tuple[FoodRecord, ...],
    *,
    hints_by_food_id: (
        dict[str, list[str]] | None
    ) = None,
) -> list[RetrievalDocument]:
    """按 food_id 稳定排序生成全部检索文档。"""

    active_hints = hints_by_food_id or {}

    return [
        build_retrieval_document(
            food,
            semantic_hints=(
                active_hints.get(
                    food.food_id,
                    [],
                )
            ),
        )
        for food in sorted(
            foods,
            key=lambda item: item.food_id,
        )
    ]