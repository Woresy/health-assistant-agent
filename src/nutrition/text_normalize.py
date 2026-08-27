"""食物名称归一化、拆分与字符串相似度纯函数。"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


_SQUARE_BRACKET_PATTERN = re.compile(r"[\[［【](.*?)[\]］】]")
_ROUND_BRACKET_PATTERN = re.compile(r"[（(](.*?)[）)]")
_DISCARD_ANNOTATIONS = {
    "代表值",
    "均值",
    "平均值",
    "代表",
    "平均",
}


def normalize_text(value: str) -> str:
    """执行 NFKC、casefold，并抹平空白、标点和符号噪声。"""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("Z", "P", "S", "C"))
    )


def _unique_nonempty(values: list[str], standard_name: str) -> list[str]:
    """按首次出现顺序去重，并排除空值及标准名本身。"""

    seen = {normalize_text(standard_name)}
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        normalized = normalize_text(cleaned)
        if cleaned and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)
    return result


def split_food_name(food_name: str) -> tuple[str, list[str]]:
    """从上游名称提取标准名与书中别名，不改写名称语义。"""

    value = unicodedata.normalize("NFKC", food_name).strip()
    square_aliases = [
        alias.strip()
        for block in _SQUARE_BRACKET_PATTERN.findall(value)
        for alias in re.split(r"[,，、;/／]", block)
        if alias.strip()
    ]
    without_square = _SQUARE_BRACKET_PATTERN.sub("", value).strip()

    round_annotations = _ROUND_BRACKET_PATTERN.findall(without_square)
    standard_name = without_square
    aliases = list(square_aliases)

    for annotation in round_annotations:
        annotation = annotation.strip()
        if normalize_text(annotation) in {
            normalize_text(item) for item in _DISCARD_ANNOTATIONS
        }:
            standard_name = standard_name.replace(f"({annotation})", "")
            standard_name = standard_name.replace(f"（{annotation}）", "")
        elif annotation:
            aliases.append(annotation)

    standard_name = standard_name.strip()
    without_round = _ROUND_BRACKET_PATTERN.sub("", standard_name).strip()
    if without_round and without_round != standard_name:
        aliases.append(without_round)

    return standard_name, _unique_nonempty(aliases, standard_name)


def ngram_dice(left: str, right: str, n: int) -> float:
    """计算字符 n-gram 多重集 Dice 系数。"""

    if n <= 0:
        raise ValueError("n 必须大于 0")
    if not left and not right:
        return 1.0
    if len(left) < n or len(right) < n:
        return 0.0

    left_grams = Counter(left[index : index + n] for index in range(len(left) - n + 1))
    right_grams = Counter(
        right[index : index + n] for index in range(len(right) - n + 1)
    )
    overlap = sum((left_grams & right_grams).values())
    total = sum(left_grams.values()) + sum(right_grams.values())
    return (2.0 * overlap / total) if total else 1.0


def levenshtein_distance(left: str, right: str) -> int:
    """以 O(min(m, n)) 空间计算 Levenshtein 编辑距离。"""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def levenshtein_ratio(left: str, right: str) -> float:
    """把编辑距离转换为 0—1 相似度。"""

    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(left, right) / longest


def fuzzy_score(left: str, right: str) -> float:
    """严格按 PRD 组合字符 Dice 与 Levenshtein 相似度。"""

    unigram = ngram_dice(left, right, 1)
    bigram = ngram_dice(left, right, 2)
    dice_score = 0.6 * unigram + 0.4 * bigram
    return max(dice_score, levenshtein_ratio(left, right))
