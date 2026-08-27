"""将上游手动矫正版 JSON 清洗为仓库 FoodRecord 格式。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nutrition.repository import FoodRecord  # noqa: E402
from src.nutrition.text_normalize import normalize_text, split_food_name  # noqa: E402


PLACEHOLDERS = {"", "—", "-", "/", "\\", "无", "未测", "n/a", "na"}
SOURCE_NAME = "《中国食物成分表》第6版整理仓库（仅供个人学习研究）"
SOURCE_VERSION = "json_data_v3_20260825_qwen38max_kimi_k3_fixed"
UPDATED_AT = "2026-08-25"


def _parse_number(value: Any, field: str) -> tuple[float | None, str | None]:
    """解析字符串数值；微量值归零并返回质量标记。"""

    if not isinstance(value, str):
        return None, None
    cleaned = value.strip()
    if cleaned.casefold() == "tr":
        return 0.0, f"trace_value:{field}"
    if cleaned.casefold() in PLACEHOLDERS:
        return None, None
    try:
        number = float(cleaned.replace(",", ""))
    except ValueError:
        return None, None
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return None, None
    return number, None


def _load_aliases(path: Path) -> dict[str, dict[str, list[str]]]:
    """加载并校验两层人工别名表。"""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"别名文件无法读取：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("别名文件顶层必须是对象")

    result: dict[str, dict[str, list[str]]] = {}
    for section in ("by_food_code", "by_name_keyword"):
        mapping = raw.get(section, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"别名文件 {section} 必须是对象")
        checked: dict[str, list[str]] = {}
        for key, values in mapping.items():
            if not isinstance(key, str) or not isinstance(values, list):
                raise ValueError(f"别名文件 {section} 的键和值类型无效")
            if not all(isinstance(value, str) for value in values):
                raise ValueError(f"别名文件 {section} 只能包含字符串别名")
            checked[key] = values
        result[section] = checked
    return result


def _merge_aliases(
    food_code: str,
    name: str,
    extracted: list[str],
    aliases_config: dict[str, dict[str, list[str]]],
) -> list[str]:
    """按抽取、foodCode、关键词顺序合并别名并归一化去重。"""

    values = [
        *extracted,
        *aliases_config["by_food_code"].get(food_code, []),
    ]
    normalized_name = normalize_text(name)
    for keyword, aliases in aliases_config["by_name_keyword"].items():
        if normalize_text(keyword) in normalized_name:
            values.extend(aliases)

    seen = {normalized_name}
    merged: list[str] = []
    for value in values:
        cleaned = value.strip()
        normalized = normalize_text(cleaned)
        if cleaned and normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(cleaned)
    return merged


def _quality_flags(raw: dict[str, Any]) -> list[str]:
    """计算只标记、不修正的能量与总量异常。"""

    flags: list[str] = []
    parsed: dict[str, float | None] = {}
    for field in ("energyKCal", "energyKJ", "water", "protein", "fat", "CHO", "ash"):
        number, trace_flag = _parse_number(raw.get(field), field)
        parsed[field] = number
        if trace_flag is not None:
            flags.append(trace_flag)

    kcal = parsed["energyKCal"]
    kilojoules = parsed["energyKJ"]
    if kcal is not None and kilojoules is not None and kilojoules > 0:
        ratio_error = abs(kcal * 4.184 - kilojoules) / kilojoules
        if ratio_error > 0.20:
            flags.append("energy_unit_ratio_suspect")

    total_fields = ["water", "protein", "fat", "CHO", "ash"]
    if all(parsed[field] is not None for field in total_fields):
        if sum(parsed[field] or 0.0 for field in total_fields) > 100:
            flags.append("macro_sum_over_100")
    return sorted(set(flags))


def _exclusion_sample(
    source_file: Path,
    record_index: int,
    food_code: str,
    reason: str,
) -> dict[str, Any]:
    """构造不含整条原始数据的可审计排除样例。"""

    return {
        "source_file": source_file.name,
        "record_index": record_index,
        "food_code": food_code,
        "reason": reason,
    }


def prepare_food_data(
    src: Path,
    out: Path,
    aliases_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """清洗目录中全部 merged_*.json，并写入数据与报告。"""

    source_files = sorted(src.glob("merged_*.json"))
    aliases_config = _load_aliases(aliases_path)
    foods: list[FoodRecord] = []
    seen_codes: set[str] = set()
    exclusion_reasons: Counter[str] = Counter()
    exclusion_samples: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()

    for source_file in source_files:
        try:
            records = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"源文件无法读取：{source_file}：{exc}") from exc
        if not isinstance(records, list):
            raise ValueError(f"源文件顶层不是数组：{source_file}")

        category = source_file.stem.removeprefix("merged_")
        for record_index, raw in enumerate(records, start=1):
            if not isinstance(raw, dict):
                food_code = ""
                reason = "record_not_object"
            else:
                food_code = str(raw.get("foodCode", "")).strip()
                if not food_code:
                    reason = "food_code_missing"
                elif food_code in seen_codes:
                    reason = "duplicate_food_code"
                else:
                    reason = ""
                    seen_codes.add(food_code)

            if reason:
                exclusion_reasons[reason] += 1
                if len(exclusion_samples) < 50:
                    exclusion_samples.append(
                        _exclusion_sample(
                            source_file,
                            record_index,
                            food_code,
                            reason,
                        )
                    )
                continue

            raw_name = raw.get("foodName")
            if not isinstance(raw_name, str) or not normalize_text(raw_name):
                reason = "food_name_missing"
                exclusion_reasons[reason] += 1
                if len(exclusion_samples) < 50:
                    exclusion_samples.append(
                        _exclusion_sample(
                            source_file, record_index, food_code, reason
                        )
                    )
                continue

            core_fields = {
                "energyKCal": "calories_per_100g",
                "protein": "protein_per_100g",
                "fat": "fat_per_100g",
                "CHO": "carbs_per_100g",
            }
            nutrition: dict[str, float] = {}
            invalid_field: str | None = None
            for upstream_field, output_field in core_fields.items():
                number, _ = _parse_number(raw.get(upstream_field), upstream_field)
                if number is None:
                    invalid_field = upstream_field
                    break
                nutrition[output_field] = number

            if invalid_field is not None:
                reason = f"core_field_invalid:{invalid_field}"
                exclusion_reasons[reason] += 1
                if len(exclusion_samples) < 50:
                    exclusion_samples.append(
                        _exclusion_sample(
                            source_file, record_index, food_code, reason
                        )
                    )
                continue

            name, extracted_aliases = split_food_name(raw_name)
            aliases = _merge_aliases(
                food_code,
                name,
                extracted_aliases,
                aliases_config,
            )
            flags = _quality_flags(raw)
            food = FoodRecord.model_validate(
                {
                    "food_id": food_code,
                    "name": name,
                    "aliases": aliases,
                    "category": category,
                    **nutrition,
                    "source": SOURCE_NAME,
                    "source_version": SOURCE_VERSION,
                    "updated_at": UPDATED_AT,
                    "quality_flags": flags,
                }
            )
            foods.append(food)
            quality_counts.update(flags)

    foods.sort(key=lambda item: item.food_id)
    serialized_foods = [food.model_dump(mode="json") for food in foods]
    alias_total = sum(len(food.aliases) for food in foods)
    excluded_count = sum(exclusion_reasons.values())
    report: dict[str, Any] = {
        "source_file_count": len(source_files),
        "written_count": len(foods),
        "excluded_count": excluded_count,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "exclusion_samples": exclusion_samples,
        "alias_total": alias_total,
        "alias_mean": round(alias_total / len(foods), 4) if foods else 0.0,
        "quality_flag_counts": dict(sorted(quality_counts.items())),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(serialized_foods, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    """解析命令行并执行可复现清洗。"""

    parser = argparse.ArgumentParser(description="清洗中国食物成分 JSON 数据")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = prepare_food_data(
        src=args.src,
        out=args.out,
        aliases_path=args.aliases,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
