"""食物清洗、四阶段检索、Trace 与工具协议单元测试。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.prepare_food_data import prepare_food_data
from src.nutrition.repository import FoodRepository
from src.nutrition.text_normalize import (
    fuzzy_score,
    levenshtein_distance,
    levenshtein_ratio,
    ngram_dice,
    normalize_text,
    split_food_name,
)
from src.storage.trace_store import TraceWriteError
from src.tools.retrieve_nutrition_candidates import (
    retrieve_nutrition_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"


class FailingTraceStore:
    """模拟 Trace 存储故障，验证主检索不被降级为失败。"""

    def append(self, trace: object) -> None:
        raise TraceWriteError("模拟写入失败")


def test_text_normalization_and_name_split() -> None:
    """归一化抹平全半角与噪声，名称拆分保留有效别名。"""

    assert normalize_text(" Ｔｏｍａｔｏ-饭！ ") == "tomato饭"
    assert split_food_name("鸡（代表值）") == ("鸡", [])
    assert split_food_name("牛肉(胸部肉)［牛胸］") == (
        "牛肉(胸部肉)",
        ["牛胸", "胸部肉", "牛肉"],
    )


def test_similarity_functions_are_deterministic() -> None:
    """Dice 和编辑距离遵守边界并对错字给出稳定相似度。"""

    assert ngram_dice("番茄", "番茄", 1) == 1.0
    assert ngram_dice("番茄", "蕃茄", 2) == 0.0
    assert levenshtein_distance("鸡胸肉", "鸡匈肉") == 1
    assert levenshtein_ratio("鸡胸肉", "鸡匈肉") == pytest.approx(2 / 3)
    assert fuzzy_score("鸡胸肉", "鸡匈肉") == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="n 必须大于 0"):
        ngram_dice("a", "a", 0)


def test_four_stage_search_and_explainability() -> None:
    """四阶段命中、分数、去重和 not_found 返回均符合固定协议。"""

    repository = FoodRepository(SAMPLE_PATH)

    exact = repository.search("番茄", top_k=3)
    assert exact.candidates[0].stage == 0
    assert exact.candidates[0].score == 1.0
    assert exact.candidates[0].matched_term == "番茄"

    alias = repository.search("西红柿", top_k=3)
    assert alias.candidates[0].food_id == "FOOD_001"
    assert alias.candidates[0].match_type == "alias"
    assert alias.candidates[0].stage == 1
    assert alias.candidates[0].score == 0.95
    assert len({item.food_id for item in alias.candidates}) == len(alias.candidates)

    contains = repository.search("番茄果切", top_k=3)
    assert contains.candidates[0].food_id == "FOOD_001"
    assert contains.candidates[0].stage == 2
    assert contains.candidates[0].matched_term == "番茄果"
    assert contains.candidates[0].score == 0.825

    fuzzy = repository.search("鸡匈肉", top_k=3)
    assert fuzzy.candidates[0].food_id == "FOOD_004"
    assert fuzzy.candidates[0].stage == 3
    assert fuzzy.candidates[0].score == 0.4933
    assert fuzzy.selection_mode == "user_required"

    rejected = repository.search("火星不存在食物", top_k=3)
    assert rejected.status == "not_found"
    assert rejected.candidates == []
    assert rejected.auto_select_allowed is False
    assert rejected.dataset_record_count == 28


@pytest.mark.parametrize("top_k", [True, 0, 11, 1.5, "3"])
def test_tool_rejects_invalid_top_k(top_k: object) -> None:
    """工具层对 int 1—10 之外的 top_k 使用稳定错误码。"""

    result = retrieve_nutrition_candidates("番茄", top_k=top_k)
    assert result["ok"] is False
    assert result["error"]["error_code"] == "TOP_K_INVALID"


def test_tool_not_found_is_success_and_trace_failure_is_non_fatal() -> None:
    """拒答是业务成功，Trace 失败仅作为 warning 返回。"""

    repository = FoodRepository(SAMPLE_PATH)
    result = retrieve_nutrition_candidates(
        "火星不存在食物",
        top_k=3,
        repository=repository,
        trace_store=FailingTraceStore(),  # type: ignore[arg-type]
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert result["data"]["status"] == "not_found"
    assert result["data"]["trace_warning"]["error_code"] == (
        "TRACE_WRITE_FAILED"
    )


def test_prepare_food_data_with_synthetic_fixture(tmp_path: Path) -> None:
    """本机无上游数据时，用合成文件覆盖 Tr、缺失、别名、异常与重复。"""

    source_dir = tmp_path / "upstream"
    source_dir.mkdir()
    records = [
        {
            "foodCode": "TEST_001",
            "foodName": "牛肉(胸部肉)［牛胸］",
            "energyKCal": "100",
            "energyKJ": "100",
            "protein": "Tr",
            "fat": "2",
            "CHO": "3",
            "water": "90",
            "ash": "2",
        },
        {
            "foodCode": "TEST_002",
            "foodName": "缺失核心值",
            "energyKCal": "",
            "energyKJ": "",
            "protein": "1",
            "fat": "1",
            "CHO": "1",
            "water": "90",
            "ash": "1",
        },
        {
            "foodCode": "TEST_001",
            "foodName": "重复编码",
            "energyKCal": "100",
            "energyKJ": "418.4",
            "protein": "1",
            "fat": "1",
            "CHO": "1",
            "water": "90",
            "ash": "1",
        },
        {
            "foodCode": "TEST_003",
            "foodName": "总量异常",
            "energyKCal": "100",
            "energyKJ": "418.4",
            "protein": "10",
            "fat": "10",
            "CHO": "20",
            "water": "70",
            "ash": "1",
        },
    ]
    source_path = source_dir / "merged_测试类.json"
    source_path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                "by_food_code": {"TEST_003": ["测试别名"]},
                "by_name_keyword": {"牛肉": ["黄牛肉"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "foods.json"
    report_path = tmp_path / "report.json"

    report = prepare_food_data(
        source_dir,
        output_path,
        aliases_path,
        report_path,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["source_file_count"] == 1
    assert report["written_count"] == 2
    assert report["excluded_count"] == 2
    assert report["exclusion_reasons"] == {
        "core_field_invalid:energyKCal": 1,
        "duplicate_food_code": 1,
    }
    assert [item["food_id"] for item in output] == ["TEST_001", "TEST_003"]
    assert output[0]["protein_per_100g"] == 0.0
    assert output[0]["aliases"] == ["牛胸", "胸部肉", "牛肉", "黄牛肉"]
    assert "trace_value:protein" in output[0]["quality_flags"]
    assert "energy_unit_ratio_suspect" in output[0]["quality_flags"]
    assert "macro_sum_over_100" in output[1]["quality_flags"]
    assert Counter(report["quality_flag_counts"])["trace_value:protein"] == 1
