"""Gradio 单页应用。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import gradio as gr

from src.health.models import HealthEvent
from src.nutrition.calculator import (
    NutritionCalculationError,
    calculate_nutrition,
    parse_grams,
)
from src.nutrition.repository import (
    FoodRepository,
    NutritionDataError,
)
from src.storage.jsonl_store import HealthEventStore, JsonlReadError
from src.tools.confirmation import issue_confirmation_token
from src.tools.save_health_event import save_health_event
from src.ui.image_input import validate_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOODS_PATH = PROJECT_ROOT / "data" / "samples" / "foods_sample.json"
EVENTS_PATH = PROJECT_ROOT / "data" / "health_events.jsonl"

repository = FoodRepository(FOODS_PATH)
event_store = HealthEventStore(EVENTS_PATH)


def _error_text(error_code: str, message: str) -> str:
    return f"错误 [{error_code}]：{message}"


def search_candidates(
    image_path: str | None,
    food_query: str,
) -> tuple[gr.Dropdown, list[list[Any]], str]:
    """校验图片并根据用户手填名称展示候选。"""

    image_result = validate_image(image_path)
    if not image_result.ok:
        return (
            gr.Dropdown(choices=[], value=None),
            [],
            _error_text(
                image_result.error_code or "IMAGE_INVALID",
                image_result.message,
            ),
        )

    try:
        result = repository.search(food_query, top_k=5)
    except NutritionDataError as exc:
        return (
            gr.Dropdown(choices=[], value=None),
            [],
            _error_text(exc.error_code, exc.message),
        )

    if result.status == "not_found":
        return (
            gr.Dropdown(choices=[], value=None),
            [],
            _error_text(
                "NOT_FOUND",
                f"未找到“{result.query}”对应的固定食物数据；不会猜测营养值",
            ),
        )

    choices: list[tuple[str, str]] = []
    rows: list[list[Any]] = []

    for candidate in result.candidates:
        choices.append(
            (
                f"{candidate.name}｜{candidate.category}｜"
                f"{candidate.match_type}",
                candidate.food_id,
            )
        )
        rows.append(
            [
                candidate.food_id,
                candidate.name,
                candidate.category,
                candidate.score,
                candidate.match_type,
                candidate.source,
                candidate.source_version,
                "manual",
            ]
        )

    return (
        gr.Dropdown(
            choices=choices,
            value=result.candidates[0].food_id,
        ),
        rows,
        f"已加载 {len(rows)} 个手动检索候选，请选择后计算。",
    )


def calculate_preview(
    image_path: str | None,
    food_query: str,
    selected_food_id: str | None,
    raw_grams: Any,
) -> tuple[str, dict[str, Any] | None, str]:
    """计算预览，并签发与该预览绑定的确认令牌。"""

    image_result = validate_image(image_path)
    if not image_result.ok:
        return (
            _error_text(
                image_result.error_code or "IMAGE_INVALID",
                image_result.message,
            ),
            None,
            "",
        )

    try:
        search_result = repository.search(food_query, top_k=5)
        if search_result.status == "not_found":
            return (
                _error_text(
                    "NOT_FOUND",
                    "没有固定食物数据，不能计算或保存",
                ),
                None,
                "",
            )

        candidate_ids = {
            candidate.food_id
            for candidate in search_result.candidates
        }
        if not selected_food_id or selected_food_id not in candidate_ids:
            return (
                _error_text(
                    "CANDIDATE_REQUIRED",
                    "请选择当前检索结果中的一个候选",
                ),
                None,
                "",
            )

        food = repository.get_by_food_id(selected_food_id)
        grams = parse_grams(raw_grams)
        estimate = calculate_nutrition(
            food=food,
            raw_grams=grams,
            retrieval_query=food_query,
        )
    except NutritionDataError as exc:
        return _error_text(exc.error_code, exc.message), None, ""
    except NutritionCalculationError as exc:
        return _error_text(exc.error_code, exc.message), None, ""

    now = datetime.now(timezone.utc)
    event = HealthEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_id": str(uuid4()),
            "user_id": "local-demo-user",
            "event_type": "meal",
            "occurred_at": now,
            "payload": {
                "food": {
                    "food_id": food.food_id,
                    "name": food.name,
                    "category": food.category,
                },
                "portion": {
                    "grams": float(grams),
                    "unit": "g",
                },
                "nutrition": estimate.model_dump(),
                "retrieval_query": food_query.strip(),
                "candidate_source": "manual",
                "estimated": True,
            },
            "source_refs": [estimate.source_ref],
            "input_source": "image_manual",
            "created_at": now,
            "updated_at": now,
        }
    )

    confirmation_token = issue_confirmation_token(event)
    preview_state = {
        "event": event.model_dump(mode="json"),
        "confirmation_token": confirmation_token,
        "idempotency_key": str(uuid4()),
    }

    summary = (
        "### 待确认饮食记录\n\n"
        f"- 食物：{food.name}（{food.food_id}）\n"
        f"- 份量：{float(grams):g} g\n"
        f"- 热量估算：{estimate.calories_kcal:.2f} kcal\n"
        f"- 蛋白质估算：{estimate.protein_g:.2f} g\n"
        f"- 脂肪估算：{estimate.fat_g:.2f} g\n"
        f"- 碳水估算：{estimate.carbs_g:.2f} g\n"
        f"- 数据来源：{food.source}\n"
        f"- 来源版本：{food.source_version}\n"
        f"- 检索词：{estimate.retrieval_query}\n"
        f"- 份量假设：{estimate.portion_assumption}\n\n"
        "**以上均为估算值，仅供学习，不构成医疗建议。**"
    )

    return summary, preview_state, "计算完成，请核对后确认保存。"


def _today_records() -> tuple[list[list[Any]], str]:
    """每次都从 JSONL 重新读取今日 meal 事件。"""

    try:
        events = event_store.read_all()
    except JsonlReadError as exc:
        return [], _error_text("STORAGE_READ_FAILED", str(exc))

    timezone_name = os.getenv("APP_TIMEZONE", "Asia/Shanghai")
    try:
        local_timezone = ZoneInfo(timezone_name)
    except Exception:
        return [], _error_text(
            "TIMEZONE_INVALID",
            f"无法加载时区：{timezone_name}",
        )

    today = datetime.now(local_timezone).date()
    today_events = [
        event
        for event in events
        if event.occurred_at.astimezone(local_timezone).date() == today
    ]

    rows: list[list[Any]] = []
    total_calories = 0.0
    total_protein = 0.0
    total_fat = 0.0
    total_carbs = 0.0

    for event in today_events:
        nutrition = event.payload.nutrition
        total_calories += nutrition.calories_kcal
        total_protein += nutrition.protein_g
        total_fat += nutrition.fat_g
        total_carbs += nutrition.carbs_g

        rows.append(
            [
                event.event_id,
                event.occurred_at.astimezone(local_timezone).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                event.payload.food.name,
                event.payload.portion.grams,
                nutrition.calories_kcal,
                nutrition.protein_g,
                nutrition.fat_g,
                nutrition.carbs_g,
            ]
        )

    summary = (
        "### 今日汇总\n\n"
        f"- 记录数：{len(today_events)}\n"
        f"- 热量估算合计：{total_calories:.2f} kcal\n"
        f"- 蛋白质估算合计：{total_protein:.2f} g\n"
        f"- 脂肪估算合计：{total_fat:.2f} g\n"
        f"- 碳水估算合计：{total_carbs:.2f} g\n\n"
        "**估算值，仅供学习，不构成医疗建议。**"
    )
    return rows, summary


def confirm_save(
    preview_state: dict[str, Any] | None,
) -> tuple[str, list[list[Any]], str]:
    """用户明确点击后才调用保存工具。"""

    if not preview_state:
        rows, today_summary = _today_records()
        return (
            _error_text(
                "PREVIEW_REQUIRED",
                "请先完成检索和营养计算",
            ),
            rows,
            today_summary,
        )

    result = save_health_event(
        event_input=preview_state["event"],
        confirmation_token=preview_state["confirmation_token"],
        idempotency_key=preview_state["idempotency_key"],
        store=event_store,
    )

    rows, today_summary = _today_records()

    if not result["ok"]:
        error = result["error"]
        return (
            _error_text(
                error["error_code"],
                error["message"],
            ),
            rows,
            today_summary,
        )

    if result["data"]["idempotent"]:
        status = "该请求已保存过，未新增重复记录。"
    else:
        status = (
            "记录成功："
            f"{result['data']['event']['event_id']}"
        )

    return status, rows, today_summary


def cancel_preview() -> tuple[None, str, str]:
    """取消仅清除内存中的待保存状态，不写文件。"""

    return (
        None,
        "已取消，未写入任何半成品。",
        "尚未计算待确认记录。",
    )


def build_demo() -> gr.Blocks:
    """构建单页三 Tab 应用。"""

    with gr.Blocks(title="食物健康助手") as demo:
        gr.Markdown(
            "# 食物健康助手\n"
            "上传图片后手动填写食物；图片仅作为输入入口，不做图片识别。"
        )

        preview_state = gr.State(value=None)

        with gr.Tab("记录饮食"):
            image_input = gr.File(
                label="上传一张食物图片",
                file_count="single",
                file_types=[".jpg", ".jpeg", ".png"],
                type="filepath",
            )
            food_query = gr.Textbox(
                label="手动填写食物名称",
                placeholder="例如：西红柿",
            )
            grams_input = gr.Number(
                label="食物克重（g）",
                minimum=0.01,
                maximum=10000,
            )

            search_button = gr.Button("查找候选", variant="secondary")
            candidate_status = gr.Markdown()

            candidate_table = gr.Dataframe(
                headers=[
                    "food_id",
                    "name",
                    "category",
                    "匹配分",
                    "match_type",
                    "source",
                    "source_version",
                    "candidate_source",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                value=[],
                interactive=False,
                label="手动检索候选",
            )

            selected_food = gr.Dropdown(
                label="选择候选食物",
                choices=[],
                value=None,
                interactive=True,
            )

            calculate_button = gr.Button("计算营养估算", variant="primary")
            calculation_status = gr.Markdown()
            preview = gr.Markdown("尚未计算待确认记录。")

            with gr.Row():
                save_button = gr.Button("确认保存", variant="primary")
                cancel_button = gr.Button("取消", variant="stop")

            save_status = gr.Markdown()

        with gr.Tab("今日记录"):
            refresh_button = gr.Button("重新读取今日记录")
            today_table = gr.Dataframe(
                headers=[
                    "event_id",
                    "occurred_at",
                    "food",
                    "grams",
                    "calories_kcal",
                    "protein_g",
                    "fat_g",
                    "carbs_g",
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                ],
                value=[],
                interactive=False,
                label="今日饮食记录",
            )
            today_summary = gr.Markdown()

        with gr.Tab("隐私与数据"):
            gr.Markdown(
                "## 隐私与数据\n\n"
                "- 上传图片只在当前交互中读取，不复制到项目数据目录。\n"
                "- 本项目不进行图片识别。\n"
                "- 饮食记录保存在本机 `data/health_events.jsonl`。\n"
                "- 运行时健康数据已被 Git 忽略。\n"
                "- 示例食物数据仅供学习，不是完整数据集。\n"
                "- 页面结果均为估算值，不构成医疗建议。\n"
                "- 可以停止应用后手动删除 JSONL，以清除本地记录。"
            )

        search_button.click(
            fn=search_candidates,
            inputs=[image_input, food_query],
            outputs=[
                selected_food,
                candidate_table,
                candidate_status,
            ],
        )

        calculate_button.click(
            fn=calculate_preview,
            inputs=[
                image_input,
                food_query,
                selected_food,
                grams_input,
            ],
            outputs=[
                preview,
                preview_state,
                calculation_status,
            ],
        )

        save_button.click(
            fn=confirm_save,
            inputs=[preview_state],
            outputs=[
                save_status,
                today_table,
                today_summary,
            ],
        )

        cancel_button.click(
            fn=cancel_preview,
            outputs=[
                preview_state,
                save_status,
                preview,
            ],
        )

        refresh_button.click(
            fn=_today_records,
            outputs=[today_table, today_summary],
        )

        demo.load(
            fn=_today_records,
            outputs=[today_table, today_summary],
        )

    return demo


demo = build_demo()