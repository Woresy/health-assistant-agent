"""Read-only health visualizations built from confirmed daily/period summaries."""

from html import escape
from math import ceil
from typing import Any


METRICS = (
    ("meal", "饮食记录", "calories_kcal", "kcal"),
    ("water", "饮水", "total_ml", "ml"),
    ("exercise", "运动", "total_duration_minutes", "分钟"),
    ("weight", "体重", "latest_weight_kg", "kg"),
)


def _calorie_visual(samples: list[tuple[str, dict[str, Any]]]) -> str:
    meals = [summary["meal"] for _, summary in samples if summary["meal"]["count"]]
    total = sum(float(meal["calories_kcal"]) for meal in meals)
    step = max(50, ceil(total / 40 / 50) * 50)
    ticks = ''.join(
        f'<span class="calorie-tick"><i style="height:{min(1, max(0, total / step - i)) * 100:.2f}%"></i></span>'
        for i in range(40)
    )
    macros = []
    for field, label in (("carbs_g", "碳水化合物"), ("protein_g", "蛋白质"), ("fat_g", "脂肪")):
        value = f'{sum(float(meal[field]) for meal in meals):.1f}' if meals and all(field in meal for meal in meals) else '—'
        macros.append(f'<div><strong>{value}<small> g</small></strong><span>{label}</span></div>')
    return (
        f'<div class="calorie-scale-label"><span>0</span><span>{40 * step:g} kcal</span></div>'
        f'<div class="calorie-ruler" role="img" aria-label="累计已记录 {total:g} kcal，每格 {step:g} kcal，不代表摄入目标">{ticks}</div>'
        f'<p class="visual-caption">每格 {step:g} kcal · 累计记录刻度，非摄入目标</p>'
        f'<div class="macro-totals">{"".join(macros)}</div><p class="macro-note">观察期营养合计 · 仅含已记录餐食</p>'
    )


def _weight_visual(samples: list[tuple[str, dict[str, Any]]]) -> str:
    points = [(i, date, float(summary["weight"]["latest_weight_kg"])) for i, (date, summary) in enumerate(samples)
              if summary["weight"]["count"] and summary["weight"]["latest_weight_kg"] is not None]
    if not points:
        return '<div class="weight-empty">记录一次体重，开始观察变化</div>'
    low = min(p[2] for p in points) - 1
    high = max(p[2] for p in points) + 1
    def x(index):
        if len(points) == 1:
            return 295
        return 60 + (index - points[0][0]) * 470 / (points[-1][0] - points[0][0])
    def y(value):
        return 125 - (value - low) / (high - low) * 80
    shapes = []
    for left, right in zip(points, points[1:]):
        gap = right[0] - left[0] > 1
        shapes.append(f'<path class="weight-trajectory {"weight-gap" if gap else ""}" d="M{x(left[0]):.1f} {y(left[2]):.1f} L{x(right[0]):.1f} {y(right[2]):.1f}"/>')
    for index, date, value in points:
        shapes.append(f'<circle class="weight-observation" cx="{x(index):.1f}" cy="{y(value):.1f}" r="6"><title>{escape(date)}：{value:g} kg</title></circle>')
    # Label the observed endpoints, keeping dense series legible.
    for index, date, value in ([points[0], points[-1]] if len(points) > 1 else points):
        shapes.append(f'<text x="{x(index):.1f}" y="{y(value)-18:.1f}" text-anchor="middle">{value:g} kg</text>')
        shapes.append(f'<text x="{x(index):.1f}" y="170" text-anchor="middle">{escape(date)}</text>')
    note = '实心点为实测体重；虚线跨过未记录日期，不代表每日变化。' if any(b[0]-a[0] > 1 for a, b in zip(points, points[1:])) else '实心点为实测体重，按记录日期连接。'
    if len(points) == 1:
        note = '目前只有一天的记录，再记录一次即可比较变化。'
    return f'<svg class="weight-overview" viewBox="0 0 590 188" role="img" aria-label="实测体重变化轨迹">{"".join(shapes)}</svg><p class="visual-caption">{note}</p>'


def trend_charts(samples: list[tuple[str, dict[str, Any]]]) -> str:
    """Keep missing dates visible and never interpolate across missing weights."""
    if not samples:
        return '<p class="chart-guide">暂无可展示的记录。</p>'
    panels = []
    for key, title, field, unit in METRICS:
        values = [float(s[key][field]) if s[key]["count"] and s[key][field] is not None else None for _, s in samples]
        known = [v for v in values if v is not None]
        low = max(0, min(known) - 1) if key == "weight" and known else 0
        high = max(known) if known else 1
        high = high + 1 if key == "weight" else max(high * 1.15, 1)
        x = lambda i: 44 + i * 512 / max(len(values) - 1, 1)
        y = lambda v: 142 - (v - low) / (high - low) * 112
        shapes = []
        for tick in ((0, 0.5, 1) if known else ()):
            value = low + (high - low) * tick
            yy = y(value)
            label = f"{value:.1f}" if key == "weight" else (f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}")
            shapes.append(f'<line x1="44" y1="{yy:.1f}" x2="556" y2="{yy:.1f}" class="chart-grid"/><text x="36" y="{yy + 4:.1f}" text-anchor="end">{label}</text>')

        previous = None
        for i, value in enumerate(values):
            date = samples[i][0]
            if value is None:
                shapes.append(f'<circle cx="{x(i):.1f}" cy="151" r="3" class="chart-missing"><title>{date}：未记录</title></circle>')
                previous = None
                continue
            tooltip = escape(f'{date}：{value:g} {unit}')
            if key == "weight":
                if previous is not None:
                    shapes.append(f'<line x1="{x(i-1):.1f}" y1="{y(previous):.1f}" x2="{x(i):.1f}" y2="{y(value):.1f}" class="chart-line"/>')
                shapes.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" class="chart-point"><title>{tooltip}</title></circle>')
            else:
                width = min(24, 330 / len(values))
                shapes.append(f'<rect x="{x(i)-width/2:.1f}" y="{y(value):.1f}" width="{width:.1f}" height="{max(2,142-y(value)):.1f}" rx="2" class="chart-bar"><title>{tooltip}</title></rect>')
            previous = value
        for i in sorted({0, len(samples)//2, len(samples)-1}):
            shapes.append(f'<text x="{x(i):.1f}" y="174" text-anchor="middle">{escape(samples[i][0])}</text>')
        if key == "weight":
            headline = f'最近一次 {known[-1]:g} {unit}' if known else '还没有体重记录'
            detail = f'首末记录变化 {known[-1]-known[0]:+g} kg' if len(known) > 1 else '至少记录两天后，才能比较变化'
        else:
            headline = f'已记录 {sum(known):g} {unit}' if known else '这段时间还没有记录'
            detail = f'{len(known)} 天有记录 · 有记录日均 {sum(known)/len(known):.0f} {unit}' if known else '记录后，这里会显示每天的变化'
        if key == "meal":
            detail += ' · 仅统计已记录餐食'
        entries = ''.join(f'<li><span>{escape(date)}</span><strong>{f"{value:g} {unit}" if value is not None else "未记录"}</strong></li>' for (date, _), value in zip(samples, values))
        chart = f'<svg viewBox="0 0 590 188" role="img" aria-label="{title}逐日趋势，详细数值见下方展开记录">{"".join(shapes)}</svg>'
        visual = chart
        context = f'最近 {len(samples)} 天'
        if key == "water":
            # Cup size adapts to the observation window; capacity is not a health goal.
            volume = sum(known)
            cup_ml = max(250, ceil(volume / 16 / 250) * 250)
            cups = []
            for i in range(16):
                filled = min(1, max(0, volume / cup_ml - i))
                cups.append(
                    f'<svg viewBox="0 0 32 40" aria-hidden="true"><defs><clipPath id="water-cup-{i}"><rect x="0" y="{35 - 30 * filled:.2f}" width="32" height="{30 * filled:.2f}"/></clipPath></defs>'
                    f'<path class="cup-outline" d="M4 5H28L24 35H8Z"/><path class="cup-fill" clip-path="url(#water-cup-{i})" d="M4 5H28L24 35H8Z"/></svg>'
                )
            visual = f'<div class="hydration-cups" role="img" aria-label="观察期已记录饮水 {volume:g} ml，每杯代表 {cup_ml:g} ml">{"".join(cups)}</div><p class="visual-caption">每杯 {cup_ml:g} ml · 表示累计记录量，不代表目标</p>'
            context += ' · 累计饮水'
        elif key == "exercise":
            tiles = ''.join(f'<div class="activity-day {"has-activity" if value is not None else "no-activity"}"><span>{escape(date[3:])}</span><strong>{f"{value:g}" if value is not None else "—"}</strong></div>' for (date, _), value in zip(samples, values))
            visual = f'<div class="activity-days" role="group" aria-label="逐日运动分钟数">{tiles}</div><p class="visual-caption">日期 / 运动分钟 · 横线表示未记录</p>'
            context += ' · 运动足迹'
        elif key == "meal":
            visual = _calorie_visual(samples)
        elif key == "weight":
            visual = _weight_visual(samples)
        # Keep exact values and the daily plot available behind a native disclosure.
        extra_chart = chart if key in {"water", "exercise", "meal"} else ''
        number = known[-1] if key == "weight" and known else sum(known)
        number_text = f'{number:g}' if known else '—'
        caption = '最近一次记录' if key == "weight" else '观察期已记录'
        if key == "weight" and known:
            last_date = next(date for date, summary in reversed(samples) if summary["weight"]["count"] and summary["weight"]["latest_weight_kg"] is not None)
            caption = f'{escape(last_date)} · 最近一次记录'
        measure = f'<p class="chart-headline"><span class="metric-number">{number_text}</span><span class="metric-unit">{unit}</span><span class="metric-caption">{caption}</span></p>'
        if key == "weight":
            comparison = f'<div class="weight-comparison"><strong>{known[-1]-known[0]:+g} kg</strong><span>较观察期首次记录</span></div>' if len(known) > 1 else ''
            body = f'{visual}<div class="weight-reading">{measure}{comparison}</div>'
        else:
            body = f'{measure}<p class="chart-description">{detail}</p>{visual}'
        panels.append((key,
            f'<article class="health-chart health-chart-{key}"><header><h3>{title}</h3><span>{context}</span></header>{body}'
            f'<details><summary>查看每日{title} · {len(known)} 天有记录</summary>{extra_chart}<ul class="chart-records">{entries}</ul></details></article>'
        ))

    order = {"exercise": 0, "water": 1, "meal": 2, "weight": 3}
    panels.sort(key=lambda panel: order[panel[0]])
    return '<section class="health-trend-visuals"><div class="health-chart-grid">' + ''.join(panel for _, panel in panels) + '</div><p class="chart-guide">只展示已确认的记录。未记录不计为零；体重实测点之间的虚线表示间隔日期未记录。</p></section>'



def goal_cards(goals: list[dict[str, Any]], period: dict[str, Any]) -> str:
    cards = []
    days = period["period"]["days"]
    period_labels = {"daily": "每天", "weekly": "每周", "monthly": "每月", "8_weeks": "8 周"}
    status_labels = {"active": "进行中", "paused": "已暂停", "completed": "已完成"}
    for goal in goals:
        if not goal.get("versions"):
            continue
        current = goal["versions"][-1]
        title = escape(str(current.get("title", "健康目标")))
        unit = escape(str(current.get("unit", "")))
        target = float(current["target_value"])
        cadence = current["period"]
        status = current["status"]
        kind = current["goal_type"]
        progress = '<p class="goal-context">此目标暂不支持自动计算进度，可通过对话记录和复盘。</p>'
        if kind in {"water", "exercise", "nutrition"} and status == "active":
            category, field, expected_unit = {"water": ("water", "total_ml", "ml"), "exercise": ("exercise", "total_minutes", "分钟"), "nutrition": ("meal", "calories_kcal", "kcal")}[kind]
            # Only compare compatible units; custom unit labels must not imply conversion.
            if current["unit"] in {expected_unit, "minutes" if kind == "exercise" else expected_unit}:
                actual = float(period[category][field])
                scaled = target * days / {"daily": 1, "weekly": 7, "monthly": 30, "8_weeks": 56}[cadence]
                percent = actual / scaled * 100 if scaled > 0 else 0
                if period[category]["count"]:
                    progress = (
                        f'<div class="goal-measure"><strong>{actual:g} <small>{unit}</small></strong><span>同期目标 {scaled:g} {unit}</span></div>'
                        f'<div class="goal-progress-label"><span>记录进度</span><strong>{percent:.0f}%</strong></div>'
                        f'<meter min="0" max="{scaled:g}" value="{min(actual, scaled):g}" aria-label="{title}：已记录 {actual:g}，同期目标 {scaled:g} {unit}"></meter>'
                        f'<p class="goal-context">已记录量为目标的 {percent:.0f}% · 最近 {days} 天'
                        + (f' · 按{period_labels[cadence]}目标折算' if days != {"daily": 1, "weekly": 7, "monthly": 30, "8_weeks": 56}[cadence] else '')
                        + '。记录可能不完整。</p>'
                    )
                else:
                    progress = f'<p class="goal-context">最近 {days} 天还没有相关记录，暂不计算进度。</p>'
        elif kind == "weight" and status == "active":
            change = period["weight"].get("change_kg")
            fact = f'最近 {days} 天体重变化 {change:+g} kg。' if change is not None else f'最近 {days} 天的体重记录不足两次。'
            progress = f'<p class="goal-context">{fact}需确认目标起始体重与目标含义后，才能计算完成比例。</p>'
        if status != "active":
            progress = f'<p class="goal-context">目标{status_labels.get(status, status)}，不计算当前进度。</p>'
        cards.append(
            f'<article class="health-goal health-goal-{escape(kind)}"><header><h3>{title}</h3><span class="goal-state">{escape(status_labels.get(status, status))}</span></header>'
            f'<p class="goal-target">{escape(period_labels.get(cadence, cadence))} · 目标 {target:g} {unit}</p>{progress}'
            f'<details><summary>目标版本与更新时间</summary><p>第 {escape(str(current.get("version", len(goal["versions"]))))} 版 · 更新于 {escape(str(current.get("created_at", "未记录"))[:10])}</p></details></article>'
        )
    return '<section class="health-goal-list">' + (''.join(cards) or '<p>还没有健康目标。点击“创建目标”，从一个想改善的习惯开始。</p>') + '</section>'
