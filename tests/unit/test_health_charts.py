from src.ui.health_charts import goal_cards, trend_charts


def sample(weight=None, water=None):
    return {
        'meal': {'count': 0, 'calories_kcal': 0},
        'water': {'count': int(water is not None), 'total_ml': water or 0},
        'exercise': {'count': 0, 'total_duration_minutes': 0},
        'weight': {'count': int(weight is not None), 'latest_weight_kg': weight},
    }


def test_missing_days_stay_explicit_in_weight_comparisons():
    html = trend_charts([('09/01', sample(70, 350)), ('09/02', sample()), ('09/03', sample(69))])
    assert '有记录日均 350 ml' in html
    assert 'chart-line"' not in html
    assert 'weight-gap' in html
    assert '较观察期首次记录' in html
    assert '-1 kg' in html
    assert '虚线跨过未记录日期' in html
    assert '未记录</strong>' in html


def period():
    return {'period': {'days': 7}, 'exercise': {'count': 1, 'total_minutes': 32}, 'weight': {'change_kg': -1}}


def goal(**overrides):
    return {'versions': [dict(title='每周运动 <180>', goal_type='exercise', target_value=180, unit='分钟', period='weekly', status='active', **overrides)]}


def test_weekly_goal_compares_same_period_and_escapes_title():
    html = goal_cards([goal()], period())
    assert '同期目标 180 分钟' in html
    assert '18%' in html
    assert '&lt;180&gt;' in html


def test_weight_and_paused_goals_do_not_invent_completion():
    weight = goal()
    weight['versions'][0].update(goal_type='weight', unit='kg', target_value=4, period='8_weeks')
    html = goal_cards([weight], period())
    assert '<meter' not in html
    assert '目标起始体重' in html
    paused = goal()
    paused['versions'][0]['status'] = 'paused'
    assert '<meter' not in goal_cards([paused], period())


def test_missing_and_incompatible_records_do_not_show_zero_progress():
    data = period()
    data['exercise'] = {'count': 0, 'total_minutes': 0}
    assert '<meter' not in goal_cards([goal()], data)
    incompatible = goal()
    incompatible['versions'][0]['unit'] = '小时'
    assert '<meter' not in goal_cards([incompatible], period())


def test_cups_represent_volume_without_assuming_a_goal():
    html = trend_charts([('09/01', sample(water=350))])
    assert '每杯 250 ml' in html
    assert '不代表目标' in html
    assert 'id="water-cup-1"><rect x="0" y="23.00" width="32" height="12.00"' in html
    assert '运动足迹' in html


def test_calorie_ruler_and_macros_use_only_recorded_meals():
    recorded = sample()
    recorded['meal'] = {'count': 2, 'calories_kcal': 1843, 'carbs_g': 269, 'protein_g': 164, 'fat_g': 110}
    html = trend_charts([('09/01', recorded), ('09/02', sample())])
    assert '累计已记录 1843 kcal，每格 50 kcal' in html
    assert '269.0<small> g' in html
    assert '164.0<small> g' in html
    assert '110.0<small> g' in html
    assert '非摄入目标' in html


def test_single_weight_and_absent_meals_do_not_invent_trends_or_macros():
    html = trend_charts([('09/01', sample(weight=70))])
    assert '目前只有一天的记录' in html
    assert '较观察期首次记录' not in html
    assert '—<small> g' in html
