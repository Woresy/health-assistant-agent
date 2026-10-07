"""多食物图片 → 估量 → 整餐确认 → 持久化，无外部调用。"""
import json
from pathlib import Path

import pytest

from src.ui import app
from src.storage.jsonl_store import HealthEventStore
from src.vision.vlm_detector import VlmFoodDetector
from unittest.mock import Mock
from src.vision.detector import DetectionError

IMAGE = str(Path(__file__).parents[1] / 'fixtures' / 'meal.png')


def detected_data(foods=None):
    if foods is None:
        foods = [
            {'name': '苹果', 'confidence': .92, 'estimated_grams': 180, 'portion_description': '约一个'},
            {'name': '香蕉', 'confidence': .88, 'estimated_grams': 100, 'portion_description': '约一根'},
        ]
    client = Mock()
    client.chat.completions.create.return_value.choices = [Mock(message=Mock(content=json.dumps({'foods': foods})))]
    result = VlmFoodDetector(client, 'test').detect(IMAGE)
    return result.model_dump(mode='json')


def test_multi_meal_confirm_and_retry_without_duplicate(tmp_path, monkeypatch):
    store = HealthEventStore(tmp_path / 'events.jsonl')
    monkeypatch.setattr(app, 'event_store', store)
    preview, draft = app.prepare_detected_meal(IMAGE, detected_data())
    assert '共 2 份食物' in preview and '约 180 克' in preview and '约 100 克' in preview
    assert draft and len(draft['items']) == 2
    assert not store.path.exists()
    result = app.confirm_meal_save(draft, [])
    assert result[4] is None
    rows = store.path.read_text().splitlines()
    assert len(rows) == 2
    assert [json.loads(row)['payload']['portion']['grams'] for row in rows] == [180, 100]
    app.confirm_meal_save(draft, [])
    assert len(store.path.read_text().splitlines()) == 2


@pytest.mark.parametrize('change', [
    {'estimated_grams': None}, {'name': '不存在的未知菜品'}, {'confidence': .1},
])
def test_incomplete_meal_never_creates_partial_draft(change):
    foods = [{'name': '苹果', 'confidence': .9, 'estimated_grams': 180},
             {'name': '香蕉', 'confidence': .9, 'estimated_grams': 100, **change}]
    preview, draft = app.prepare_detected_meal(IMAGE, detected_data(foods))
    assert draft is None
    assert '还没估出来' in preview


def test_same_food_portions_are_summed():
    data = detected_data([
        {'name': '苹果', 'confidence': .9, 'estimated_grams': 180},
        {'name': '苹果', 'confidence': .8, 'estimated_grams': 100},
    ])
    assert len(data['detections']) == 1
    assert data['detections'][0]['estimated_grams'] == 280


@pytest.mark.parametrize('grams', [0, -1, 10001, float('inf'), float('nan')])
def test_invalid_portions_rejected(grams):
    with pytest.raises(DetectionError):
        detected_data([{'name': '苹果', 'confidence': .9, 'estimated_grams': grams}])


def test_more_than_five_foods_are_preserved():
    data = detected_data([
        {'name': f'菜品{i}', 'confidence': .9, 'estimated_grams': 100} for i in range(7)
    ])
    assert len(data['detections']) == 7


def test_vlm_upload_has_no_manual_fields_and_clears_old_draft(monkeypatch):
    from src.vision.config import FoodDetectionConfig
    monkeypatch.setattr(app, 'DETECTION_CONFIG', FoodDetectionConfig(mode='vlm'))
    monkeypatch.setattr(app, 'detect_food', lambda *a, **kw: {'ok': True, 'data': detected_data()})
    result = app.open_meal_confirmation(IMAGE)
    assert result[5].visible is False
    assert len(result[9]['items']) == 2
    assert result[11] == ''
    cleared = app.open_meal_confirmation(None)
    assert cleared[9] is None
    assert cleared[0].visible is False


def test_partial_write_can_resume(tmp_path, monkeypatch):
    store = HealthEventStore(tmp_path / 'events.jsonl')
    monkeypatch.setattr(app, 'event_store', store)
    _, draft = app.prepare_detected_meal(IMAGE, detected_data())
    save = app.save_health_event
    calls = 0
    def flaky(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {'ok': False, 'error': {'error_code': 'WRITE_FAILED', 'message': 'test failure'}}
        return save(**kwargs)
    monkeypatch.setattr(app, 'save_health_event', flaky)
    result = app.confirm_meal_save(draft, [])
    assert '已保存 1/2 项' in result[0]
    assert len(store.path.read_text().splitlines()) == 1
    app.confirm_meal_save(draft, [])
    assert len(store.path.read_text().splitlines()) == 2


def test_unmatched_meal_box_can_be_confirmed_with_labeled_model_estimate(tmp_path, monkeypatch):
    """用户要按整盒确认；同名营养表缺失不能拦住明确标注的模型估算。"""
    def unavailable(*args):
        raise AssertionError('已有图片营养估算时，不应逐项等待联网检索')
    monkeypatch.setattr(app, 'prepare_web_meal_draft', unavailable)
    monkeypatch.setattr(app, 'event_store', HealthEventStore(tmp_path / 'events.jsonl'))
    data = detected_data([{
        'name': '鸡肉三明治餐盒', 'confidence': .88, 'estimated_grams': 350,
        'group_id': 'box_left', 'portion_description': '左侧约一盒',
        'components': ['鸡肉三明治', '蔬菜配菜', '酱料'],
        'estimated_nutrition': {'calories_per_100g': 200, 'protein_per_100g': 12,
                                'fat_per_100g': 8, 'carbs_per_100g': 20},
    }])
    preview, draft = app.prepare_detected_meal(IMAGE, data)
    assert draft is not None, preview
    assert '这一餐约 700 千卡' in preview
    assert '按照片估计' in preview
    assert len(draft['items']) == 1
    event = draft['items'][0]['event']
    assert event['payload']['nutrition']['calories_kcal'] == 700
    assert '模型估算' in event['payload']['nutrition']['source_ref']
    assert not app.event_store.path.exists()
    app.confirm_meal_save(draft, [])
    assert len(app.event_store.path.read_text().splitlines()) == 1


def test_separate_boxes_with_same_name_are_not_merged():
    data = detected_data([
        {'name': '三明治餐盒', 'group_id': 'left', 'confidence': .9, 'estimated_grams': 300},
        {'name': '三明治餐盒', 'group_id': 'right', 'confidence': .9, 'estimated_grams': 400},
    ])
    assert [item['estimated_grams'] for item in data['detections']] == [300, 400]


def test_same_container_is_not_counted_twice():
    item = {'name': '三明治餐盒', 'group_id': 'left', 'confidence': .9, 'estimated_grams': 300}
    assert len(detected_data([item, item])['detections']) == 1


@pytest.mark.parametrize('values', [
    {'calories_per_100g': float('nan')}, {'fat_per_100g': -1},
    {'protein_per_100g': 70, 'fat_per_100g': 70}, {'calories_per_100g': 2000},
])
def test_invalid_model_nutrition_cannot_be_saved(values):
    with pytest.raises(DetectionError):
        detected_data([{
            'name': '三明治餐盒', 'confidence': .9, 'estimated_grams': 300,
            'estimated_nutrition': {'calories_per_100g': 200, 'protein_per_100g': 12,
                                    'fat_per_100g': 8, 'carbs_per_100g': 20, **values},
        }])


def test_mixed_box_does_not_apply_rice_table_to_whole_box(monkeypatch):
    def wrong_path(*args, **kwargs):
        raise AssertionError('不能将米饭单品的每100克数据用于含肉菜的整盒重量')
    monkeypatch.setattr(app, 'refresh_meal_panel', wrong_path)
    data = detected_data([{
        'name': '米饭', 'group_id': 'box', 'confidence': .9, 'estimated_grams': 400,
        'components': ['米饭', '鸡肉', '蔬菜'],
        'estimated_nutrition': {'calories_per_100g': 180, 'protein_per_100g': 12,
                                'fat_per_100g': 8, 'carbs_per_100g': 15},
    }])
    preview, draft = app.prepare_detected_meal(IMAGE, data)
    assert draft is not None, preview
    nutrition = draft['items'][0]['event']['payload']['nutrition']
    assert nutrition['calories_kcal'] == 720
    assert '米饭、鸡肉、蔬菜' in nutrition['portion_assumption']
