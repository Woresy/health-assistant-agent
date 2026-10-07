"""Only sourced per-100g values may repair an unmatched photo meal."""
import pytest
from src.nutrition.web_lookup import WebNutritionLookup, WebNutritionError, parse_source

SOURCE = {
    'title': '煎鱼块的热量与营养成分',
    'url': 'https://www.boohee.com/shiwu/test_fish',
    'raw_content': '营养素含量（每100克）\n热量（大卡）180\n蛋白质（克）20\n脂肪（克）10\n碳水化合物（克）2',
}


def test_source_table_preserves_values_and_url():
    food = parse_source('煎鱼块', SOURCE)
    assert food.calories_per_100g == 180
    assert food.protein_per_100g == 20
    assert food.source == SOURCE['url']
    assert food.food_id.startswith('WEB_')


@pytest.mark.parametrize('changes', [
    {'url': 'https://www.boohee.com.evil.example/shiwu/fish'},
    {'url': 'http://127.0.0.1/food'},
    {'url': 'javascript:alert(1)'},
    {'title': '生鱼的营养成分'},
    {'title': '某品牌 煎鱼块的热量'},
    {'raw_content': SOURCE['raw_content'].replace('蛋白质', '每份 蛋白质')},
    {'raw_content': SOURCE['raw_content'].replace('每100克', '每份')},
    {'raw_content': SOURCE['raw_content'].replace('脂肪（克）10', '')},
    {'raw_content': SOURCE['raw_content'].replace('180', '9999')},
    {'raw_content': SOURCE['raw_content'].replace('大卡', '千焦')},
])
def test_unverifiable_data_is_not_used(changes):
    assert parse_source('煎鱼块', {**SOURCE, **changes}) is None


def test_network_disabled_or_unconfigured_is_actionable(monkeypatch):
    monkeypatch.setenv('NUTRITION_WEB_ENABLED', 'false')
    with pytest.raises(WebNutritionError, match='关闭'):
        WebNutritionLookup(api_key='test').lookup('煎鱼块')
    monkeypatch.setenv('NUTRITION_WEB_ENABLED', 'true')
    with pytest.raises(WebNutritionError, match='TAVILY_API_KEY'):
        WebNutritionLookup(api_key='').lookup('煎鱼块')


def test_missing_local_food_gets_sourced_draft_and_saves(tmp_path, monkeypatch):
    from src.ui import app
    from src.storage.jsonl_store import HealthEventStore
    monkeypatch.setenv('NUTRITION_WEB_ENABLED', 'true')
    lookup = WebNutritionLookup(api_key='test', search=lambda query: [SOURCE])
    monkeypatch.setattr(app, 'WebNutritionLookup', lambda: lookup)
    store = HealthEventStore(tmp_path / 'events.jsonl')
    monkeypatch.setattr(app, 'event_store', store)
    preview, draft = app.prepare_detected_meal('tests/fixtures/meal.png', {
        'detections': [{'suggested_query': '煎鱼块', 'estimated_grams': 90}],
    })
    assert draft is not None, preview
    assert '查看参考资料' in preview and SOURCE['url'] in preview
    assert not store.path.exists()
    nutrition = draft['items'][0]['event']['payload']['nutrition']
    assert nutrition['calories_kcal'] == 162
    assert nutrition['protein_g'] == 18
    assert '图片估算' in nutrition['portion_assumption']
    app.confirm_meal_save(draft, [])
    assert SOURCE['url'] in store.path.read_text()


def test_network_failure_does_not_discard_local_preview(monkeypatch):
    from src.ui import app
    def fail(*args):
        raise WebNutritionError('联网查询超时')
    monkeypatch.setattr(app, 'prepare_web_meal_draft', fail)
    preview, draft = app.prepare_detected_meal('tests/fixtures/meal.png', {
        'detections': [{'suggested_query': '苹果', 'estimated_grams': 100},
                       {'suggested_query': '煎鱼块', 'estimated_grams': 90}],
    })
    assert '苹果' in preview and '联网查询超时' in preview
    assert '换一张清晰照片' not in preview
    assert draft is None


def test_search_request_only_sends_food_name_and_uses_raw_sources(monkeypatch):
    import json
    from io import BytesIO
    from src.nutrition import web_lookup
    requests = []
    def response(request, timeout):
        requests.append(json.loads(request.data))
        assert request.full_url == 'https://api.tavily.com/search'
        assert timeout == 12
        return BytesIO(json.dumps({'results': [SOURCE]}).encode())
    monkeypatch.setattr(web_lookup, 'urlopen', response)
    monkeypatch.setenv('NUTRITION_WEB_ENABLED', 'true')
    food = WebNutritionLookup(api_key='test-key').lookup('煎鱼块')
    assert food.calories_per_100g == 180
    assert requests[0]['include_raw_content'] is True
    assert 'image' not in requests[0] and 'user_id' not in requests[0]


def test_provider_error_never_exposes_credentials(monkeypatch):
    from urllib.error import URLError
    from src.nutrition import web_lookup
    def fail(*args, **kwargs):
        raise URLError('secret-test-key')
    monkeypatch.setattr(web_lookup, 'urlopen', fail)
    monkeypatch.setenv('NUTRITION_WEB_ENABLED', 'true')
    with pytest.raises(WebNutritionError) as error:
        WebNutritionLookup(api_key='secret-test-key').lookup('煎鱼块')
    assert 'secret-test-key' not in str(error.value)
