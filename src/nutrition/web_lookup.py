"""Opt-in web nutrition lookup. Only explicit per-100g source tables become records."""
from datetime import date
from hashlib import sha256
import json
import os
import re
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.agent.progress import report_progress
from src.nutrition.repository import FoodRecord
from src.nutrition.text_normalize import normalize_text

# Deliberately limited to food databases; general search snippets are not nutrition evidence.
SOURCE_DOMAINS = ('boohee.com', 'hpcn21.com', 'fatsecret.cn')


class WebNutritionError(Exception):
    pass


def _allowed_source(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    return (parsed.scheme == 'https' and not parsed.username and not parsed.password
            and parsed.port in (None, 443)
            and any(host == d or host.endswith('.' + d) for d in SOURCE_DOMAINS))


def parse_source(query: str, source: dict) -> FoodRecord | None:
    """Extract one complete table with an explicit mass basis; never fill missing values."""
    url = str(source.get('url', ''))
    try:
        if not _allowed_source(url) or len(url) > 600:
            return None
    except ValueError:
        return None
    title = str(source.get('title', ''))[:200]
    # A differently named dish needs explicit matching logic, not first-hit selection.
    source_name = re.split(r'的热量|的营养|营养成分|热量', title, maxsplit=1)[0].strip(' _-')
    if normalize_text(query) != normalize_text(source_name):
        return None
    if any(word in title and word not in query for word in ('干粉', '干制', '方便', '速食', '冲泡', '袋装')):
        return None
    content = str(source.get('raw_content') or '')[:80000]
    content = re.sub(r'<[^>]+>', ' ', content)
    basis = re.search(r'(?:每\s*100\s*(?:克|g)|100\s*克可食部分)', content, re.I)
    if not basis:
        return None
    # Do not consume another food's table or a different serving size below this table.
    table = content[basis.start():basis.start() + 1800]
    if re.search(r'每\s*(?:份|碗|盘|包)', table):
        return None
    values = {}
    for field, label, unit in (
        ('calories_per_100g', '热量|能量', '千卡|大卡|kcal'),
        ('protein_per_100g', '蛋白质', '克|g'),
        ('fat_per_100g', '脂肪', '克|g'),
        ('carbs_per_100g', '碳水化合物|碳水', '克|g'),
    ):
        sep = r'[\s|:：*（）()]*'
        # Known databases display label (unit) then value, or value+unit then label.
        match = re.search(rf'(?:{label}){sep}(?:{unit}){sep}([0-9]+(?:\.[0-9]+)?)', table, re.I)
        if match is None:
            match = re.search(rf'([0-9]+(?:\.[0-9]+)?){sep}(?:{unit}){sep}(?:{label})', table, re.I)
        if match is None:
            return None
        values[field] = float(match.group(1))
    if values['calories_per_100g'] > 950 or sum(values[k] for k in values if k != 'calories_per_100g') > 105:
        return None
    return FoodRecord(
        food_id='WEB_' + sha256((query + url).encode()).hexdigest()[:20],
        name=query, aliases=[], category='联网食物参考', **values,
        source=url, source_version=f'联网查询 {date.today().isoformat()}；条目：{title}',
        updated_at=date.today(), quality_flags=['web_reference', 'recipe_may_differ'],
    )


class WebNutritionLookup:
    def __init__(self, *, api_key: str | None = None, search=None):
        self.api_key = api_key if api_key is not None else os.getenv('TAVILY_API_KEY', '').strip()
        self._search = search or self._request

    def _request(self, query: str) -> list[dict]:
        payload = {
            'api_key': self.api_key,
            'query': f'{query} 营养成分 每100克 热量 蛋白质 脂肪 碳水化合物',
            'search_depth': 'basic', 'max_results': 5,
            'include_raw_content': True, 'include_domains': list(SOURCE_DOMAINS),
        }
        request = Request('https://api.tavily.com/search',
                          data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=12) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError('response too large')
                data = json.loads(raw)
            results = data['results']
            if not isinstance(results, list):
                raise ValueError('invalid results')
            return [item for item in results[:5] if isinstance(item, dict)]
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            # No provider error text: it may contain credentials or request content.
            raise WebNutritionError('联网营养查询失败，请稍后重试；未使用猜测数值') from exc

    def lookup(self, query: str) -> FoodRecord:
        if os.getenv('NUTRITION_WEB_ENABLED', 'true').lower() == 'false':
            raise WebNutritionError('联网营养查询已关闭')
        if not self.api_key:
            raise WebNutritionError('本地没有匹配数据；联网查询尚未配置 TAVILY_API_KEY')
        report_progress(f'本地未匹配「{query}」，正在联网查询营养来源')
        for source in self._search(query):
            food = parse_source(query, source)
            if food is not None:
                report_progress(f'已找到「{query}」每 100 克营养来源，正在按份量计算')
                return food
        raise WebNutritionError('未找到同名食物的完整每 100 克营养数据；不能用一份热量或干粉数据代替')
