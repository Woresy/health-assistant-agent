"""Request-scoped execution events; never expose model reasoning or tool arguments."""
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from functools import wraps
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Callable, Iterator

_sink: ContextVar[Callable[[str], None] | None] = ContextVar('agent_progress_sink', default=None)

TOOL_LABELS = {
    'get_user_profile': '读取个人设置', 'get_health_goals': '读取健康目标',
    'get_health_events': '查询健康记录', 'query_health_events': '查询健康记录',
    'get_daily_summary': '汇总今日记录', 'get_period_summary': '汇总近期记录',
    'get_daily_health_summary': '汇总今日记录',
    'prepare_health_event': '生成待确认的健康记录',
    'prepare_event_change': '准备记录修改草稿',
    'prepare_profile_update': '准备个人设置草稿', 'prepare_goal_change': '准备目标草稿',
    'retrieve_nutrition_candidates': '查询本地营养数据',
    'calculate_nutrition': '按份量计算营养', 'detect_food': '识别餐食与份量',
    'retrieve_health_knowledge': '检索健康知识来源',
    'create_reminder_draft': '准备提醒草稿', 'list_or_cancel_reminders': '核对提醒状态',
}


def report_progress(message: str) -> None:
    sink = _sink.get()
    if sink is not None:
        sink(message)


def report_tool(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        label = TOOL_LABELS.get(kwargs.get('tool_name'), '执行健康工具')
        report_progress('正在' + label)
        try:
            result = function(*args, **kwargs)
        except Exception:
            report_progress(label + '遇到问题')
            raise
        status = getattr(result, 'status', '')
        detail = getattr(result, 'result', None)
        if status == 'needs_clarification':
            report_progress(label + '：需要补充信息')
        elif status == 'invalid' or (isinstance(detail, dict) and not detail.get('ok', True)):
            report_progress(label + '：未完成，正在整理说明')
        else:
            report_progress(label + '步骤已结束')
        return result
    return wrapped


@dataclass
class ProgressUpdate:
    steps: list[str]
    elapsed_seconds: int
    done: bool = False
    result: Any = None


def run_with_progress(function: Callable[[], Any], *, interval: float = .3) -> Iterator[ProgressUpdate]:
    """Run existing synchronous work while streaming real steps and a waiting heartbeat."""
    events: Queue = Queue()
    started = monotonic()
    steps = ['已收到请求，正在准备处理']
    yield ProgressUpdate(list(steps), 0)
    def work():
        token = _sink.set(lambda message: events.put(('step', message)))
        try:
            events.put(('result', function()))
        except Exception as exc:
            events.put(('error', exc))
        finally:
            _sink.reset(token)
    context = copy_context()
    thread = Thread(target=lambda: context.run(work), daemon=True)
    thread.start()
    while True:
        try:
            kind, value = events.get(timeout=interval)
        except Empty:
            yield ProgressUpdate(list(steps), int(monotonic() - started))
            continue
        if kind == 'error':
            raise value
        if kind == 'result':
            thread.join()
            yield ProgressUpdate(list(steps), int(monotonic() - started), True, value)
            return
        if not steps or steps[-1] != value:
            steps.append(value)
        yield ProgressUpdate(list(steps), int(monotonic() - started))
