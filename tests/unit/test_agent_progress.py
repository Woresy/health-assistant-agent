from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import pytest
from src.agent.progress import report_progress, report_tool, run_with_progress


def test_steps_arrive_before_result_and_waiting_does_not_invent_steps():
    release = Event()
    def work():
        report_progress('正在查询本地营养数据')
        assert release.wait(2)
        return 'done'
    stream = run_with_progress(work, interval=.01)
    assert not next(stream).done
    step = next(stream)
    assert step.steps[-1] == '正在查询本地营养数据'
    waiting = next(stream)
    assert waiting.steps == step.steps and not waiting.done
    release.set()
    assert list(stream)[-1].result == 'done'


def test_parallel_requests_do_not_share_progress():
    def run(name):
        def work():
            report_progress(name)
            return name
        return list(run_with_progress(work))
    with ThreadPoolExecutor(2) as executor:
        a, b = list(executor.map(run, ['session-a', 'session-b']))
    assert 'session-b' not in str(a) and 'session-a' not in str(b)


def test_tool_events_hide_arguments_and_report_failure():
    @report_tool
    def tool(**kwargs):
        return SimpleNamespace(status='executed', result={'ok': False})
    events = list(run_with_progress(lambda: tool(tool_name='get_health_events', arguments={'secret':'do-not-display'})))
    text = str([e.steps for e in events])
    assert '正在查询健康记录' in text and '未完成' in text
    assert 'do-not-display' not in text


def test_background_error_is_propagated():
    def work():
        raise ValueError('failed')
    with pytest.raises(ValueError):
        list(run_with_progress(work))


def test_chat_stream_yields_progress_then_unchanged_reply(monkeypatch):
    from src.ui import app
    def send(*args):
        report_progress('正在查询健康记录')
        return tuple(range(10))
    monkeypatch.setattr(app, 'send_chat_message', send)
    updates = list(app.send_chat_message_stream('查询今天', [], {}, None))
    assert updates[-1][:10] == tuple(range(10))
    assert len(updates) >= 3
    assert '正在查询健康记录' in updates[-1][-1].value


def test_langgraph_propagates_progress_context(tmp_path):
    from src.agent.langgraph_runner import LangGraphAgentRunner
    from src.agent.models import AgentModelReply
    from src.agent.runner import ConversationSession
    from src.agent.tool_router import HealthToolRouter
    from src.storage.jsonl_store import HealthEventStore
    class Model:
        def complete(self, *args):
            return AgentModelReply(content='你好')
    runner = LangGraphAgentRunner(model=Model(), router=HealthToolRouter(HealthEventStore(tmp_path / 'events.jsonl')))
    session = ConversationSession(runner=runner, session_id='progress-test', user_id='test')
    updates = list(run_with_progress(lambda: session.send('你好')))
    assert any('正在分析请求并整理下一步' in update.steps for update in updates)
