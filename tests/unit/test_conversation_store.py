"""本地会话持久化测试。"""

from pathlib import Path

import pytest

from src.agent.models import (
    AgentMessage,
    SessionState,
)
from src.storage.conversation_store import (
    ConversationStore,
)


SESSION_ID = (
    "conversation-"
    "0123456789abcdef0123456789abcdef"
)


def _state() -> SessionState:
    return SessionState(
        session_id=SESSION_ID,
        user_id="local-demo-user",
        messages=(
            AgentMessage(
                role="system",
                content="system",
            ),
            AgentMessage(
                role="user",
                content="我喝了 500 毫升水",
            ),
            AgentMessage(
                role="assistant",
                content="已整理好饮水记录。",
            ),
        ),
        turn_count=1,
    )


def test_save_load_and_delete(
    tmp_path: Path,
) -> None:
    """会话刷新后可恢复，并能被用户明确删除。"""

    store = ConversationStore(
        tmp_path / "conversations"
    )
    state = _state()

    store.save(state)

    assert store.load(SESSION_ID) == state

    store.delete(SESSION_ID)

    assert store.load(SESSION_ID) is None


def test_corrupted_conversation_is_ignored(
    tmp_path: Path,
) -> None:
    """损坏历史不能阻止应用创建新会话。"""

    directory = tmp_path / "conversations"
    directory.mkdir()
    path = directory / f"{SESSION_ID}.json"
    path.write_text("not-json", encoding="utf-8")
    store = ConversationStore(directory)

    assert store.load(SESSION_ID) is None


def test_rejects_unsafe_session_id(
    tmp_path: Path,
) -> None:
    """会话标识不能用于访问存储目录之外的路径。"""

    store = ConversationStore(
        tmp_path / "conversations"
    )

    with pytest.raises(ValueError):
        store.load("../../outside")


def test_lists_only_the_requested_users_conversations(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations")
    first = _state()
    second = first.model_copy(
        update={
            "session_id": "conversation-fedcba9876543210fedcba9876543210",
            "messages": (
                AgentMessage(role="user", content="帮我回看最近的运动记录和饮水情况"),
                AgentMessage(role="assistant", content="可以。"),
            ),
        }
    )
    other_user = first.model_copy(
        update={
            "session_id": "conversation-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "user_id": "other-user",
        }
    )
    store.save(first)
    store.save(second)
    store.save(other_user)

    summaries = store.list_summaries("local-demo-user")

    assert {item.session_id for item in summaries} == {first.session_id, second.session_id}
    assert next(item for item in summaries if item.session_id == second.session_id).title == "帮我回看最近的运动记录和饮水情况"
    assert all(item.user_id == "local-demo-user" for item in summaries)
