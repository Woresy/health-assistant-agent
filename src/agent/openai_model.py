"""OpenAI-compatible Agent 模型适配器。"""

from __future__ import annotations

import json
import os
from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from src.agent.models import (
    AgentMessage,
    AgentModelReply,
    ModelToolCall,
)


PROJECT_ROOT = (
    Path(__file__).resolve()
    .parents[2]
)
DEFAULT_ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)


class AgentConfigurationError(
    Exception
):
    """Agent Provider 配置错误。"""


class AgentProviderError(
    Exception
):
    """Agent Provider 调用失败。"""


class AgentProtocolError(
    AgentProviderError
):
    """Provider 响应不符合 Agent 协议。"""


@dataclass(frozen=True)
class AgentProviderSettings:
    """校验后的 Provider 配置。"""

    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int
    max_tokens: int


def _parse_positive_float(
    *,
    value: str,
    field_name: str,
    maximum: float,
) -> float:
    """解析大于零的浮点配置。"""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise AgentConfigurationError(
            f"{field_name} 必须是数字"
        ) from exc

    if (
        parsed <= 0
        or parsed > maximum
    ):
        raise AgentConfigurationError(
            f"{field_name} 必须大于 0 "
            f"且不超过 {maximum:g}"
        )

    return parsed


def _parse_integer(
    *,
    value: str,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    """解析整数配置。"""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise AgentConfigurationError(
            f"{field_name} 必须是整数"
        ) from exc

    if (
        parsed < minimum
        or parsed > maximum
    ):
        raise AgentConfigurationError(
            f"{field_name} 必须在 "
            f"{minimum} 到 {maximum} 之间"
        )

    return parsed


def _validate_base_url(
    raw_value: str,
) -> str:
    """验证 Provider Base URL。"""

    normalized = (
        raw_value.strip()
        .rstrip("/")
    )

    if not normalized:
        raise AgentConfigurationError(
            "AGENT_BASE_URL 不能为空"
        )

    parsed = urlparse(
        normalized
    )

    if (
        parsed.scheme
        not in {
            "http",
            "https",
        }
        or not parsed.netloc
    ):
        raise AgentConfigurationError(
            "AGENT_BASE_URL 必须是有效的 "
            "HTTP 或 HTTPS 地址"
        )

    return normalized


def load_agent_settings(
    env_file: (
        str
        | Path
        | None
    ) = None,
) -> AgentProviderSettings | None:
    """
    读取 Agent Provider 配置。

    disabled 返回 None，使页面能在没有 API Key 时启动。
    """

    selected_env_file = (
        Path(env_file)
        if env_file is not None
        else DEFAULT_ENV_FILE
    )

    load_dotenv(
        dotenv_path=(
            selected_env_file
        ),
        override=False,
    )

    provider_mode = os.getenv(
        "AGENT_PROVIDER_MODE",
        "disabled",
    ).strip().lower()

    if provider_mode == "disabled":
        return None

    if (
        provider_mode
        != "openai_compatible"
    ):
        raise AgentConfigurationError(
            "AGENT_PROVIDER_MODE 只能是 "
            "disabled 或 openai_compatible"
        )

    api_key = os.getenv(
        "AGENT_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise AgentConfigurationError(
            "启用 Agent 后必须填写 "
            "AGENT_API_KEY"
        )

    model = os.getenv(
        "AGENT_MODEL",
        "",
    ).strip()

    if not model:
        raise AgentConfigurationError(
            "启用 Agent 后必须填写 "
            "AGENT_MODEL"
        )

    base_url = _validate_base_url(
        os.getenv(
            "AGENT_BASE_URL",
            "https://api.openai.com/v1",
        )
    )

    timeout = (
        _parse_positive_float(
            value=os.getenv(
                "AGENT_REQUEST_TIMEOUT",
                "30",
            ),
            field_name=(
                "AGENT_REQUEST_TIMEOUT"
            ),
            maximum=300,
        )
    )

    max_retries = _parse_integer(
        value=os.getenv(
            "AGENT_MAX_RETRIES",
            "2",
        ),
        field_name=(
            "AGENT_MAX_RETRIES"
        ),
        minimum=0,
        maximum=10,
    )

    max_tokens = _parse_integer(
        value=os.getenv(
            "AGENT_MAX_TOKENS",
            "1024",
        ),
        field_name=(
            "AGENT_MAX_TOKENS"
        ),
        minimum=128,
        maximum=8192,
    )

    return AgentProviderSettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )


def _convert_agent_message(
    message: AgentMessage,
) -> dict[str, Any]:
    """
    将内部 AgentMessage 转成 Chat Completions 消息。

    阶段 D 用普通 JSON 保存工具请求；
    这里恢复为 Provider 需要的 tool_calls 结构。
    """

    if (
        message.role
        == "assistant"
        and message.tool_call_id
        is not None
        and message.tool_name
        is not None
    ):
        try:
            raw_body = json.loads(
                message.content
            )
            raw_tool_call = (
                raw_body[
                    "tool_call"
                ]
            )
            arguments = (
                raw_tool_call[
                    "arguments"
                ]
            )
        except (
            json.JSONDecodeError,
            TypeError,
            KeyError,
        ) as exc:
            raise AgentProtocolError(
                "无法恢复 assistant "
                "工具调用消息"
            ) from exc

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": (
                        message
                        .tool_call_id
                    ),
                    "type": "function",
                    "function": {
                        "name": (
                            message
                            .tool_name
                        ),
                        "arguments": (
                            json.dumps(
                                arguments,
                                ensure_ascii=False,
                                separators=(
                                    ",",
                                    ":",
                                ),
                            )
                        ),
                    },
                }
            ],
        }

    if message.role == "tool":
        if (
            message.tool_call_id
            is None
        ):
            raise AgentProtocolError(
                "tool 消息缺少 "
                "tool_call_id"
            )

        return {
            "role": "tool",
            "tool_call_id": (
                message.tool_call_id
            ),
            "content": (
                message.content
            ),
        }

    return {
        "role": message.role,
        "content": message.content,
    }


class OpenAICompatibleAgentModel:
    """通过 Chat Completions tools 调用模型。"""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        normalized_model = (
            model.strip()
        )

        if not normalized_model:
            raise ValueError(
                "model 不能为空"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens 必须大于 0"
            )

        self._client = client
        self._model = (
            normalized_model
        )
        self._max_tokens = (
            max_tokens
        )

    def complete(
        self,
        messages: Sequence[
            AgentMessage
        ],
        tool_definitions: Sequence[
            dict[str, Any]
        ],
    ) -> AgentModelReply:
        """执行一次带工具定义的模型调用。"""

        provider_messages = [
            _convert_agent_message(
                message
            )
            for message in messages
        ]

        try:
            response = (
                self._client
                .chat
                .completions
                .create(
                    model=self._model,
                    messages=(
                        provider_messages
                    ),
                    tools=list(
                        tool_definitions
                    ),
                    tool_choice="auto",
                    max_tokens=(
                        self._max_tokens
                    ),
                )
            )
        except APITimeoutError as exc:
            raise AgentProviderError(
                "模型响应超时，请稍后重试"
            ) from exc
        except AuthenticationError as exc:
            raise AgentProviderError(
                "模型认证失败，请检查 "
                "AGENT_API_KEY"
            ) from exc
        except RateLimitError as exc:
            raise AgentProviderError(
                "模型请求被限流或额度不足"
            ) from exc
        except APIConnectionError as exc:
            raise AgentProviderError(
                "无法连接模型 Provider，"
                "请检查网络和 Base URL"
            ) from exc
        except APIStatusError as exc:
            request_id = (
                getattr(
                    exc,
                    "request_id",
                    None,
                )
                or "unknown"
            )

            raise AgentProviderError(
                "模型 Provider 返回异常："
                f"HTTP {exc.status_code}，"
                f"request_id={request_id}"
            ) from exc
        except OpenAIError as exc:
            raise AgentProviderError(
                "模型 SDK 返回未分类异常"
            ) from exc

        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:
            raise AgentProtocolError(
                "模型响应中没有 choices"
            )

        choice = choices[0]

        if (
            getattr(
                choice,
                "finish_reason",
                None,
            )
            == "length"
        ):
            raise AgentProtocolError(
                "模型响应因长度限制被截断"
            )

        message = getattr(
            choice,
            "message",
            None,
        )

        if message is None:
            raise AgentProtocolError(
                "模型响应中没有 message"
            )

        raw_content = getattr(
            message,
            "content",
            None,
        )

        content = (
            raw_content.strip()
            if isinstance(
                raw_content,
                str,
            )
            and raw_content.strip()
            else None
        )

        raw_tool_calls = (
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

        tool_calls: list[
            ModelToolCall
        ] = []

        for raw_tool_call in (
            raw_tool_calls
        ):
            call_id = getattr(
                raw_tool_call,
                "id",
                None,
            )
            function = getattr(
                raw_tool_call,
                "function",
                None,
            )
            name = getattr(
                function,
                "name",
                None,
            )
            arguments_json = getattr(
                function,
                "arguments",
                None,
            )

            if not isinstance(
                call_id,
                str,
            ):
                raise AgentProtocolError(
                    "模型工具调用缺少 id"
                )

            if not isinstance(
                name,
                str,
            ):
                raise AgentProtocolError(
                    "模型工具调用缺少 name"
                )

            if not isinstance(
                arguments_json,
                str,
            ):
                raise AgentProtocolError(
                    "模型工具调用缺少 "
                    "arguments"
                )

            try:
                arguments = json.loads(
                    arguments_json
                )
            except (
                json.JSONDecodeError
            ) as exc:
                raise AgentProtocolError(
                    "模型工具参数不是有效 JSON"
                ) from exc

            if not isinstance(
                arguments,
                dict,
            ):
                raise AgentProtocolError(
                    "模型工具参数必须是对象"
                )

            tool_calls.append(
                ModelToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        try:
            return AgentModelReply(
                content=content,
                tool_calls=tuple(
                    tool_calls
                ),
            )
        except ValueError as exc:
            raise AgentProtocolError(
                "模型返回了空响应"
            ) from exc


def create_agent_model_from_environment(
) -> tuple[
    OpenAICompatibleAgentModel
    | None,
    str,
]:
    """根据环境变量创建模型及安全状态文本。"""

    settings = (
        load_agent_settings()
    )

    if settings is None:
        return (
            None,
            "Agent 模型未启用。"
            "页面、饮食手动流程、"
            "时间线和汇总仍可使用。",
        )

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=(
            settings.max_retries
        ),
    )

    model = (
        OpenAICompatibleAgentModel(
            client=client,
            model=settings.model,
            max_tokens=(
                settings.max_tokens
            ),
        )
    )

    return (
        model,
        "Agent 模型已启用："
        f"{settings.model}；"
        f"Provider：{settings.base_url}",
    )