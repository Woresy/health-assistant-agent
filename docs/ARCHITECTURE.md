# 系统架构

## 目标与边界

本项目是单用户、本地运行的健康记录助理。LangGraph 负责 Agent 编排，不负责
生成营养数值，也不替代领域工具、确认协议或持久化存储。

业务事实边界：

- HealthEvent：`data/health_events.jsonl`；
- Agent Trace：`data/agent_traces.jsonl`；
- 营养检索 Trace：`data/traces.jsonl`；
- LangGraph checkpoint：P0 使用进程内 `InMemorySaver`，只保存短期会话状态；
- 图片默认不长期保存；
- HealthOS P1 档案、目标版本和提醒：`data/healthos_state.json`；
- 对话会话：`data/conversations/conversation-*.json`，浏览器刷新后可恢复；
- 模型不能直接执行保存、修改或删除。

## 分层结构

```mermaid
flowchart TD
    UI[Gradio UI] --> SESSION[Traced Conversation Session]
    SESSION --> ORCH{AGENT_ORCHESTRATOR}
    ORCH -->|langgraph| GRAPH[LangGraphAgentRunner]
    ORCH -->|legacy| LOOP[AgentRunner]

    GRAPH --> MODEL[AgentModel Protocol]
    LOOP --> MODEL
    GRAPH --> ROUTER[HealthToolRouter]
    LOOP --> ROUTER

    ROUTER --> READ[Query / Daily Summary]
    ROUTER --> DRAFT[Prepare Save / Update / Delete]
    DRAFT --> TOKEN[Signed Confirmation Token]
    TOKEN --> WRITE[Save / Update / Delete]
    READ --> STORE[HealthEvent JSONL]
    WRITE --> STORE

    SESSION -.脱敏元数据.-> TRACE[AgentTrace JSONL]
```

## LangGraph 节点

```mermaid
flowchart TD
    START --> MODEL[call_model]
    MODEL --> ROUTE{响应类型}
    ROUTE -->|文本| TEXT[handle_text]
    TEXT -->|必须调用工具| MODEL
    TEXT -->|完成| END

    ROUTE -->|单个工具| DISPATCH[dispatch_tool]
    ROUTE -->|多个工具| PARALLEL[reject_parallel_tools]
    PARALLEL --> END

    DISPATCH -->|只读成功| MODEL
    DISPATCH -->|缺参| CLARIFY[await_clarification interrupt]
    CLARIFY -->|补充参数| MODEL
    CLARIFY -->|取消| END

    DISPATCH -->|写操作草稿| APPROVE[await_confirmation interrupt]
    APPROVE -->|确认| EXECUTE[execute_confirmation]
    APPROVE -->|取消| END
    EXECUTE -->|成功| END
    EXECUTE -->|失败可重试| APPROVE

    DISPATCH -->|非法或失败| END
```

## 图状态

checkpoint 中只保存编排需要的数据：

- `session_id`、`user_id`、`timezone_name`；
- 标准化消息；
- `agent_state`、`finish_reason`、`model_rounds`；
- 脱敏展示使用的 `tool_steps`；
- `pending_task`；
- `pending_confirmation`；
- 当前模型响应和下一跳。

P0 的 checkpoint 位于内存中，避免把完整短期对话和健康工具结果额外持久化。
如果以后改为 SQLite/Postgres checkpointer，必须先补充保留期限、清除、加密和
敏感字段审查。

## 缺参流程

```text
用户：我跑步了
→ 模型调用 prepare_health_event(activity_type=跑步)
→ Router 返回 needs_clarification(duration_minutes)
→ 图停在 await_clarification
→ UI 展示“持续了多少分钟”
→ 用户：30分钟
→ Command(resume={action: clarify, text: 30分钟})
→ 模型重新调用工具，已知参数与新参数合并
→ 生成草稿
→ 图停在 await_confirmation
```

取消缺参任务会清除 pending task，不写 HealthEvent。

## 写操作确认流程

```text
模型提出 prepare_* 工具调用
→ Router 校验参数
→ 确定性代码生成草稿、确认令牌和幂等键
→ checkpoint 保存草稿
→ await_confirmation interrupt
→ 用户确认
→ Command(resume={action: confirm})
→ execute_confirmation 调用真正写工具
→ 工具再次校验确认令牌和幂等键
→ 写入成功后状态才变为 completed
```

草稿生成和 interrupt 位于不同节点，避免恢复 interrupt 时重复生成令牌。真正的
写工具位于 interrupt 之后；即使执行节点重试，幂等键也阻止重复副作用。

## 失败与回滚

- 未知工具：Router 拒绝，图进入 `failed`，不写事件；
- 参数非法：Pydantic 校验失败，不写事件；
- 缺参：图暂停，不产生半成品 HealthEvent；
- 用户取消：清除 pending 状态，不写或修改事件；
- 写入失败：保留待确认草稿并再次暂停，允许安全重试或取消；
- Trace 写入失败：向开发者证据区返回警告，不回滚已经成功的 HealthEvent；
- 达到模型轮次上限：终止为 `loop_limit`，不把模型文本当作工具成功。

## Legacy 回退

```dotenv
AGENT_ORCHESTRATOR=legacy
```

该配置切回原有 `AgentRunner`。两种实现共享模型协议、HealthToolRouter、领域工具、
JSONL Store 和 Trace，因此回退不会迁移或改变已保存健康事件。

## HealthOS P1 工具与状态

Router 对模型只公开 PRD 规定的 15 个工具：档案 2 个、目标 2 个、健康事实 3 个、
营养 2 个、知识 1 个、汇总 2 个和提醒 3 个。P0 的旧工具名称只在内部保留为
兼容别名，不再出现在 Provider Tool Schema 中。

档案、目标和提醒使用同一个确认中间件：

```text
prepare_* / create_* / list_or_cancel_*(写意图)
→ 严格 Schema 与领域校验
→ 返回无副作用草稿 + 短时签名令牌 + 幂等键
→ 用户确认
→ Router.confirm
→ 再次验证 action、user、payload 摘要与幂等键
→ 原子替换 healthos_state.json
```

目标保存不可变版本数组；提醒保存每次状态转换；档案只保存明确允许的最小字段。
模型推断的疾病、心理状态和原始敏感 Prompt 不进入长期状态。周期汇总只从 committed
HealthEvent 计算，不把缺失日期补成事实，也不解释变化原因。
