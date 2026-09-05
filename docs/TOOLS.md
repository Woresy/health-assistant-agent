# HealthOS Tool 清单与扩展策略

本文以 `src/agent/tool_router.py` 中的 `TOOL_DEFINITIONS` 和
`TOOL_CONTRACTS` 为实现依据，描述当前公开给模型的 15 个受控 Tool。

## 1. 公共约定

模型只填写 Tool Schema 中公开的业务参数。`user_id`、`session_id`、
`call_id`、默认时区、存储对象和大部分幂等键由服务端注入，不能让模型自行决定。

领域工具使用统一结果协议：

```json
{
  "ok": true,
  "data": {},
  "error": null
}
```

失败时：

```json
{
  "ok": false,
  "data": null,
  "error": {
    "error_code": "STABLE_ERROR_CODE",
    "message": "面向用户或编排器的恢复说明"
  }
}
```

Router 还会区分三种调度结果：

- `executed`：Tool 已完成读取、计算或草稿生成；
- `needs_clarification`：缺少关键字段，不调用领域 Tool，先向用户追问；
- `invalid`：未知 Tool、Schema 校验失败或执行异常，不允许继续写入。

## 2. 15 个 Tool 契约

| Tool | 作用 | 模型输入 | 主要输出 | 数据写入 | 用户确认 | 失败与恢复 |
|---|---|---|---|---|---|---|
| `get_user_profile` | 读取最小必要个人设置 | 无；用户与时区由服务端注入 | `profile`、`field_scope` | 否 | 否 | 档案或存储读取失败时返回稳定错误；不猜测缺失偏好，可使用默认档案重新展示 |
| `prepare_profile_update` | 生成档案、表达风格或提醒偏好的变更草稿 | `patch`：时区、`coach_style`、饮食偏好、忌口、提醒开关、免打扰时间 | `before/after` 预览、payload、确认令牌、幂等键 | 本次调用不写；确认后写入新版本 | 是，落库前确认 | 非法字段或枚举返回校验错误；向用户展示可选值并重新生成草稿；版本冲突时重新读取档案 |
| `get_health_goals` | 读取目标当前状态和完整版本历史 | 无 | `goals`、`count`，每个目标含版本列表 | 否 | 否 | 存储失败时终止本轮读取；不可用旧缓存假装最新状态 |
| `prepare_goal_change` | 创建、调整、暂停或恢复目标，保留历史版本 | `operation`；创建时需要标题、类型、目标值、单位、周期；其他操作需要 `goal_id`，可附 `reason` | 前后对比、目标版本 payload、确认令牌、幂等键 | 本次调用不写；确认后追加版本 | 是，落库前确认 | 缺字段时追问；目标不存在或版本已变化时重新查询并生成草稿，不覆盖旧版本 |
| `get_health_events` | 查询已确认的饮食、饮水、体重和运动事实 | 可选 `event_type`、`date`、`timezone_name`、`newest_first`、`limit`（1–500） | `events`、匹配数、返回数、实际过滤条件 | 否 | 否 | 日期、时区、类型或 limit 不合法时修正参数；存储失败时返回错误。结果可能很长，应分页或压缩后交给模型 |
| `prepare_health_event` | 生成一条健康记录草稿 | `event_type`、可选时间；饮水需 `amount_ml`；体重需 `weight_kg`；运动需类型和时长；饮食需完整计算结果与来源 | 规范化事件、用户预览、确认令牌、幂等键、令牌有效期 | 本次调用不写；确认后写入事件库 | 是，保存前确认 | 缺字段时定向追问；参数不合法时修正；令牌失效或草稿被改动时必须重新生成，禁止绕过确认 |
| `prepare_event_change` | 为已有记录生成修改对比或删除草稿 | `operation=update/delete`、`event_id`；修改还需 `patch` | 修改时返回当前值与建议值；删除时返回目标记录；均含令牌和幂等键 | 本次调用不写；确认后修改或删除 | 是，执行前确认 | 找不到记录时重新查询；相同内容不更新；修改餐食营养时先重新检索和计算，再生成完整草稿 |
| `retrieve_nutrition_candidates` | 检索 Top-K 标准食物候选 | `query`（1–64 字符）、`top_k`（1–10，默认 5） | 候选、匹配分数、来源、Retrieval Trace、可选 Trace 告警 | 不写健康事实；会尽力追加检索 Trace | 否 | 查询或 Top-K 非法时修正；数据源失败时返回错误；Trace 写入失败不推翻主结果，只返回 `trace_warning` |
| `calculate_nutrition` | 基于选中数据行和克重确定性计算营养 | `food_code`、`grams`（大于 0 且不超过 10000）、`retrieval_query` | 食物数据行、份量、营养结果、公式“每 100g × 克重 ÷ 100” | 否 | 否 | 食物不存在时重新检索；克重或数据异常时停止计算，不让模型补造营养值 |
| `retrieve_health_knowledge` | 检索带来源的一般健康知识并执行安全边界 | `question`（1–500 字符）、`top_k`（1–5，默认 3） | `answer_scope`、带 URL 和更新时间的 citations、数量 | 否 | 否 | 未命中则明确证据不足；医疗、用药、紧急风险或提示注入触发拒答/就医引导，不自动重试绕过边界 |
| `get_daily_summary` | 汇总指定日期的已确认事实和目标差距 | `date`、可选 `timezone_name` | 原始 events、分类汇总、目标差距、数据完整度 | 否 | 否 | 修正日期或时区后重试；存储失败则停止。当前返回原始 events，进入模型前应裁剪 |
| `get_period_summary` | 汇总 7/14/30 天事实趋势，不推断原因 | `days=7/14/30`、可选 `end_date`、`timezone_name` | 日期范围、记录数、有数据天数、完整度、饮食/饮水/运动/体重事实、目标进度、解释边界 | 否 | 否 | 不支持的周期直接校验失败；查询或存储失败时返回错误，不用缺失数据补推原因 |
| `create_reminder_draft` | 生成本地提醒草稿 | `content`（1–300 字符）、`scheduled_for`（ISO 8601）、可选时区 | 提醒 payload、预览、确认令牌、幂等键 | 本次调用不写；确认后创建提醒 | 是，安排前确认 | 提醒关闭时先征得同意修改设置；过去时间或格式错误时询问新的未来时间 |
| `execute_reminder` | 用有效令牌真正执行提醒草稿 | `draft`、`confirmation_token`、`idempotency_key` | 已创建/变更的提醒、`idempotent_replay` | 是 | 是，必须已有有效确认令牌 | 令牌无效、payload 被篡改或草稿格式错误时拒绝执行；重复请求依赖幂等结果，不能生成第二条提醒 |
| `list_or_cancel_reminders` | 查看提醒，或生成取消、延后、暂停、恢复草稿 | `action=list/cancel/snooze/pause/resume`；写操作需 `reminder_id`，延后还需新时间 | list 返回提醒列表；其他操作返回前后预览、令牌和幂等键 | `list` 不写；其他调用不立即写，确认后变更 | list 否；其他操作是 | 缺 ID/时间时追问；已结束提醒拒绝再次修改；存储失败时不显示成功 |

### 关于 `execute_reminder`

它目前属于 15 个公开 Schema，但从安全架构看应优先作为“确认中间件的服务端执行器”：
模型负责提出提醒草稿，用户在 UI 确认，服务端再注入确认令牌和幂等键执行。不要让模型自行复制、改写或保存令牌。

## 3. 当前是否每次把 15 个 Tool 都给模型

是。当前 legacy runner 和 LangGraph runner 每一轮都会把
`router.tool_definitions` 全量传给模型，OpenAI-compatible adapter 再把它们放入
请求的 `tools` 字段。

当前 15 个 Schema 紧凑序列化后约为：

- 10,157 个字符；
- 11,089 个 UTF-8 字节；
- 实际 token 数取决于 Provider 的 tokenizer，通常会占用数千 token。

15 个 Tool 尚在可接受范围，但全量发送会在每个模型轮次重复产生输入成本，也会增加选错相似 Tool 的概率。MCP 的 `tools/list` 只负责发现能力，并不要求宿主把全部 Tool 永久放进每次模型请求；是否全量注入由 MCP Client 或 Agent 编排层决定。

## 4. 推荐的 Tool 选择策略

保留服务端 15 个 Tool 的完整白名单，但每轮只给模型一个与当前意图相关的子集：

| 能力包 | 建议注入的 Tool |
|---|---|
| 今日与查询 | `get_health_events`、`get_daily_summary`、`get_period_summary` |
| 新增/修改记录 | `get_health_events`、`prepare_health_event`、`prepare_event_change` |
| 餐食营养 | `retrieve_nutrition_candidates`、`calculate_nutrition`、`prepare_health_event` |
| 档案与目标 | `get_user_profile`、`prepare_profile_update`、`get_health_goals`、`prepare_goal_change`、`get_period_summary` |
| 健康知识 | `retrieve_health_knowledge` |
| 提醒 | `create_reminder_draft`、`list_or_cancel_reminders`；`execute_reminder` 由确认中间件调用 |

路由原则：

1. 先用规则或轻量意图分类器选能力包，不需要先让主模型看完所有 Schema；
2. 每轮通常注入 3–6 个 Tool；多意图请求可合并两个能力包；
3. 存在 `pending_task` 或 `pending_confirmation` 时，必须固定保留对应 Tool，不能重新路由丢失状态；
4. 服务端 Router 仍执行完整白名单、权限、Schema、超时和确认校验，动态选择不能代替安全边界；
5. `execute_reminder` 等真正写入执行器不进入常规模型 Tool 集，由受信任代码调用。

15 个 Tool 暂时不需要再增加一个 `search_tools` 元工具。多一次工具发现往返的收益有限，意图能力包更简单、更可测试。

## 5. 防止搜索结果撑大上下文

当前 `_tool_result_message` 会把完整 Tool Result 序列化进消息历史。对于事件查询、每日汇总和检索结果，这是比 Tool Schema 更明显的上下文增长点。

推荐把 Tool Result 分成三层：

1. **模型视图**：摘要、最多 3–5 条必要记录、总数、截断标记和下一页游标；
2. **UI 视图**：页面按需读取完整列表或表格，不经过模型上下文；
3. **审计视图**：完整结果写入受控 Trace/Store，返回用户隔离且带 TTL 的 `result_ref`。

建议的模型返回形状：

```json
{
  "ok": true,
  "data": {
    "summary": "找到 42 条饮水记录，最近 7 天共 6 条",
    "items": ["最多返回 3—5 条模型真正需要的记录"],
    "total_count": 42,
    "returned_count": 5,
    "truncated": true,
    "next_cursor": "opaque-cursor",
    "result_ref": "user-scoped-result-id"
  },
  "error": null
}
```

具体约束：

- 把 `get_health_events` 的模型默认 limit 从 100 降至 10 或 20，并增加 cursor；
- `get_daily_summary` 给模型只返回 summary、goal gaps 和 completeness，不重复返回全部 events；
- 检索工具维持小 Top-K，限制单条 snippet 字符数，去重来源，只保留回答需要的字段；
- 旧 Tool Result 不永久留在对话历史：保留最近一次完整小结果，更早结果压成事实摘要和 `result_ref`；
- 完整健康数据不能放进不受控缓存；`result_ref` 必须绑定 user/session、设置 TTL，并在读取时重新鉴权；
- 截断发生在 Tool/编排层，不依赖提示词要求模型“少看一点”。

## 6. Tool 继续增多时怎么优化

### 15–20 个：动态子集

- 使用上面的能力包，每轮注入 3–6 个；
- 缩短描述，Schema 保持严格枚举、范围和 `additionalProperties=false`；
- 统计 Tool 选择准确率、结果 token P95、超时率和确认转化率。

### 20–100 个：注册表与分层发现

- 建立 Tool Registry：名称、命名空间、用途摘要、风险、权限、延迟、成本、结果大小；
- 先按权限和风险过滤，再按规则/语义检索选 Top-N Schema；
- 使用 `health.events.*`、`health.goals.*`、`reminders.*` 等稳定命名空间；
- 合并高度重叠的 Tool，但不要把无关读写塞进一个万能 Tool；
- 只允许无依赖的只读 Tool 并行，写操作仍串行并经过确认。

### 100 个以上：控制面与执行面分离

- Router/Planner 只看轻量能力卡片，不看全部 JSON Schema；
- 选中能力后再按需加载完整 Schema；
- 独立 Policy Gateway 负责身份、权限、风险、确认、限流、超时、幂等和审计；
- 结果保存在外部状态中，模型通过受控引用继续任务；
- 对路由建立离线评测集，验证召回正确 Tool、没有漏掉必要 Tool、没有暴露越权 Tool。

## 7. 适合本项目的实施顺序

1. 先增加 `_compact_tool_result`，阻止完整事件数组进入持久会话；
2. 把 `get_daily_summary` 的模型视图与 UI 完整视图分开；
3. 为 `get_health_events` 增加 cursor，并把模型默认 limit 调低；
4. 增加 `select_tool_definitions(intent, pending_state)`，每轮只传相关能力包；
5. 将 `execute_reminder` 从常规模型可见集合移到确认中间件内部；
6. 最后再考虑 MCP Tool Registry。当前 15 个 Tool 不需要引入复杂的语义工具搜索服务。
