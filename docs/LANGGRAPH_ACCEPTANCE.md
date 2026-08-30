# LangGraph 接入验收手册

这份手册用于亲自验证：LangGraph 确实参与了状态编排，缺参和写操作确实暂停，
用户确认前没有副作用，确认后事件、汇总和 Trace 保持一致。

## 1. 验收前准备

在仓库根目录执行：

```bash
cd /home/woresy/ai-project-practice/health-assistant-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少设置：

```dotenv
APP_HOST=127.0.0.1
APP_PORT=7860
APP_TIMEZONE=Asia/Shanghai

AGENT_PROVIDER_MODE=openai_compatible
AGENT_API_KEY=<本地填写，不要提交>
AGENT_BASE_URL=<Provider Base URL>
AGENT_MODEL=<支持 Tool Calling 的模型>
AGENT_ORCHESTRATOR=langgraph

HEALTH_CONFIRMATION_SECRET=<至少32位随机字符串>
```

不要把真实 API Key 粘贴到终端日志、截图、Git 或验收报告中。

## 2. 自动化验收

### 2.1 全量测试

```bash
.venv/bin/pytest -q
```

预期：全部通过。LangGraph 专项用例位于：

```text
tests/e2e/test_langgraph_health_flows.py
```

### 2.2 只跑 LangGraph 主链

```bash
.venv/bin/pytest -q tests/e2e/test_langgraph_health_flows.py
```

这些用例分别证明：

1. 饮水草稿停在确认 interrupt，确认前 JSONL 为空；
2. 运动缺参停在补参 interrupt，补充后继续同一个图；
3. 只读工具结果回到模型，模型产生最终回答；
4. 未知工具被白名单拒绝且没有副作用。

### 2.3 检索回归

```bash
.venv/bin/python scripts/run_eval.py \
  --mode lexical \
  --report /tmp/health-agent-eval-lexical.json

.venv/bin/python scripts/run_eval.py \
  --mode hybrid \
  --report /tmp/health-agent-eval-hybrid.json
```

预期：两种模式退出码均为 0，Recall@3 不低于 `0.85`，Rejection Accuracy
等于 `1.0`。当前示例数据的实测 Recall@3 为 `0.9474`。

## 3. 启动页面

```bash
.venv/bin/python app.py
```

打开：

```text
http://127.0.0.1:7860
```

在页面 Provider 状态中确认出现：

```text
编排器：langgraph
```

验收过程中同时打开以下区域：

- 对话；
- 今天；
- 健康时间线；
- 开发者证据。

## 4. 场景一：写操作必须暂停并确认

在对话中输入：

```text
记录喝水500毫升
```

确认以下现象：

1. 状态为 `awaiting_confirmation`；
2. `finish_reason` 为 `awaiting_confirmation`；
3. 页面展示待保存草稿；
4. `pending_confirmation.action` 为 `save`；
5. 今天的饮水量尚未增加；
6. 健康时间线中尚无这条事件。

点击“确认当前操作”。预期：

1. 状态变为 `completed`；
2. tool step 为 `save_health_event`；
3. 今天的饮水量增加 500 ml；
4. 时间线出现一条 water 事件；
5. 页面明确显示保存成功。

这一场景对应图路径：

```text
call_model
→ dispatch_tool
→ await_confirmation（暂停）
→ Command(resume=confirm)
→ execute_confirmation
→ END
```

## 5. 场景二：缺参后继续同一任务

输入：

```text
我刚跑步了
```

预期：

- 状态为 `awaiting_clarification`；
- `pending_task.tool_name` 为 `prepare_health_event`；
- `missing_parameters` 包含 `duration_minutes`；
- 系统只询问运动时长；
- 时间线没有新增运动事件。

继续输入：

```text
30分钟
```

预期：原来的“跑步”与新的“30分钟”被合并，状态进入
`awaiting_confirmation`，草稿中同时包含运动类型和时长。

点击“取消当前操作”。预期：

- 状态为 `cancelled`；
- pending task 和 pending confirmation 都被清除；
- 时间线仍然没有这条运动事件；
- 今日运动总时长不变。

这一场景对应图路径：

```text
dispatch_tool
→ await_clarification（暂停）
→ Command(resume=clarify)
→ call_model
→ dispatch_tool
→ await_confirmation（暂停）
→ Command(resume=cancel)
→ END
```

## 6. 场景三：待确认时阻止新请求

输入：

```text
记录体重65公斤
```

不要确认，再输入：

```text
再记录喝水300毫升
```

预期：

- 模型轮次为 0；
- 系统提示先确认或取消当前草稿；
- 不会创建第二个草稿；
- 确认原草稿后只新增一条 65 kg 的体重事件。

这证明 UI 不能用新消息绕过当前 LangGraph interrupt。

## 7. 场景四：查询不产生副作用

输入：

```text
查询我今天的健康记录
```

预期：

- tool step 为 `query_health_events`；
- 工具结果返回模型后才生成自然语言回答；
- 事件数量、事件 ID 和每日汇总均不发生变化；
- `model_rounds` 通常为 2：一次选择工具，一次读取 Tool Result 后回答。

再输入：

```text
汇总我今天的健康记录
```

预期调用 `get_daily_health_summary`，数值与“今天”页面一致。

## 8. 场景五：修改与删除仍受确认保护

从健康时间线复制一条 `event_id`，输入：

```text
把事件 <event_id> 的体重改为64.8公斤
```

预期先展示修改前后对比，确认前原事件不变；确认后事件 ID 保持不变、版本内容
更新、每日汇总重新计算。

然后输入：

```text
删除事件 <event_id>
```

第一次点击取消，确认事件仍存在。再次发起删除并确认，确认事件从时间线消失，
每日汇总同步更新。

## 9. 场景六：Trace 脱敏

在“开发者证据”刷新 Agent Trace，预期能看到：

- `action=send/confirm/cancel`；
- `state`、`finish_reason`、`model_rounds`；
- 工具名、参数名称、成功状态和错误码；
- `has_pending_task` 或 `has_pending_confirmation`。

不应该看到：

- 原始用户句子；
- 500 ml、65 kg 等健康参数值；
- API Key；
- `confirmation_token`。

也可以在本地检查：

```bash
tail -n 5 data/agent_traces.jsonl
```

## 10. 场景七：Legacy 回退

停止应用，将 `.env` 改为：

```dotenv
AGENT_ORCHESTRATOR=legacy
```

重新启动应用。预期 Provider 状态显示 `编排器：legacy`，此前已经确认保存的
HealthEvent 仍然存在，查询、汇总和写操作确认仍可使用。

再切回：

```dotenv
AGENT_ORCHESTRATOR=langgraph
```

已保存事件仍然不变。这证明编排器切换不迁移、不覆盖业务数据。

## 11. 最终通过清单

- [ ] 页面明确显示 `编排器：langgraph`；
- [ ] 确认前 HealthEvent 不增加；
- [ ] 确认后只增加一条事件；
- [ ] 缺参状态可以跨下一条消息恢复；
- [ ] 取消不会留下半成品事件；
- [ ] 待确认时新请求被阻止；
- [ ] 查询和汇总不修改数据；
- [ ] 修改、删除都先展示草稿；
- [ ] 时间线与每日汇总始终一致；
- [ ] Trace 包含运行证据但不包含健康参数值和确认令牌；
- [ ] legacy 回退后已保存事件不丢失；
- [ ] 全量测试和两种检索评测通过。
