# LangSmith 观测与优化

本项目同时保留两套观测：

- 本地脱敏 Trace：用于隐私审计、确认链路和离线回放；
- LangSmith：用于查看 LangGraph 节点耗时、模型调用、异常、Tool 链路与多轮会话。

LangSmith 不是业务数据库，也不参与健康事实写入、用户确认或飞书发送。
远程观测不可用时，核心功能继续运行。

## 1. 创建 API Key

登录 <https://smith.langchain.com>，进入 `Settings > API Keys` 创建 API Key。
如果密钥属于多个 Workspace，同时复制 Workspace ID。

## 2. 配置 `.env`

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<你的 LangSmith API Key>
LANGSMITH_PROJECT=health-assistant-agent-local
LANGSMITH_ENDPOINT=https://api.smith.langchain.com/v1
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
LANGSMITH_WORKSPACE_ID=
```

不要提交 `.env`。健康项目默认隐藏输入和输出，因此 LangSmith 能看到流程结构、
节点耗时、错误和模型元数据，但看不到用户的饮食、体重、提醒文本或模型回答。
只有在取得用户明确授权并完成数据治理后，才考虑关闭隐藏开关。

## 3. 启动并验证

```bash
python app.py
```

页面的 Provider 状态应包含：

```text
LangSmith 已启用：health-assistant-agent-local；输入输出已隐藏
```

完成一次普通对话后，进入 LangSmith 的 `Projects`，打开
`health-assistant-agent-local`。在 Trace 中应看到 LangGraph 节点；模型调用显示为
`HealthOS Agent Model`，对话、确认和取消分别显示为脱敏的顶层运行。`Threads` 页面使用不可逆摘要后的 thread ID 聚合多轮对话，
不会上传浏览器本地会话 ID。

## 4. 优化建议

优先建立以下筛选或 Dashboard：

1. P50/P95 总耗时及 `HealthOS Agent Model` 耗时；
2. `timeout`、Provider 错误和 Tool 参数校验失败率；
3. 每轮模型调用次数与 Tool 调用次数；
4. `awaiting_confirmation` 到确认/取消的转化；
5. 提醒、健康记录、查询、RAG 四类任务的成功率。

保留一组固定测试问题作为 Dataset，每次修改 Prompt、模型、Tool 路由或 RAG
索引后运行同一组评测，比较正确 Tool、是否需要追问、是否越过确认边界、耗时和成本。
