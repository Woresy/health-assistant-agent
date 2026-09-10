# 性能与超时契约

本项目把“快”拆成可复现的本地指标和不可伪造的外部 Provider 指标。CI 使用
Lexical RAG、关闭 LangSmith、禁用外部模型，因此不会把网络波动误当成本地回归。

| 场景 | 目标 | 验证方式 |
| --- | ---: | --- |
| Lexical 冷启动到 HTTP 可用 | ≤ 15 秒 | 启动独立 Gradio 进程并轮询首页 |
| 已启动应用的页面响应 P95 | ≤ 3 秒 | 连续请求本地首页 10 次 |
| 确定性提醒草稿 P95 | ≤ 2 秒 | 连续生成 20 个草稿，断言模型调用为 0 |
| 普通模型响应 | 目标 ≤ 15 秒 | 通过本地 Trace / LangSmith 观察真实 Provider |
| 单次模型请求 | 硬上限 60 秒 | `AGENT_REQUEST_TIMEOUT=60` 且自动重试为 0 |

运行完整基准并在超阈值时返回非零状态：

```bash
.venv/bin/python scripts/benchmark_performance.py \
  --assert-thresholds \
  --report /tmp/healthos-performance.json
```

这里的“热页面”指应用已经运行后的首页 HTTP 响应，不等同于公网部署后的 Core Web
Vitals。部署后还需在真实域名、移动设备和弱网环境测量 LCP、INP、CLS。外部模型的
15 秒是体验目标，不把测试桩结果冒充真实 Provider SLA；真实耗时应在开发者证据页的
本地 Trace 或 LangSmith 项目中查看。

自动重试会把一次 60 秒等待放大成数分钟，所以当前配置明确要求
`AGENT_MAX_RETRIES=0`。超时后由用户决定是否重试，避免重复工具意图和长时间假死。
