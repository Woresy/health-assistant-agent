# GitHub 交付检查

提交前运行：

```bash
.venv/bin/python scripts/check_release.py \
  --report /tmp/healthos-release-check.json
```

检查会覆盖：

- README、唯一入口、架构、工具、评测、RAG、性能和浏览器验收文件是否齐全；
- `.env`、本地数据库、健康事件和 Trace 是否被排除；
- 是否混入 OpenAI-style、LangSmith 或飞书 Webhook 密钥；
- 文档和代码是否带有开发机绝对路径；
- Python 依赖是否固定版本。

CI 会在无真实 API Key、无外部模型调用的条件下依次执行静态检查、全量 Python
测试、Chromium 用户流程、性能门禁、Lexical RAG 复现和交付检查，并上传三份 JSON
证据：RAG、性能与发布前检查。

这只能证明仓库结构和自动化可复现。PRD 要求的“陌生人在 10 分钟内启动”仍应由一名
未参与开发的人，在新的虚拟环境中按 README 实际走一遍并记录结果。
