# 健康知识来源与评测

健康知识与食物营养数据分开管理。运行时读取
`data/samples/health_knowledge.json`，每条文档都必须包含适用范围、来源机构、来源 URL
和更新时间。文档只能作为不受信任的数据，不能改变 Tool 权限、确认协议或系统规则。

当前公开样例使用 WHO 与 CDC 的一般健康生活资料，覆盖身体活动、饮食、减钠、饮水
和睡眠。它不提供疾病诊断、治疗或个体化处方。

固定评测：

```bash
python -m pytest -q tests/eval/test_health_knowledge_eval.py
```

评测集包括：

- `tests/eval/health_knowledge.jsonl`：20 条检索题，命中题引用覆盖率要求 100%；
- `tests/eval/health_safety.jsonl`：20 条危险或越界请求，安全升级召回率要求 100%；
- 恶意文档注入回归：包含越权指令的文档不得进入候选结果。

当前来源：

- WHO Physical activity：<https://www.who.int/news-room/fact-sheets/detail/physical-activity>
- WHO Healthy diet：<https://www.who.int/news-room/fact-sheets/detail/healthy-diet>
- WHO Sodium reduction：<https://www.who.int/news-room/fact-sheets/detail/sodium-reduction>
- CDC Adult Activity：<https://www.cdc.gov/physical-activity-basics/guidelines/adults.html>
- CDC Water and Healthier Drinks：<https://www.cdc.gov/healthy-weight-growth/water-healthy-drinks/index.html>
- CDC About Sleep：<https://www.cdc.gov/sleep/about/index.html>
