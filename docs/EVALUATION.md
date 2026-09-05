# 食物检索评测

## 公开示例档

新环境的完整 RAG 验收使用：

```bash
./scripts/reproduce_rag.sh
```

该入口会下载固定 revision 的 embedding 模型、重建索引，并同时运行 Lexical 与
Hybrid 评测。以下 `run_eval.py` 命令仅用于单独调试一种模式。

`tests/eval/nutrition_retrieval.jsonl` 固定包含 20 条查询，覆盖标准名、别名、
口语名、复合菜、相似食物、错别字和明确拒答。所有非空
`expected_food_codes` 都是 28 条公开示例中真实存在的 `FOOD_0xx`。

只运行默认 Lexical 评测：

```bash
python scripts/run_eval.py
```

脚本与 pytest 共用 `src/nutrition/evaluation.py`，报告写入
`docs/eval_report.json`。门槛为：

- `recall_at_3 >= 0.85`；
- `rejection_accuracy == 1.0`。

任一门槛不满足时退出码为 1；评测集或数据文件无效时退出码为 2。

当前 28 条示例数据的实测报告为：Recall@3 `0.9474`、Top1 Accuracy
`0.9474`、Rejection Accuracy `1.0`、Overall Pass Rate `0.95`。当前唯一失败
用例是错别字“蕃茄”；系统选择拒答而不是猜测营养数据，因此仍通过 P0 的
Recall@3 `>= 0.85` 和拒答准确率门槛。

## 本地完整档

`tests/eval/nutrition_retrieval_full.template.jsonl` 使用同样 20 条查询，但
`expected_food_codes` 一律为空数组。完整数据的真实 foodCode 未包含在仓库中，
不得猜测或虚构；需本地跑完 prepare 后人工填写，再使用显式参数评测：

```bash
python scripts/run_eval.py \
  --cases <已人工填写的完整档评测集.jsonl> \
  --data data/full/foods_normalized.json \
  --report <本地报告路径.json>
```

模板本身不是可直接计分的完整档基准。尤其是非 `not_found` 用例，必须经人工
核验并填写真实 code 后才有评测意义。

## 指标口径

- Recall@3：只统计有期望答案的用例，Top 3 中出现任一期望 code 即命中。
- Top1 Accuracy：只统计有期望答案的用例，首位是任一期望 code 即命中。
- Rejection Accuracy：`not_found` 用例是否返回空候选和 `not_found` 状态。
- Overall Pass Rate：有答案用例按 Recall@3 命中，拒答用例按正确拒答命中。
- 分组明细：按 `case_type` 给出总数、通过数和通过率。
