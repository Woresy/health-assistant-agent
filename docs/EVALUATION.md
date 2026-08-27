# 食物检索评测

## 公开示例档

`tests/eval/nutrition_retrieval.jsonl` 固定包含 20 条查询，覆盖标准名、别名、
口语名、复合菜、相似食物、错别字和明确拒答。所有非空
`expected_food_codes` 都是 28 条公开示例中真实存在的 `FOOD_0xx`。

从仓库根运行：

```bash
python scripts/run_eval.py
```

脚本与 pytest 共用 `src/nutrition/evaluation.py`，报告写入
`docs/eval_report.json`。门槛为：

- `recall_at_3 >= 0.85`；
- `rejection_accuracy == 1.0`。

任一门槛不满足时退出码为 1；评测集或数据文件无效时退出码为 2。

当前 28 条示例数据的报告为：Recall@3 `1.0`、Top1 Accuracy `1.0`、
Rejection Accuracy `1.0`、Overall Pass Rate `1.0`。

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
