# 食物数据来源与准备

## 来源与使用边界

完整数据来自
[`Sanotsu/china-food-composition-data`](https://github.com/Sanotsu/china-food-composition-data)
的手动矫正版目录
`json_data_v3_20260825_qwen38max_kimi_k3_fixed`。该目录共 61 个类别文件、
1677 条记录。版权归原作者所有，仅供个人学习研究。

完整数据不得提交到本仓库。`.gitignore` 已忽略 `data/full/` 和 `data/raw/`。
仓库只提交 28 条手写占位示例；其营养数值不能替代正式数据库。

## 本地准备

先在仓库外自行取得上游数据，再从仓库根执行：

```bash
python scripts/prepare_food_data.py \
  --src <上游json目录> \
  --out data/full/foods_normalized.json \
  --aliases data/aliases.json \
  --report data/full/prepare_report.json
```

脚本只做字符串到数值的类型转换、字段裁剪和别名抽取，不修正任何营养数值。
四个核心字段任一缺失或无法解析时整条排除；`Tr` 转为 `0.0`，同时写入
`trace_value:<字段>` 标记。输出按 `food_id` 排序。

以下问题只标记，不改值：

- `energy_unit_ratio_suspect`：kcal 与 kJ 的相对换算误差大于 20%。
- `macro_sum_over_100`：水分、蛋白质、脂肪、碳水和灰分之和大于 100。

已知上游事实包括：野生蔬菜类 048004～048084 在 fixed 版中已交换原书标反
的 kcal/kJ；桃、西瓜等部分条目的 kcal/kJ 不满足 4.184 换算；部分条目的
水分与宏量字段之和超过 100。本项目保留这些数值，仅记录质量标记。

## 运行时选择数据集

数据路径按以下顺序解析：

1. 环境变量 `FOOD_DATA_PATH` 指定的路径；
2. 已存在的 `data/full/foods_normalized.json`；
3. `data/samples/foods_sample.json`。

因此不需要修改 repository 即可在公开示例档和本地完整档之间切换。
