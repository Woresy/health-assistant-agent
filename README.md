# 食物健康助手

一个通过图片入口、手动食物检索和确定性营养计算完成本地饮食记录的 Gradio 学习项目。

## 运行环境

- Python 3.11+
- Gradio
- Pydantic
- Pillow
- pytest

所有直接依赖均已固定版本。应用不接入大模型，不需要任何 API Key。

## 从空虚拟环境启动

进入仓库根目录后，复制执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py

```

浏览器访问：

```text
http://127.0.0.1:7860
```

`app.py` 是项目唯一启动入口。应用不接入大模型，没有 API Key 也可以正常启动。

## 使用流程

1. 打开“记录饮食”Tab。
2. 上传一张 JPG、JPEG 或 PNG 图片。
3. 手动填写食物名称，例如“西红柿”。
4. 填写食物可食部分克重。
5. 点击“查找候选”。
6. 从固定食物数据中选择候选。
7. 点击“计算营养估算”。
8. 核对食物、份量、估算值、数据来源和检索词。
9. 点击“确认保存”。
10. 在“今日记录”Tab 查看重新从 JSONL 读取的记录和汇总。

图片仅作为输入入口，不进行图片识别，也不会复制到项目数据目录。点击“取消”不会保存半成品。

## 运行测试

在已激活的虚拟环境中执行：

```bash
python -m pytest -q tests/e2e
```

期望输出：

```text
1 passed
```

E2E 测试使用 `tests/fixtures/meal.png` 固定图片，不启动浏览器、不依赖 API Key，也不会写入正式健康数据文件。

检索单元测试、固定评测测试和命令行评测分别执行：

```bash
python -m pytest -q tests/unit tests/eval
python scripts/run_eval.py
```

当前 28 条示例数据、20 条固定查询的真实指标为：Recall@3 `1.0`、
Top1 Accuracy `1.0`、Rejection Accuracy `1.0`、Overall Pass Rate `1.0`。
详细口径和失败清单见 `docs/EVALUATION.md` 与 `docs/eval_report.json`。

## 数据存储

确认后的健康事件保存在：

```text
data/health_events.jsonl
```

该文件使用 JSON Lines 格式，每行保存一个完整 `HealthEvent`。

运行时健康数据、`.env`、上传原图和完整食物数据集不得提交到 GitHub，相关路径已加入 `.gitignore`。

## 食物数据来源

`data/samples/foods_sample.json` 包含 28 条手写示例占位数据。

数据来源字段标记为：

> 《中国食物成分表》第 6 版整理仓库（示例占位，仅供学习）

该文件不是《中国食物成分表》的完整数据集，不应作为正式或生产环境的营养数据库使用。

完整上游数据不得提交。取得本地数据后可执行：

```bash
python scripts/prepare_food_data.py \
  --src <上游json目录> \
  --out data/full/foods_normalized.json \
  --aliases data/aliases.json \
  --report data/full/prepare_report.json
```

应用优先读取 `FOOD_DATA_PATH`，否则读取已存在的本地完整档，最后回落到公开
示例档。数据来源、质量标记和版权边界详见 `docs/DATA_SOURCES.md`。

## 重要声明

页面展示的热量、蛋白质、脂肪和碳水均为估算值，计算公式为：

```text
每 100g 营养值 × 用户填写克重 ÷ 100
```

所有结果仅供学习，不构成医疗建议、诊断或治疗意见。

## 已实现

- Python 3.11 和固定版本依赖。
- `app.py` 唯一启动入口。
- 无 API Key 启动。
- 单张 JPG、JPEG、PNG 图片输入。
- 图片空文件、损坏、超限及格式错误校验。
- 手动填写食物名称和克重。
- 标准名、别名、双向包含和字符模糊四阶段检索。
- 可解释候选分数、命中词、数据集信息和选择模式。
- 独立检索 Trace 与可重算营养证据。
- Top-K 固定食物候选。
- “西红柿”召回“番茄”。
- 确定性营养计算。
- meal 类型 `HealthEvent`。
- 保存确认令牌。
- `idempotency_key` 幂等保存。
- JSONL 单进程存储。
- 今日记录重新读取与汇总。
- 无浏览器 E2E 测试。

## 未实现

- YOLO 或其他图片识别。
- 大模型调用。
- 教练风格和个性化健康建议。
- water、weight、exercise 记录流程。
- 可提交的完整食物数据库（完整档只允许本地准备和使用）。
- 云端数据库。
- 登录、鉴权和多用户系统。
- 医疗诊断或治疗建议。

## 许可证

本项目使用 MIT License。
