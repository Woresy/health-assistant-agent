# 食物检索 Hybrid RAG

## 检索流程

```text
用户查询
  -> 文本归一化
  -> 四阶段词法召回
  -> Dense 向量召回
  -> Reciprocal Rank Fusion
  -> 类别约束与确定性规则重排
  -> 拒答门控
  -> Top-K 食物候选
  -> 用户确认
  -> 读取结构化营养事实
```

Embedding 只用于召回候选，不生成热量或营养值。营养事实始终来自本地结构化
食物数据，并在用户确认候选和份量后进行确定性计算。

## 一条命令复现

环境要求：Linux 或 macOS、Python 3.11+、首次执行时可访问 Python 包索引和
Hugging Face。

在仓库根目录运行：

```bash
./scripts/reproduce_rag.sh
```

该命令会按顺序完成：

1. 创建或复用 `.venv`，安装 `requirements.txt` 中的固定依赖；
2. 下载并校验 `BAAI/bge-small-zh-v1.5` 的固定 revision；
3. 使用公开示例数据和人工语义提示重建 `data/index`；
4. 运行 Lexical 固定评测；
5. 运行 Hybrid 固定评测，并验证 Dense 没有降级；
6. 写出两个评测报告和一份复现汇总。

成功时最后一行会显示：

```text
复现通过。汇总报告：.../artifacts/rag-reproduction/reproduction_summary.json
```

任何步骤失败都会返回非零退出码，因此同一命令也可以用于 CI 或验收脚本。
一键流程默认使用 CPU，避免 CUDA 驱动和不同 GPU 带来的环境差异；公开示例索引只有
28 条文档，不依赖 GPU。

## 固定版本与本地文件

默认模型配置定义在 `scripts/build_food_index.py`：

```text
model: BAAI/bge-small-zh-v1.5
revision: 7999e1d3359715c523056ef9478215996d62a620
```

模型权重由 Hugging Face 缓存在本机，不提交 GitHub。若通过 `HF_HOME` 将缓存放到
仓库内，建议使用 `.cache/huggingface`，该目录已加入 `.gitignore`。

重建后的索引包含：

```text
data/index/food_documents.json
data/index/food_embeddings.npy
data/index/index_manifest.json
```

`index_manifest.json` 记录模型名称、revision、数据集 SHA-256、索引文件
SHA-256、向量维度、文档数量和查询指令。运行时会检查数据与索引是否匹配。

复现报告默认写入：

```text
artifacts/rag-reproduction/eval_lexical.json
artifacts/rag-reproduction/eval_hybrid.json
artifacts/rag-reproduction/reproduction_summary.json
```

报告目录属于本地构建产物，不提交 GitHub。

## 离线复跑

模型和依赖已经存在时，可禁止模型联网并跳过依赖安装：

```bash
RAG_SKIP_INSTALL=1 ./scripts/reproduce_rag.sh --offline
```

如果离线缓存中没有固定 revision，命令会返回
`EMBEDDING_MODEL_DOWNLOAD_FAILED`，不会静默退回 Lexical，也不会用随机向量代替。

## 自定义输入与输出

包装脚本会把其他参数原样传给 `reproduce_rag.py`：

```bash
./scripts/reproduce_rag.sh \
  --data data/samples/foods_sample.json \
  --hints data/retrieval_hints.json \
  --cases tests/eval/nutrition_retrieval.jsonl \
  --index-dir /tmp/healthos-rag-index \
  --report-dir /tmp/healthos-rag-report
```

可以通过以下环境变量调整新环境初始化：

| 变量 | 作用 |
| --- | --- |
| `RAG_PYTHON_BIN` | 创建虚拟环境所用的 Python，默认 `python3` |
| `RAG_VENV_DIR` | 虚拟环境目录，默认仓库内 `.venv` |
| `RAG_SKIP_INSTALL=1` | 跳过 `pip install`，适合依赖已准备好的环境 |
| `HF_HOME` | 覆盖 Hugging Face 模型缓存目录 |
| `HF_TOKEN` | 私有或受限模型访问令牌，当前公开模型通常不需要 |

## 评测门槛

固定评测集为 `tests/eval/nutrition_retrieval.jsonl`，包含标准名、别名、口语名、
复合菜、相似食物、错别字、不存在食物和领域外查询。

| 指标 | 门槛 |
| --- | ---: |
| Recall@3 | `>= 0.85` |
| Rejection Accuracy | `= 1.0` |
| Hybrid Dense 降级次数 | `= 0` |
| Lexical 和 Hybrid `passed` | `true` |

Hybrid 评测即使 Recall@3 达标，只要 Dense 发生一次降级，也会判定失败。

## 常见失败与恢复

| 错误码 | 含义 | 恢复方式 |
| --- | --- | --- |
| `EMBEDDING_MODEL_DOWNLOAD_FAILED` | 模型未缓存或网络下载失败 | 检查网络、代理、`HF_TOKEN` 或缓存目录后重跑同一命令 |
| `RAG_INDEX_BUILD_FAILED` | 数据、提示、模型加载或写索引失败 | 查看原始错误，确认磁盘空间和输入文件后重跑 |
| `DENSE_INDEX_DATASET_MISMATCH` | 索引与当前食物数据不一致 | 重新执行一条命令复现流程 |
| `RAG_EVALUATION_FAILED` | 评测集或运行时检索失败 | 查看错误码和对应报告，修复后重跑 |
| `RAG_EVALUATION_THRESHOLD_FAILED` | 指标未达到验收门槛 | 查看报告中的 `failures`，不要降低门槛掩盖回归 |

## 分步调试

只有排查问题时才需要绕过一键入口：

```bash
.venv/bin/python scripts/build_food_index.py \
  --data data/samples/foods_sample.json \
  --hints data/retrieval_hints.json \
  --index-dir data/index

.venv/bin/python scripts/run_eval.py \
  --mode lexical \
  --report /tmp/eval_lexical.json

.venv/bin/python scripts/run_eval.py \
  --mode hybrid \
  --index-dir data/index \
  --report /tmp/eval_hybrid.json
```

正常验收仍以 `./scripts/reproduce_rag.sh` 为准，避免漏掉模型下载、索引重建或某一种
评测。
