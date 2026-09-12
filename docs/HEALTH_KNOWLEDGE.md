# 健康知识 Hybrid RAG：来源、索引与评测

健康知识与食物营养数据分开管理。食物 RAG 负责识别结构化食物实体；本模块负责从
可信健康文档中检索带引用的回答上下文。两者都使用 Embedding，但不会让向量结果
生成或覆盖健康事实。

## 运行链路

```text
用户问题
  → 急症 / 医疗边界 / Prompt Injection 安全门控
  → topics 词法召回（pool=8）
  → BGE Embedding + 本地 NumPy cosine 召回（pool=8）
  → 加权 RRF（Dense 0.85 / Lexical 0.15，rrf_k=10）
  → 领域信号 + minScore 0.44 + minMargin 0.025
  → Top-K（默认 3）可信文档
  → 文档正文、适用范围和引用进入 Agent 上下文
  → 模型生成带来源的一般健康回答
```

当前只有 6 条短文档，使用 NumPy 做精确 cosine 检索比部署独立向量数据库更简单，
结果也完全可复现。数据量增长到需要分片、过滤、多用户隔离或增量索引时，再将
`local_numpy_exact_cosine` 替换为专用向量数据库；`top_k`、`min_score` 与引用协议
不需要随存储实现改变。

## 构建索引

```bash
.venv/bin/python scripts/build_health_knowledge_index.py --offline
```

索引保存在 `data/knowledge_index/`：

- `knowledge_embeddings.npy`：归一化后的 512 维文档向量；
- `knowledge_index_manifest.json`：模型、固定 revision、数据 SHA-256、向量 SHA-256、
  文档顺序、维度和查询指令。

运行时只有在数据哈希、文档 ID、向量哈希和形状全部一致时才启用 Dense；索引缺失或
损坏会安全降级为词法检索，不影响急症与医疗边界判断。

## 准备查询编码器

文档向量随仓库提交，查询编码器不提交（约 93MB，落在 `~/.cache/huggingface`）。只有
向量而没有编码器时无法编码用户问题，Dense 召回会降级为词法检索：语义改写问题会
直接返回 `KNOWLEDGE_NOT_FOUND`。降级是可观测的——`retrieval_receipt()` 的 `mode`
同时依赖索引和编码器，缺编码器时报告 `lexical_fallback`，并把 `vector_store` 和
`embedding_model` 置空；`index_ready` 用于区分索引损坏和编码器缺失。

```bash
.venv/bin/python scripts/prepare_health_knowledge_model.py           # 下载并校验
.venv/bin/python scripts/prepare_health_knowledge_model.py --offline # 只校验本地缓存
```

脚本按索引 manifest 固定的 `model_name` 与 `model_revision` 准备编码器，并校验查询
向量维度与已提交向量一致；不一致说明索引与模型不同源，需要重建 `data/knowledge_index/`。
CI 用 `--emit-pin` 输出的固定值做缓存键，在跑测试前执行同一个脚本，因此编码器缺失
会让 CI 直接失败，而不是降级后在语义用例上报错。

部署镜像在构建期执行同一个脚本，把编码器预置到 `HF_HOME=/opt/huggingface`，随后
用 `HF_HUB_OFFLINE=1` 锁成离线，容器运行时不访问 Hugging Face。这一步是必需的：
镜像只带向量不带编码器时，线上健康知识检索会退回纯词法，「晚餐怎么搭配」这类
问题会答不出来。

## 来源与信任边界

运行时读取 `data/samples/health_knowledge.json`。每条文档必须包含适用范围、来源机构、
来源 URL 和更新时间。文档始终是不受信任的数据，不能改变 Tool 权限、确认协议或
系统规则；包含越权指令的文档在进入词法和向量候选前就会被过滤。

当前公开样例使用 WHO 与 CDC 的身体活动、饮食、减钠、饮水和睡眠资料。它不提供
疾病诊断、治疗或个体化处方。

## 评测与结论

```bash
.venv/bin/python scripts/evaluate_health_knowledge.py
.venv/bin/python -m pytest -q tests/eval/test_health_knowledge_eval.py
```

`docs/health_knowledge_eval_report.json` 是权威对比报告。固定集合包括 24 条可回答问题
（其中 6 条为不包含原 topics 的语义改写）、2 条领域外拒答和 20 条安全请求：

| 指标 | Lexical | Hybrid |
| --- | ---: | ---: |
| Recall@3 | 0.7917 | 1.0000 |
| Top1 Accuracy | 0.7917 | 1.0000 |
| 语义改写 Recall@3 | 0.1667 | 1.0000 |
| 领域外拒答准确率 | 1.0000 | 1.0000 |
| 安全召回率 | — | 1.0000 |

Hybrid 在语义改写集上提升 `0.8333`，且没有降低拒答和安全指标，因此当前引入 Dense
是有评测收益的，而不只是架构形式上的变化。
