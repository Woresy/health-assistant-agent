# 技术决策

## 8.27 检索与清洗不新增依赖

食物名称检索仅使用 Python 标准库实现 NFKC、casefold、字符 n-gram Dice 和
Levenshtein 距离。当前固定数据规模为 1677 条，逐条字符串计算足以支撑本地
交互和固定评测，引入 `rank_bm25`、FAISS、ChromaDB、jieba 或 NumPy 不会改善
今天目标中的可解释性与可复现性，反而增加安装、版本和索引维护成本。

四阶段分数和固定排序键直接编码为业务规则。每个候选保留 stage、命中词和
来源，但不携带营养数值；只有用户或规则选中真实数据行后，calculator 才按
“每 100g 数值 × 克重 ÷ 100”计算。

检索 Trace 使用独立 JSONL 文件，不扩展 HealthEvent。Trace 写入属于旁路审计：
失败会返回稳定警告，但不会把已经成功的候选检索改成失败。健康记录保存失败
则仍然禁止展示“记录成功”。

完整上游数据只在本地清洗和使用，不进入 Git。清洗只做转换、裁剪、别名抽取
与质量标记，不修正、补零或猜测异常营养数值。

## 8.30 LangGraph 只替换编排层

- 要解决的问题：原有有限轮次 Loop 已经可以工作，但状态分支、补参、确认和
  后续 P1 流程继续增长时会集中在一个 Runner 中，难以直观看出暂停和恢复位置。
- 候选 A：继续扩展原有 Python Loop。
- 候选 B：使用 LangChain 高层 `create_agent` 和通用 ToolNode。
- 候选 C：使用 LangGraph `StateGraph`，复用现有模型协议和 HealthToolRouter。
- 最终选择：候选 C。
- 原因：可以获得显式节点、条件边、checkpoint 和 interrupt，同时不改变已通过
  测试的工具 Schema、确认令牌、幂等和 JSONL 存储边界。
- 不选 B：通用工具执行节点无法直接表达当前 `needs_clarification`、草稿生成、
  签名确认和确定性领域错误协议，迁移风险高于收益。
- 回退：通过 `AGENT_ORCHESTRATOR=legacy` 保留原 Loop；同一领域测试继续覆盖
  两种编排方式。
- checkpoint：P0 使用 `InMemorySaver`，只保存短期图状态。HealthEvent 和 Trace
  仍使用 JSONL，不把业务事实迁移到 checkpoint。
- 安全约束：生成草稿与 interrupt 分成不同节点；interrupt 恢复后才执行写工具。
  写工具继续校验 confirmation token 并使用幂等键。
- 验证证据：全量 pytest、LangGraph 专项 E2E、固定检索评测和页面开发者证据。

## 9.04 P1 状态继续使用原子 JSON 快照

- 要解决的问题：P1 增加档案、目标版本、提醒状态历史和幂等结果，必须支持跨会话读取、
  安全确认和失败不留下半写状态。
- 候选 A：继续使用 JSON/JSONL；候选 B：迁移 SQLite。
- 对比：SQLite 更适合并发、多用户、索引查询和事务迁移；原子 JSON 快照更容易在当前
  单用户本地作品中检查、演示、删除和回滚，也不会同时迁移已稳定的 HealthEvent JSONL。
- 最终选择：本阶段使用 `healthos_state.json` 原子快照，文件权限 0600，写入经同目录临时
  文件和 `os.replace` 完成。HealthEvent 继续使用独立 JSONL。
- 数据形态：档案为当前快照；目标为不可变版本数组；提醒为当前状态加转换历史；幂等结果
  与业务状态同一次原子提交。
- 已知限制：仅适合本地单进程单用户演示；关闭应用后提醒不会产生系统级通知。
- SQLite 迁移触发条件：多用户、并发写、目标/提醒超过本地展示规模，或需要后台调度器。
  迁移时保留 UUID、版本号、时间戳和幂等键；JSON 文件作为只读回退，校验记录数和摘要后
  才切换写路径。
- 验证证据：P1 工具契约测试、确认前后状态测试、重复确认、令牌篡改、目标版本和提醒状态
  历史测试，以及完整 pytest 回归。
