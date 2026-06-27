# 设计：TimeGenRec 可信闭环

## 1. 评估中接入真实时间差和类目掩码

**当前状态**（`src/eval/metrics.py:47-60`）：`_model_forward` 对 time_deltas 和 same_cat_mask 创建全零张量，导致 TiSASRec/TiSASRec-Cat 在评估中退化为 SASRec。

**设计**：
- 扩展 `EvalDataset`，可选返回时间戳和物品类目
- 扩展 `model_eval`，接受可选参数 `item_categories: Dict[int, int]` 和 `timestamps: Dict[int, List[float]]`
- 在 `_model_forward` 中，当有真实时间戳数据时，从物品级时间戳计算真实 time_deltas 矩阵
- 对于 `same_cat_mask`，查序列表中每个物品的类目并做两两比较
- 当无真实数据时，回退到零占位但发出明确的 **warning**

**关键决策**：保持向下兼容 — `model_eval` 在无时间戳/类目时仍可运行，通过 warning 提示而非崩溃。

## 2. Full-ranking 评估加固

**当前状态**：`evaluate_full_sort` 对所有 item 排名，包括 padding（item 0）和训练集已交互 item。这会导致指标虚高，因为模型可能"正确"预测了一个用户不会再交互的训练集 item。

**设计**：
- 为 `evaluate_full_sort` 和 `model_eval` 添加 `exclude_items: Set[int]` 参数
- 在 `torch.topk` 之前，将排除 item 的分数设为 `-inf`
- 默认排除 item 0（padding）
- 在 `model_eval` 中接受训练集已交互 item 作为排除集合
- `_model_forward` 中，打分后将排除 item 屏蔽再排名

**关键决策**：排除操作在打分之后完成（topk 前设为 -inf），而非限制 item embedding 矩阵。保持模型架构不变，实现更简单。

## 3. 真实数据预处理

**当前状态**：`scripts/prepare_data.py` 有基础下载+重映射逻辑，但：
- 使用随机用户切分而非 leave-one-out（最后交互作为测试）
- 不保留时间戳
- 不从元数据提取物品类目

**设计**：
- 为 `prepare_data.py` 添加 `--leave-one-out` 标志（默认 True）
- 从原始 JSON 保留 `unixReviewTime` 用于时间戳计算
- 下载并解析元数据 JSON，提取 `category`（类别路径最后一项 = 叶子类目）
- 输出额外文件：`timestamps.pt`、`item_categories.pt`
- 所有随机操作使用一致的 seed
- 添加关键统计日志（平均序列长度、稀疏度等）

**关键决策**：保留现有的 `--user-split` 模式作为 `--split-mode random|leave-one-out`，保持向下兼容。

## 4. 测试套件

**当前状态**：`tests/` 目录存在但仅有 `__init__.py`。

**设计** — 五个测试文件：

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/test_dataset.py` | SeqRecDataset 形状、padding、target 对齐；EvalDataset leave-last-out |
| `tests/test_metrics.py` | recall_at_k、ndcg_at_k、mrr_at_k 边界情况（空 gt、k > ranked 长度） |
| `tests/test_sasrec_forward.py` | SASRec 前向传播形状、输出范围、无 NaN |
| `tests/test_tisasrec_time_bias.py` | TiSASRec 非零 vs 全零 time_deltas 产出不同分数 |
| `tests/test_pipeline_smoke.py` | 端到端：recall → pre-rank → rank → re-rank 无错误完成 |

**关键决策**：仅使用合成数据（快速、确定性、无需网络）。使用 pytest。

## 5. 管道可观测性

**当前状态**：管道各阶段静默运行。无法看到每阶段候选量、命中率、耗时。

**设计**：
- 添加 `PipelineStats` 数据类，记录：阶段名称、输入候选数、输出候选数、耗时(ms)、命中率（如果有 ground truth）
- 每个 `PipelineStage.predict` 包裹自身逻辑记录统计
- `PipelineRunner.run` 返回 `(CandidateList, List[PipelineStats])`
- `run_pipeline.py` 打印汇总表

**关键决策**：统计收集通过 `PipelineRunner(collect_stats=True)` 按需开启。内部使用时不收集以减少开销。
