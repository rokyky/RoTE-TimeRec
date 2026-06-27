# 任务：TimeGenRec 可信闭环

## 任务列表

### 阶段 1：评估加固（优先）

#### Task 1.1 — 真实 time_deltas 和 same_cat_mask 接入
- [x] 扩展 `EvalDataset`，支持返回 timestamps 和 item_categories
- [x] 改造 `_model_forward`：从真实时间戳计算 time_deltas 矩阵
- [x] 改造 `_model_forward`：从 item_categories 构建 same_cat_mask
- [x] 扩展 `model_eval` 签名，接受可选参数
- [x] 更新 `train_model.py` 调用，传入合成数据的时间戳和类目
- [x] 无真实数据时发出 warning

#### Task 1.2 — Full-ranking 排除训练集 item 和 padding
- [x] `evaluate_full_sort` 添加 `exclude_items` 参数
- [x] 默认排除 item 0
- [x] `model_eval` 透传 `exclude_items`
- [x] `train_model.py` 收集训练集已交互 item 全局集合并传入

### 阶段 2：真实数据预处理

#### Task 2.1 — prepare_data.py 重构
- [x] 添加 `--split-mode`（random / leave_one_out）
- [x] leave-one-out：每条用户序列最后 1 条 → test，倒数第 2 → val，其余 → train
- [x] 保留 `unixReviewTime`，输出 `timestamps.pt`
- [x] 下载/解析 meta JSON，输出 `item_categories.pt`
- [x] 所有随机操作使用统一 seed
- [x] 添加统计日志（均值/中位数序列长度、物品数、稀疏度、类目数）

### 阶段 3：测试套件

#### Task 3.1 — test_dataset.py
- [x] `test_seqrec_dataset_shapes`
- [x] `test_seqrec_dataset_padding`
- [x] `test_seqrec_dataset_target`
- [x] `test_eval_dataset_leave_last`

#### Task 3.2 — test_metrics.py
- [x] `test_recall_empty_gt`
- [x] `test_recall_perfect`
- [x] `test_ndcg_order_matters`
- [x] `test_mrr_first_position`

#### Task 3.3 — test_sasrec_forward.py
- [x] `test_sasrec_output_shape`
- [x] `test_sasrec_no_nan`
- [x] `test_sasrec_deterministic`

#### Task 3.4 — test_tisasrec_time_bias.py
- [x] `test_time_bias_has_effect`
- [x] `test_discretize_time_delta`

#### Task 3.5 — test_pipeline_smoke.py
- [x] `test_pipeline_end_to_end`
- [x] `test_pipeline_output_structure`

### 阶段 4：管道可观测性

#### Task 4.1 — PipelineStats 和统计收集
- [x] `PipelineStats` 数据类（base.py）
- [x] `PipelineStage` 增加统计包裹方法
- [x] `PipelineRunner.run` 返回 stats 列表
- [x] `run_pipeline.py` 打印汇总表

## 验证结果
- ✅ **32/32 测试通过** (`pytest tests/ -v`)
- ✅ **训练冒烟测试通过** — TiSASRec-Cat + 真实时间戳/类目评估
- ✅ **管道冒烟测试通过** — 5 阶段 + 统计表输出
- ✅ **prepare_data.py CLI 通过** — `--split-mode` / `--seed` / `--dataset` 参数正常
