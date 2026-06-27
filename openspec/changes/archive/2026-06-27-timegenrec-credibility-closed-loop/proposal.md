# 提案：TimeGenRec 可信闭环

## Why
TimeGenRec 结构完整，但在推荐系统的五个可信支柱上缺乏**可信证据**：数据管道、评估协议、模型正确性、测试覆盖、管道可观测性。README 描述了一个完整系统，但多个关键路径仅为占位实现，导致"时间感知"和"类目条件"宣称难以验证。当前评估代码中 TiSASRec/TiSASRec-Cat 使用全零张量评估，时间感知和类目条件宣称完全无效；full-ranking 评估未排除训练集已交互 item 和 padding，指标可能虚高；无真实数据预处理链路；无测试覆盖；管道运行无统计可见。

## What Changes
五项针对性改进，按收益-成本比排序：

1. **评估中接入真实 time_deltas 和 same_cat_mask** — TiSASRec/TiSASRec-Cat 在 `model_eval` 中使用真实时间戳和类目计算评估所需矩阵
2. **full-ranking 评估加固** — 屏蔽训练集已交互 item 和 padding item 0，防止指标虚高
3. **真实数据预处理** — 为 Amazon Beauty/Sports 补 leave-one-out 切分、时间戳保留、类目编码
4. **测试套件** — 5 个测试文件：dataset、metrics、SASRec forward、TiSASRec time bias、pipeline 端到端冒烟测试
5. **管道可观测性** — 每阶段候选量、命中率、耗时统计

### 不包含
- 在线 serving / Faiss 集成
- 新模型架构
- Bootstrap 显著性检验（后续跟进）
- 分布式训练
