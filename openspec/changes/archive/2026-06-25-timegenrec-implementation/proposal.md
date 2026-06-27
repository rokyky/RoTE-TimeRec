## 为什么需要这个项目
TimeGenRec 是一个独立完整的时序推荐系统项目。
覆盖召回 -> 粗排 -> 精排 -> 重排全链路。
核心差异化在于时间感知序列建模：SASRec / TiSASRec / TiSASRec-Cat。
在统一 full-ranking 协议下验证时间间隔和类目条件时间偏置的适用边界。

## 实现内容
- src/models/：SASRec、TiSASRec、TiSASRec-Cat
- src/pipeline/：Recall -> PreRank -> Rank -> ReRank
- src/data/：SeqRecDataset + EvalDataset
- src/eval/：recall/ndcg/mrr、full-ranking、分桶评估
- src/trainer.py：组合式训练器，支持 early stopping / checkpoint / LR 调度
- src/utils/config.py：YAML 配置加载
- configs/default.yaml：实验配置
- scripts/train_model.py：训练入口
- scripts/run_pipeline.py：pipeline 入口
- tests/：单元测试 + 集成测试

## 能力定义
- seq-model：SASRec / TiSASRec / TiSASRec-Cat
- multi-stage-pipeline：召回粗排精排重排全链路
- trainer：训练循环、early stopping、checkpoint、LR 调度
- full-ranking-eval：全量 item 打分评估
- bucket-eval：分桶评估（popularity / category_entropy / seq_length）
- data-processing：序列数据集、负采样
- config：YAML 配置系统

## 范围
全新项目，所有代码独立实现。