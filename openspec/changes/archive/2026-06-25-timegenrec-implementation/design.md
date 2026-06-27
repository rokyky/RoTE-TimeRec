## 定位
完整时序推荐系统，核心叙事：统一 full-ranking 协议下验证时间偏置有效性边界。

## 目标
- 独立实现 SASRec / TiSASRec / TiSASRec-Cat
- 召回到重排完整 pipeline
- full-ranking + candidate-based 双评估协议
- YAML 配置驱动
- 合成数据端到端可运行

## 非目标
- 不上线 serving
- 不做 Faiss（DSSM stub）
- 不做分布式
- 不依赖外部推荐框架

## 架构
Sequences -> Recall(Pop/ItemCF) -> PreRank(统计特征) -> Rank(TiSASRec-Cat) -> ReRank(MMR)

## 决策
D1: 序列模型三层继承，forward 签名差异化
D2: Pipeline 配置驱动，YAML 控制各阶段参数
D3: 训练评估解耦，eval 由外部 callable 传入
D4: Full-ranking 用于模型对比，candidate-based 用于 pipeline 验证