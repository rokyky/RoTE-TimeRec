## ADDED Requirements

### Requirement: SHALL 提供 RoTE 时间编码器
系统 SHALL 提供可配置的多粒度时间编码器，用于 timestamp-aware sequential recommendation。

#### Scenario: 多粒度编码 timestamp
- **When** 输入 item 序列和 timestamp 序列
- **Then** RoTE 编码器输出与序列长度对齐的时间表示
- **And** 输出不包含 NaN 或 Inf

### Requirement: SHALL 提供 RoTE 模型变体
系统 SHALL 暴露 SASRec+RoTE 和 TiSASRec+RoTE 模型变体。

#### Scenario: 训练和评估 RoTE 变体
- **When** 配置模型名为 `sasrec_rote` 或 `tisasrec_rote`
- **Then** 模型使用 timestamp-aware RoTE 特征
- **And** 在现有 full-ranking 协议下输出 HR/NDCG/Recall

### Requirement: SHALL 保持模型调用向后兼容
系统 SHALL 保持原有 SASRec 和 TiSASRec 调用无需 timestamp 参数也能继续运行。

#### Scenario: 原 baseline forward 调用
- **When** 调用方只传入原有参数
- **Then** baseline 模型成功返回 item scores
- **And** 输出 shape 与原实现一致
