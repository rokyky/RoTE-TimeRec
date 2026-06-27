## ADDED Requirements

### Requirement: SHALL 支持多种序列切分协议
数据层 SHALL 支持 leave-one-out、no SSS、sliding-window SSS 和 prefix-target SSS。

#### Scenario: 按协议构造样本
- **When** 选择某个 split protocol
- **Then** 生成样本必须保留输入 item、输入 timestamp、target item、target timestamp 和 user id

### Requirement: SHALL 防止 target leakage
数据层 SHALL 确保生成的 history 不包含当前评估 target item。

#### Scenario: 验证 target 排除
- **When** 对评估样本运行 leakage 检查
- **Then** target item 不得出现在输入 prefix 中

### Requirement: SHALL 输出 split audit 报告
评估器 SHALL 按 split protocol 分组输出指标。

#### Scenario: 比较切分协议
- **When** 同一模型在多个 split protocol 下评估
- **Then** 输出必须包含每个 protocol 的 HR/NDCG/Recall
