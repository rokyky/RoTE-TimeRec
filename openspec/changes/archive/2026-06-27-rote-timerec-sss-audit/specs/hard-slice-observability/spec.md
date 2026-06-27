## ADDED Requirements

### Requirement: SHALL 输出 hard-slice 指标
系统 SHALL 在评估报告中输出时间和行为切片指标，用于解释模型在不同困难样本上的表现。

#### Scenario: 生成切片指标
- **When** 输入 full-ranking scores、targets、timestamps、categories 和 item popularity
- **Then** 输出 history length、target time gap、target item popularity、category switch 等切片指标

### Requirement: SHALL 输出延迟和内存观测
系统 SHALL 报告模型运行成本，支持比较 baseline 与 RoTE 时间模块的工程开销。

#### Scenario: 比较时间模块开销
- **When** 评估 baseline 和 RoTE 模型
- **Then** 报告必须包含平均延迟、p95 延迟和可用的峰值内存信息
