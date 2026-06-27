## 为什么需要这个变更

RoTE-TimeRec 已经有可复用的序列推荐底座：SASRec、TiSASRec、TiSASRec-Cat、full-ranking 评估和多阶段候选链路。下一步不应该推倒重做，而是补强两个关键问题：

1. 时间建模是否真的有效：对比位置编码、相对时间间隔、类目条件时间偏置和 RoTE 多粒度时间表示。
2. 离线评测是否可信：审计 sequence split 方式是否放大或扭曲 HR/NDCG。

## 改动内容

- 增加 RoTE 风格的多粒度 rotary time embedding。
- 增加 `sasrec_rote` 和 `tisasrec_rote` 模型变体。
- 增加 sequence split 协议：leave-one-out、no SSS、sliding-window SSS、prefix-target SSS。
- 扩展 hard-slice 评估：短/长历史用户、短/长时间间隔、长尾物品、类目切换会话。
- 增加延迟和显存观测，用于比较时间模块的工程成本。
- 保持现有 TimeGenRec API 和非 RoTE 模型行为兼容。

## 不做什么

- 不替换整个模型栈。
- 不做在线 serving。
- 不做分布式训练。
- 不引入 HoloMamba / Mamba 主干。
- 不做用户意图蒸馏。

## 验收标准

- RoTE 变体能在现有 full-ranking 协议下训练和评估。
- 同一模型能在至少三种 sequence split 协议下输出可比指标。
- 评估报告包含 aggregate 指标、hard-slice 指标和 runtime 指标。
- 原有 SASRec / TiSASRec / TiSASRec-Cat 使用方式保持可运行。
