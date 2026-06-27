## 1. RoTE 时间建模

- [x] 1.1 实现可配置多粒度 RoTE 时间编码器。
- [x] 1.2 增加 `sasrec_rote` 模型变体。
- [x] 1.3 增加 `tisasrec_rote` 模型变体，并支持相对 bias / RoTE 消融。
- [x] 1.4 扩展模型工厂和配置加载。
- [x] 1.5 保持原 SASRec / TiSASRec 调用兼容。

## 2. Sequence Split Audit

- [x] 2.1 增加 leave-one-out、no SSS、sliding-window SSS、prefix-target SSS 配置。
- [x] 2.2 在所有协议中保留 timestamp 序列和 target timestamp。
- [x] 2.3 增加 target leakage 检查。
- [x] 2.4 增加跨 split protocol 的运行脚本和配置。

## 3. 评估与观测

- [x] 3.1 按 split protocol 汇总指标。
- [x] 3.2 增加 history length、time gap、item popularity、category switch 切片。
- [x] 3.3 增加平均延迟、p95 延迟、显存/内存摘要。
- [x] 3.4 导出 aggregate 和 slice 结果表。

## 4. 测试与文档

- [x] 4.1 测试 RoTE encoder shape 和确定性。
- [x] 4.2 测试 RoTE 模型 forward 输出有限值。
- [x] 4.3 测试 split protocol 保留 timestamp 且不泄漏 target。
- [x] 4.4 测试 audit 输出包含 aggregate、slice、runtime 字段。
- [x] 4.5 更新 README，说明 RoTE-TimeRec 和 SSS audit 用法。
