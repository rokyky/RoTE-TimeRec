## 总体设计

本变更将 RoTE-TimeRec 升级为“时间建模 + 评测审计”项目。RoTE 作为可选时间模块接入模型层，SSS audit 作为数据构造和评估协议层接入，不破坏现有 TimeGenRec 结构。

## 模型设计

### RoTE 时间编码器

新增时间编码器，将 timestamp 或 pairwise time gap 转为多粒度时间表示。粒度可以包含小时、天、周、月等 bucket。

模型接口保持兼容：

```text
SASRec.forward(seqs, positions, timestamps=None)
TiSASRec.forward(seqs, positions, time_deltas, timestamps=None)
```

没有传入 `timestamps` 时，原模型行为不变。

### 模型变体

新增配置名称：

- `sasrec`
- `tisasrec`
- `tisasrec_cat`
- `sasrec_rote`
- `tisasrec_rote`

`tisasrec_rote` 需要支持相对时间 bias 和 RoTE 分别开启/关闭，用于消融。

## 数据与切分设计

支持四种序列切分：

- `leave_one_out`：接近 serving 的留一评估。
- `no_sss`：每个用户一条训练序列，不做子序列扩增。
- `sliding_window_sss`：滑窗构造多条训练样本。
- `prefix_target_sss`：多个 prefix-target 训练样本。

每个样本必须保留：

- item 序列
- timestamp 序列
- target item
- target timestamp
- user id

## 评估设计

评估结果需要增加：

- split protocol 字段
- hard-slice 指标
- latency / memory 汇总

hard slice 包括：

- short-history / long-history 用户
- short-gap / long-gap target
- head / tail target item
- same-category / category-switch session

## 测试策略

- RoTE encoder shape 和确定性测试。
- SASRec / TiSASRec 旧接口兼容测试。
- 各 split protocol 样本数量和 timestamp 保留测试。
- target leakage 检查。
- audit 输出字段完整性测试。

## 风险

- RoTE 实现可能退化成普通 time bias。需要保留清晰文档和消融开关。
- SSS 构造可能发生 target leakage。需要测试覆盖。
- 模型变体增加后配置会变复杂。默认配置必须保持原行为。
