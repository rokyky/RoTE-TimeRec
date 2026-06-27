# RoTE-TimeRec：多粒度时间建模与序列评测审计

RoTE-TimeRec 基于原 TimeGenRec 代码资产升级而来，保留 SASRec、TiSASRec、TiSASRec-Cat、full-ranking 评估和多阶段推荐链路，在此基础上加入 2026 年更前沿的多粒度时间建模和序列切分评测审计。

项目核心问题是：

> 序列推荐里的时间信息到底应该如何建模？离线指标提升究竟来自模型能力，还是来自 sequence split 协议带来的评测偏差？

## 项目定位

RoTE-TimeRec 是三项目矩阵里的传统序列推荐主项目：

| 项目 | 角色 |
|---|---|
| RoTE-TimeRec | 时间建模、full-ranking 评估、评测可信度 |
| MiniMind-IntentRec | LLM / MiniMind 蒸馏用户会话意图 |
| Gryphon-lite | Semantic ID 生成式推荐与 item-level scoring 校准 |

这个项目不是推倒重写 TimeGenRec，而是在已有推荐系统底座上做可信升级。

## 技术底座

- SASRec：自注意力序列推荐基线。
- TiSASRec：相对时间间隔 attention bias。
- TiSASRec-Cat：同类 / 跨类目条件时间偏置。
- Popularity / ItemCF：召回基线。
- Recall -> PreRank -> Rank -> ReRank：候选链路。
- full-ranking / candidate-based：双评估协议。
- bucket evaluation：按用户、物品、类目和时间切片分析。

## 新增方向

### RoTE 多粒度时间表示

引入 RoTE 风格的 coarse-to-fine multi-level rotary time embedding，将真实时间跨度拆成多个粒度后注入 Transformer 序列模型。

对比矩阵：

| 模型 | 时间信号 | 目的 |
|---|---|---|
| SASRec | 仅位置编码 | 纯序列基线 |
| TiSASRec | 相对时间间隔 | interval-aware 基线 |
| TiSASRec-Cat | 时间间隔 + 类目关系 | 类目条件时间动态 |
| SASRec + RoTE | 多粒度时间 embedding | plug-in 时间模块 |
| TiSASRec + RoTE | 相对 bias + RoTE | 时间模块互补性消融 |

### SSS Evaluation Audit

引入 sequence split audit，比较不同序列构造方式对 HR/NDCG 的影响：

- leave-one-out
- no sub-sequence split
- sliding-window SSS
- prefix-target SSS

目标不是单纯刷指标，而是判断离线协议是否接近真实 serving 场景。

## 评估体系

主指标：

- HR@K / Recall@K
- NDCG@K
- MRR
- candidate-stage hit rate

可信度指标：

- 不同 split protocol 下的指标变化
- short-history / long-history 用户切片
- short-gap / long-gap 时间切片
- long-tail item 切片
- category-switch session 切片
- bootstrap 置信区间
- 平均延迟 / p95 延迟 / 显存占用

## 当前边界与必须补的实验

当前代码层面已经形成 RoTE 时间建模、split audit、hard-slice 和 runtime 观测闭环；真正的风险不在“能不能跑”，而在真实数据实验是否足够可信。

### 已解决的代码级风险

- 5 种模型变体可以通过统一 `model_eval` 路由评估。
- 4 种 split protocol 可以产出带 timestamp 的样本。
- full-ranking 默认排除 padding item，并支持排除训练集已交互 item。
- split audit 会检查 target leakage，并输出 aggregate / slice / runtime 结果。

### 当前实验硬伤

- README 中的结果表如果仍是占位，不能在面试里当作有效结论。
- 必须在 Amazon Beauty / Sports / Toys 等真实数据上跑出主结果表。
- 必须做完整消融：`sasrec`、`tisasrec`、`tisasrec_cat`、`sasrec_rote`、`tisasrec_rote`。
- 必须报告 RoTE 在 short-history、long-gap、category-switch、long-tail 等切片上的收益，否则很难说明 RoTE 不是普通 time embedding。
- 必须固定 seed、split protocol 和评估脚本，避免指标来自随机切分或 SSS 协议偏差。

### 面试叙事边界

推荐表述：这是一个“时间建模 + 可信离线评测闭环”的序列推荐项目。RoTE 是可插拔时间模块，重点不是宣称全面超越所有时间模型，而是验证多粒度绝对时间信息与 TiSASRec 相对时间 bias 是否互补。

不推荐表述：不要说已经在真实工业流量上验证，也不要把 synthetic smoke 的指标当作主实验结论。

## 当前 OpenSpec

```text
openspec/changes/rote-timerec-sss-audit/
```

该变更定义 RoTE 时间模块、SSS 评测审计、hard-slice 观测和测试要求。

## 快速开始

```bash
pip install torch numpy scipy pyyaml

# 训练基线模型
python scripts/train_model.py --model sasrec --epochs 3

# 训练 RoTE 变体
python scripts/train_model.py --model sasrec_rote --epochs 3
python scripts/train_model.py --model tisasrec_rote --epochs 3

# 使用不同 split 协议训练
python scripts/train_model.py --model sasrec --split sliding_window_sss --epochs 3
python scripts/train_model.py --model sasrec --split prefix_target_sss --epochs 3

# 运行多阶段召回-排序管道
python scripts/run_pipeline.py

# 运行 SSS Audit（跨 split 协议对比）
python -c "
from src.eval.audit import SplitProtocolAuditor
from src.models import build_model
from src.data.split_protocols import apply_split
# ... 参见 tests/test_audit_report.py 完整示例
"

# 运行测试
python -m pytest tests/ -v
```

## 算力估算与实验建议

### 最低配置

单卡 **RTX 4090（24GB）** 完全足够，A100 只是压缩时间不是必须。

### 资源估算

| 版本 | 实验范围 | 4090 单卡 | A100 单卡 |
|------|---------|----------|----------|
| 最小闭环 | Beauty × 1 seed，SASRec / TiSASRec / RoTE，少量 epoch | 4–8 h | 3–6 h |
| **可投递可信版** | Beauty + Sports × 2–3 seeds，RoTE 消融，SSS audit，hard slice | **20–40 h** | **15–30 h** |
| 完整实验 | 3 数据集 × 3 seeds，多 split、多 RoTE 变体、pipeline 全评估 | 50–100 h | 35–75 h |

### 必须跑的实验

```
SASRec（基线）
TiSASRec（时间间隔基线）
TiSASRec-Cat（类目条件时间偏置）
SASRec + RoTE（多粒度时间 embedding）
TiSASRec + RoTE（相对 bias + RoTE 消融）
SSS audit（leave_one_out / no_sss / sliding_window / prefix_target）
hard slice（short/long history、time gap、popularity、category switch）
```

### 可以砍的实验

- 所有模型都跑 3 seeds → 主对比 1 seed，核心模型补 3 seeds
- 所有 split 都跑 3 seeds → 1 seed 足以观测 split 偏差
- 三个数据集全量 audit → Beauty + Sports 足够

### 建议跑法

1. **Beauty 小闭环**（4–8 h）：确认代码全通、指标正常
2. **补 Sports + 多 seed**（16–32 h）：RoTE 消融、SSS audit、hard slice

性价比最高的投入产出比：**Beauty + Sports × 2–3 seeds × 全部消融**。

### RoTE 模型变体

| 模型 | 命令 |
|------|------|
| SASRec (基线) | `--model sasrec` |
| TiSASRec | `--model tisasrec` |
| TiSASRec-Cat | `--model tisasrec_cat` |
| SASRec + RoTE | `--model sasrec_rote` |
| TiSASRec + RoTE | `--model tisasrec_rote` |

TiSASRec + RoTE 支持消融开关（`configs/default.yaml`）：
- `use_relative_bias: true/false` — 启用/禁用 TiSASRec 时间间隔偏置
- `use_rote: true/false` — 启用/禁用 RoTE 多粒度时间编码

### Split 协议

| 协议 | 说明 | 参数 |
|------|------|------|
| `leave_one_out` | 留一评估（接近 serving） | — |
| `no_sss` | 每用户一条序列，不做子序列扩增 | — |
| `sliding_window_sss` | 滑窗构造多条训练样本 | `sliding_window_size` |
| `prefix_target_sss` | 多个 prefix-target 训练样本 | `prefix_min_len` |

## 范围

当前阶段聚焦离线训练、离线评估和可复现实验，不包含在线 serving、分布式训练和 Faiss 向量召回。
