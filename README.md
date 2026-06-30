# RoTE-TimeRec：多粒度时间建模与序列评测审计

RoTE-TimeRec 基于原 TimeGenRec 代码资产升级而来，保留 SASRec、TiSASRec、TiSASRec-Cat、full-ranking 评估和多阶段推荐链路，在此基础上加入 2026 年更前沿的多粒度时间建模和序列切分评测审计。

项目核心问题是：

> 序列推荐里的时间信息到底应该如何建模？离线指标提升究竟来自模型能力，还是来自 sequence split 协议带来的评测偏差？

## 项目定位

RoTE-TimeRec 是项目矩阵里的传统序列推荐主项目（当前优先级 P1）：

| 项目 | 角色 | 当前状态 |
|------|------|---------|
| 快手 LLM-Rec 挑战赛 | 真实工业数据比赛 + LLM4Rec 实战 | **P0（进行中）** |
| RoTE-TimeRec | 时间建模、full-ranking 评估、评测可信度 | **P1（进行中）** |
| MiniMind-IntentRec | LLM / MiniMind 蒸馏用户会话意图 | P2（8.1 后） |
| Gryphon-lite | Semantic ID 生成式推荐（冻结） | P3（暂停） |

这个项目不是推倒重写 TimeGenRec，而是在已有推荐系统底座上做可信升级。

## 技术底座

- SASRec：自注意力序列推荐基线。
- TiSASRec：相对时间间隔 attention bias。
- TiSASRec-Cat：同类 / 跨类目条件时间偏置。
- Popularity / ItemCF / **DSSM**：召回基线（DSSM 双塔 + Faiss ANN 索引）。
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

## 多阶段推荐管道

```
数据 → 召回 (Recall) → 粗排 (PreRank) → 精排 (Rank) → 重排 (ReRank) → 推荐列表
```

当前管道实现了三路召回 + 粗排 + 排序 + 重排的完整链路。

### 召回阶段 (Recall)

召回层负责从全量物品库中快速筛选候选，当前支持三路召回：

#### 1. PopularityRecall — 热门召回

**原理：** 全局统计物品交互频次，按热度排序。所有用户返回相同的 Top-K 热门物品。适合冷启动用户兜底和 baseline 对照。

**训练：** `fit()` 接收 `item_counts: Dict[int, int]`，按 value 降序排列保存 top_items。

**评分：** `score = 1.0 / (rank + 1)`，排名越靠前得分越高。

**特点：** 无个性化，O(1) 推理，命中率高但不具备区分度。

#### 2. ItemCFRecall — 物品协同过滤

**原理：** 利用用户历史交互序列，统计物品共现次数作为相似度。对用户历史中的每个物品，找到最相似的物品并累加分数。

**训练：** `fit()` 接收 `item_sim: Dict[int, Dict[int, float]]` 相似矩阵。

**评估：** predict() 遍历用户历史物品，在 sim_matrix 中查找相似物品，按累加分数取 Top-K。

**当前简化：** 使用原始共现次数而非余弦归一化 `sim(i,j) = co_count / sqrt(count(i) * count(j))`，后续可升级。参见 [Day2 笔记](docs/Day2_召回基础_FunRec召回全景_结合RoTE-TimeRec.md) 中的余弦归一化版本。

#### 3. DSSMRecall — 双塔向量召回

**原理：** 双塔模型将 user 和 item 分别编码为固定维度的向量（经过 MLP 和 L2 归一化），通过向量点积衡量相关性，使用 Faiss 加速 Top-K 检索。

```
user_id → Embedding → MLP → L2 Norm → user_emb
                                                → dot product → score
item_id → Embedding → MLP → L2 Norm → item_emb
```

**训练方式对比：**

| 方式 | Loss | 负样本策略 | 适用场景 | 速度 |
|------|------|-----------|---------|------|
| In-batch softmax | CrossEntropy | batch 内其他样本为负 | 密集信号、大 batch | 快 |
| BPR pairwise | `-log σ(pos - neg)` | 显式采样 num_neg 个随机负样本 | 稀疏 CF 数据 | 中（需优化） |

**遇到的问题与解决：**

1. **in-batch softmax 在稀疏数据上效果差**  
   → Amazon Beauty (94K 交互, 6K items) 上 Recall@50 = 0.0095，接近随机猜测。原因：稀疏 CF 中 batch 内其他用户的样本不能代表真实负分布。  
   → **解决：** 增加 BPR pairwise loss + 显式随机负采样训练。

2. **BPR 负样本多次前向导致训练极慢**  
   → 初始实现让 `num_neg=200` 个负样本各自通过 MLP，每 epoch 耗时 7 分钟。  
   → **解决：** 预计算全量 item embedding 表 `all_i_mlp = model.item_mlp(model.item_emb.weight)`，负样本直接从表中索引，避免重复计算。训练从 7 分钟/epoch 降至 15 秒/epoch。

4. **MLP 在稀疏 CF 数据上反而降低效果**  
   → DSSM 默认 MLP (64→128→64) 在 Amazon Beauty 上 Recall@10 = 0.002，去掉 MLP 后 Recall@10 = 0.069（提升 35x）。原因：MLP 的非线性变换破坏了 embedding 点积的线性可分性；稀疏数据下参数增多反而过拟合。  
   → **解决：** 添加 `no_mlp=True` 选项退化为标准 BPR-MF，在稀疏 ID-CF 场景推荐使用。

**Faiss 集成：**

```python
import faiss
index = faiss.IndexFlatIP(dim)    # 最大内积搜索
index.add(item_emb_matrix)        # 建索引
scores, indices = index.search(query, top_k)  # ANN 搜索
```

Faiss 为可选依赖，未安装时自动回退到 `np.argpartition` 暴力搜索（适合 item 数 < 10 万的小数据集）。

**Amazon Beauty 真实数据实验结果：**

```text
数据集: Amazon Beauty 2014 (ratings_Beauty.csv)
过滤:   5-core (rating ≥ 4)
样本:   10,553 用户, 6,086 物品, 94,148 交互
训练:   BPR loss, 50 epoch, batch=256, 随机负采样 100/pair
切分:   per-user leave-one-out（最后一条为 target，其余为训练）
评估:   排除用户历史已见物品

训练时间 (CPU i7): ~14 分钟 (17s/epoch)
```

| 配置 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|------|---------|---------|----------|----------|----------|
| MLP (64→128→64) | 0.000 | 0.001 | 0.002 | 0.004 | 0.008 |
| **纯 embedding (no MLP)** | **0.012** | **0.041** | **0.069** | **0.101** | **0.163** |

> MLP 版本接近随机猜测，去掉 MLP 后 Recall 提升 10-20x。原因：稀疏 CF 数据上 MLP 的非线性变换损害了 embedding 点积的线性可分性。后续可在更丰富特征（用户序列/物品属性）上重新引入 MLP。

### 粗排阶段 (PreRank)

SimplePreRank：多路召回合并后，按分数截断到 keep_k，减轻排序压力。

### 精排阶段 (Rank)

SequenceRanker：加载 SASRec/TiSASRec/RoTE 等序列模型，对候选物品打分排序。

### 重排阶段 (ReRank)

MMRReRank：最大边际相关性重排，平衡相关性与多样性。

### 管道运行

```bash
# 合成数据 smoke test（验证链路通断）
python scripts/run_pipeline.py

# 真实数据训练 DSSM + Faiss 召回评估（无 MLP 版本推荐）
python scripts/train_dssm_beauty.py --epochs 50 --batch-size 256 --no-mlp

# 对比 MLP 版本
python scripts/train_dssm_beauty.py --epochs 50 --batch-size 256
```

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

# Debug / smoke test：默认 leave_one_out，跑得快，但每用户通常只有 1 条训练样本
python scripts/train_model.py --model sasrec --epochs 3

# 正式主实验：训练 split 推荐使用 prefix_target_sss，样本量更充分
python scripts/train_model.py --model sasrec --split prefix_target_sss --epochs 3
python scripts/train_model.py --model tisasrec --split prefix_target_sss --epochs 3
python scripts/train_model.py --model tisasrec_cat --split prefix_target_sss --epochs 3
python scripts/train_model.py --model sasrec_rote --split prefix_target_sss --epochs 3
python scripts/train_model.py --model tisasrec_rote --split prefix_target_sss --epochs 3

# Split 消融：只选 1 个代表模型跑四种 split，看训练样本构造对指标的影响
python scripts/train_model.py --model tisasrec_rote --split leave_one_out --epochs 3
python scripts/train_model.py --model tisasrec_rote --split no_sss --epochs 3
python scripts/train_model.py --model tisasrec_rote --split sliding_window_sss --epochs 3
python scripts/train_model.py --model tisasrec_rote --split prefix_target_sss --epochs 3

# 运行多阶段召回-排序管道（含 DSSM 双塔向量召回 + Faiss ANN 搜索）
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

#### 训练 split 与评估协议的口径

`configs/default.yaml` 里默认是 `leave_one_out`，这是为了 demo、debug、smoke test 跑得快，不代表正式实验推荐这样训练。`leave_one_out` 通常每个用户只产生 1 条训练样本，样本量偏小，容易低估模型能力。

正式主实验建议固定训练协议为 `prefix_target_sss`：它把一个用户序列拆成多个“历史前缀 -> 下一个目标”的样本，样本量更充分，尤其适合 Amazon 这类短到中等长度用户序列。主结果表只需要在 `prefix_target_sss` 下比较 `sasrec`、`tisasrec`、`tisasrec_cat`、`sasrec_rote`、`tisasrec_rote`。

四种 split 不需要全部作为主实验跑。推荐分层如下：

| 用途 | 推荐做法 | 说明 |
|---|---|---|
| Debug / smoke test | `leave_one_out` | 默认值，最快，只验证代码链路和指标是否正常。 |
| 主实验训练 | `prefix_target_sss` | 样本量更充分，用来生成主结果表。 |
| 长序列训练对照 | `sliding_window_sss` | 适合长序列，可作为训练协议消融。 |
| 负面对照 / audit | `no_sss` | 不做子序列增强，主要用来说明样本量和协议偏差。 |
| Split 消融 | 只选 1 个代表模型跑四种 split | 例如只用 `tisasrec_rote` 或 `sasrec_rote` 跑 `leave_one_out/no_sss/sliding_window_sss/prefix_target_sss`。 |

评估协议要固定，不要每个模型用不同口径。推荐最终评估统一使用留一目标和 full-ranking 评估；SSS audit 的作用是暴露不同训练样本构造带来的指标偏差，而不是把四种 split 都当成主结果表。

## 范围

当前阶段聚焦离线训练、离线评估和可复现实验，不包含在线 serving 和分布式训练。
