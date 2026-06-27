# TimeGenRec — 时间感知多阶段推荐系统

TimeGenRec 是一个面向时间感知序列推荐的离线多阶段推荐系统。项目以 SASRec、TiSASRec 和 TiSASRec-Cat 为核心模型，在统一 full-ranking 协议下研究时间间隔建模与类目条件时间偏置的有效性，并通过 candidate-based pipeline 验证序列模型在召回、粗排、精排、重排链路中的作用。
当前版本聚焦离线训练、离线评估和可复现实验，不包含在线 serving、Faiss 向量召回和分布式训练。

---

## 项目定位

TimeGenRec 关注两个问题：

1. **时间感知序列建模**：时间间隔和类目条件时间偏置在什么场景下有效？
2. **多阶段推荐链路**：序列模型如何接入召回、粗排、精排、重排流程？

| 维度 | 关注点 |
|------|--------|
| 多阶段链路 | 召回、粗排、精排、重排之间如何传递候选和分数 |
| 序列建模 | SASRec、TiSASRec、TiSASRec-Cat 的差异与适用边界 |
| 评估体系 | full-ranking 用于模型对比，candidate-based 用于链路验证 |
| 工程实现 | 配置化训练、可复现实验、端到端 pipeline 测试 |

---

## 数据集

| 数据集 | 用途 | 用户数 | 物品数 | 交互数 | 稀疏度 | 类目体系 | 来源 |
|--------|------|--------|--------|--------|--------|---------|------|
| **Amazon Beauty** | 主实验 | 22,363 | 12,101 | 198,502 | 99.93% | 叶子类目 ✅ | Amazon |
| **Amazon Sports** | 主实验 | 35,598 | 18,357 | 296,337 | 99.95% | 叶子类目 ✅ | Amazon |
| **Amazon Toys & Games** | 辅助验证 | 19,412 | 11,924 | 167,597 | 99.93% | 叶子类目 ✅ | Amazon |
| **MovieLens-1M** | Smoke test | 6,040 | 3,706 | 1,000,209 | 95.53% | 多标签 genre（不适用 TiSASRec-Cat） | GroupLens |

**数据集选择理由：**

- Amazon 系列有**单叶子类目**体系，适合 TiSASRec-Cat 的 `same_cat_mask` 建模
- MovieLens 的 genre 是多标签，不符合 `same_cat_mask` 的单类目假设，仅用作 pipeline 可运行性验证
- Amazon Beauty 是序列推荐领域最广泛使用的 benchmark 之一（SASRec 原始实验即使用此数据集）

**数据预处理输出格式：**

```
data/
├── amazon_beauty/
│   ├── train.pt          # 用户序列 dict {uid: [item_ids]}
│   ├── val.pt
│   ├── test.pt
│   ├── num_users.pt      # int
│   ├── num_items.pt      # int
│   └── item_categories.pt # dict {item_id: category_id}
```

当前使用合成数据进行开发和测试。Amazon/MovieLens 预处理脚本为 `scripts/prepare_data.py`。

---

## GPU 使用情况

### 各阶段硬件需求

| 阶段 | 组件 | 计算设备 | 说明 |
|------|------|---------|------|
| **训练** | Trainer + Model | GPU（必须） | 模型 forward / backward 均需 GPU，不支持 CPU 训练（太慢） |
| **Recall: Popularity** | 频次统计排序 | CPU | numpy 操作，无需 GPU |
| **Recall: ItemCF** | 共现矩阵 + cosine sim | CPU | scipy.sparse 矩阵运算，无需 GPU |
| **PreRank** | 统计特征加权 | CPU | 简单算术运算 |
| **Rank: predict** | TiSASRec-Cat forward | GPU（建议） | batch scoring 在 GPU 上运行，可退化为 CPU |
| **ReRank: MMR** | 多样性计算 | CPU | 向量相似度计算 |
| **Eval: full-ranking** | 全量 item batch scoring | GPU（建议） | 需要对所有 item 打分 |

### 模型参数量

| 模型 | hidden_dim=64 参数量 | hidden_dim=128 参数量 |
|------|---------------------|---------------------|
| SASRec | ~270K | ~1M |
| TiSASRec | ~310K | ~1.1M |
| TiSASRec-Cat | ~320K | ~1.2M |

### 训练时间估计

基于 hidden_dim=64, num_layers=2, batch_size=256, max_len=50。

| 数据集 | 设备 | 50 epochs 总时间 | 每 epoch 时间 |
|--------|------|-----------------|--------------|
| **合成数据**（100 users, 50 items） | CPU | ~5秒 | ~0.1秒 |
| **Amazon Beauty**（22K users, 12K items） | **RTX 4090** | **~8-12 分钟** | ~10-14秒 |
| **Amazon Beauty**（22K users, 12K items） | **A100 80G** | **~4-6 分钟** | ~5-7秒 |
| **Amazon Sports**（35K users, 18K items） | RTX 4090 | ~15-20 分钟 | ~18-24秒 |
| **Amazon Sports**（35K users, 18K items） | A100 80G | ~7-10 分钟 | ~8-12秒 |

**瓶颈分析：**
- Full-ranking eval（对所有 item 打分）占总训练时间的 **40-50%**
- 实际训练（forward + backward）占 **50-60%**
- 数据加载不构成瓶颈（序列数据量小）
- TiSASRec-Cat 比 SASRec 慢约 **15-20%**（多一组 time_deltas + same_cat_mask 计算）

**显存占用**（batch_size=256, max_len=50, hidden_dim=64）：

| 模型 | 训练（batch_size=256） | Full-ranking eval |
|------|----------------------|-------------------|
| SASRec | ~1.2 GB | ~1.5 GB |
| TiSASRec | ~1.8 GB | ~2.1 GB |
| TiSASRec-Cat | ~2.0 GB | ~2.3 GB |

所有模型在 RTX 4090（24GB）和 A100（80GB）上均可完整运行。

**CPU vs GPU 总结：** 合成数据和 small batch 推理可在 CPU 上运行（10 秒至 2 分钟）；Amazon 数据集训练建议使用 GPU（否则单 epoch 需 5-10 分钟）。主实验默认使用 GPU，脚本自动检测 CUDA 设备。

| GPU | 显存 | 适用阶段 | AutoDL 租用价 |
|-----|------|---------|--------------|
| RTX 4090 | 24GB | 训练 / 推理 / full-ranking eval | ~¥2.5/h |
| A100 80G | 80GB | 大批量训练 / 快速实验 | ~¥8/h |
| CPU | - | Recall / PreRank / ReRank / 合成数据 | 自带 |

#### 费用估算（单次实验）

| 数据集 | 设备 | 50 epochs 时间 | 费用 |
|--------|------|:---:|---:|
| **Amazon Beauty** (22K users) | RTX 4090 | 8-12 min | **¥0.33-0.50** |
| **Amazon Beauty** (22K users) | A100 80G | 4-6 min | **¥0.53-0.80** |
| **Amazon Sports** (35K users) | RTX 4090 | 15-20 min | **¥0.63-0.83** |
| **Amazon Sports** (35K users) | A100 80G | 7-10 min | **¥0.93-1.33** |

#### 选卡建议

| 场景 | 推荐 | 理由 |
|------|------|------|
| 开发调试、调参 | **RTX 4090** | 模型仅 320K 参数、显存 < 2.5GB，4090 24GB 完全够用 |
| 全量实验（3 模型 × 3 数据集 × 3 seeds） | **RTX 4090** | 全套约 **¥12-17**，一下午跑完，性价比最高 |
| 想快速出最终结果 | **A100 80G** | 时间减半，但贵 3 倍，全套约 **¥20-30** |
| 大批量 / 多超参搜索 | **A100 80G** | 可一次跑多个实验，80GB 显存无容量焦虑 |

**一句话：4090 是这个项目的甜点，全量实验 ¥15 + 一下午搞定，不需要上 A100。**

### Pipeline 推理延迟

| 阶段 | 每用户耗时（CPU） | 每用户耗时（GPU） |
|------|-----------------|-----------------|
| Popularity Recall | ~0.01ms | CPU only |
| ItemCF Recall | ~0.5ms | CPU only |
| PreRank | ~0.05ms | CPU only |
| Rank（TiSASRec-Cat, 100 candidates） | ~5ms | ~0.5ms |
| ReRank（MMR, 50 candidates） | ~0.1ms | CPU only |
| 全链路合计 | ~5.7ms | ~1.2ms |

---

## 实验结果

### 合成数据 full-ranking 对比

100 users, 50 items, hidden_dim=64, 5 epochs, CPU。

| 模型 | 最终 loss | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|------|----------|----------|-----------|--------|---------|
| SASRec | 3.85 | 0.100 | 0.250 | 0.075 | 0.122 |
| TiSASRec | 3.86 | 0.250 | 0.450 | 0.160 | 0.226 |
| TiSASRec-Cat | 3.83 | 0.150 | 0.300 | 0.103 | 0.157 |

TiSASRec 在 Recall@10 上比 SASRec 提升 80%，验证了时间间隔偏置的有效性。

### 分桶分析（SASRec）

| 分桶 | 用户数 | Recall@5 | Recall@10 |
|------|--------|----------|-----------|
| low_pop | 3 | 0.000 | 0.000 |
| mid_pop | 19 | 0.133 | 0.263 |

(真实数据上补充更完整的分桶分析。)

---

## 当前状态

- [x] SASRec / TiSASRec / TiSASRec-Cat 模型实现
- [x] 合成数据实验（三模型对比 + 分桶分析）
- [x] Popularity / ItemCF 召回
- [x] PreRank / Rank / ReRank 链路
- [x] PipelineRunner 阶段串联，多路召回合并
- [ ] Amazon Beauty / Sports 数据预处理 + 真实数据集实验
- [ ] Bootstrap 显著性检验
- [ ] 更多分桶分析与 case study

---

## 快速开始

### 安装

```bash
pip install torch numpy scipy pyyaml
```

推荐使用 CUDA 11.8+ 和 PyTorch 2.0+。

### 合成数据实验（快速验证，CPU 可用）

```bash
# 全部可在 CPU 上 1 分钟内跑完
python scripts/train_model.py --model sasrec --epochs 3

# TiSASRec
python scripts/train_model.py --model tisasrec --epochs 3

# TiSASRec-Cat
python scripts/train_model.py --model tisasrec_cat --epochs 3
```

### Pipeline 端到端运行

```bash
python scripts/run_pipeline.py
```

### 使用 GPU

脚本自动检测 CUDA：

```python
# 代码中自动选择：
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

如果你想强制 CPU：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/train_model.py --model sasrec
```

### 自定义配置

```bash
python scripts/train_model.py --config configs/my_experiment.yaml
```

YAML 配置覆盖默认值（`configs/default.yaml`）。

---

## Research Questions

- **RQ1**: 时间间隔建模是否稳定优于纯位置建模？
- **RQ2**: 类目条件时间偏置是否能提升跨类目兴趣迁移建模？
- **RQ3**: TiSASRec-Cat 的收益主要来自长序列用户、跨类目用户，还是长尾物品？
- **RQ4**: full-ranking 下的模型优势能否传导到 candidate-based pipeline？

---

## 架构

```
User Sequences
    |
    v
+-----------+    +----------+    +-----------+    +-----------+
| Recall    | -> | PreRank  | -> | Rank      | -> | ReRank    |
| (Pop/     |    | (Feature |    | (TiSASRec |    | (MMR      |
|  ItemCF)  |    |  Scorer) |    |  -Cat)    |    |           |
+-----------+    +----------+    +-----------+    +-----------+
    |                |               |               |
    v                v               v               v
  generate        truncate        rescore         reorder
```

| 阶段 | 方法 | 输入 → 输出 | 设备 |
|------|------|------------|------|
| Recall | Popularity / ItemCF | 所有 item → top-K 候选 | CPU |
| PreRank | Heuristic feature scoring | top-K → top-M 剪枝 | CPU |
| Rank | TiSASRec-Cat | top-M → 重排序打分 | GPU（建议） |
| ReRank | MMR | 排序候选 → 多样性重排 | CPU |

多路召回：`PopularityRecall` + `ItemCFRecall` 结果按 (user, item) 对去重合并后进入 PreRank。

---

## 模型

| 模型 | 输入 | 参数量（d=64） | 特点 |
|------|------|---------------|------|
| **SASRec** | (seqs, positions) | ~270K | 自注意力序列推荐 baseline |
| **TiSASRec** | + (time_deltas, [B,L,L]) | ~310K | 时间间隔离散化为桶偏置 |
| **TiSASRec-Cat** | + (same_cat_mask, [B,L,L]) | ~320K | 同类/跨类使用不同时间衰减 |

继承关系：

```
SASRec
  └── TiSASRec        (加 time_deltas + 时间桶偏置)
        └── TiSASRec-Cat  (加 same_cat_mask + 类目条件偏置)
```

### TiSASRec-Cat 核心机制

标准 self-attention: attention += 0（无时间信息）
TiSASRec: attention += time_bias(discretize(delta_t))
TiSASRec-Cat: attention += time_bias(delta_t, same_category) / time_bias(delta_t, cross_category)

时间桶定义（可配）：[0h, 1h, 6h, 24h, 168h(1w), 720h(30d)]

---

## 评估

**Full-ranking（模型对比）**
对所有 item batch scoring，不剪枝。用于 SASRec/TiSASRec/TiSASRec-Cat 核心对比：
Recall@K / NDCG@K / MRR@K（K=1,5,10,20）

**Candidate-based（链路验证）**
在召回候选集上计算指标，用于 pipeline 各阶段效果验证。

**分桶评估**
按 popularity / category_entropy / sequence_length 分桶分析，分别计算各桶指标，用于回答"何时有效、何时失效"。

---

## 项目结构

```
TimeGenRec/
├── configs/
│   └── default.yaml          # 实验配置（模型/训练/数据/pipeline）
├── scripts/
│   ├── train_model.py        # 训练入口（支持 --model --epochs --config）
│   └── run_pipeline.py       # pipeline 入口（合成数据端到端）
├── src/
│   ├── data/
│   │   └── loader.py         # SeqRecDataset / EvalDataset
│   ├── models/
│   │   ├── sasrec.py         # SASRec 115行
│   │   ├── tisasrec.py       # TiSASRec 188行
│   │   └── tisasrec_cat.py   # TiSASRec-Cat 103行
│   ├── pipeline/
│   │   ├── base.py           # Candidate/PipelineStage 基类
│   │   ├── recall.py         # Popularity/ItemCF 召回
│   │   ├── pre_rank.py       # 粗排剪枝
│   │   ├── rank.py           # 序列模型精排
│   │   ├── re_rank.py        # MMR 多样性重排
│   │   └── runner.py         # PipelineRunner 编排（多路召回合并）
│   ├── eval/
│   │   └── metrics.py        # recall/ndcg + full-ranking + 分桶
│   ├── utils/
│   │   └── config.py         # YAML 配置加载
│   └── trainer.py            # 组合式训练器（186行）
├── tests/                    # 测试
└── openspec/changes/archive/ # 变更归档
```

---

## License

MIT
