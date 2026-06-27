## 新增需求

### 需求: 模型接受完整输入（含 time_deltas 和 same_cat_mask）
系统必须接受 (seqs, positions, time_deltas, same_cat_mask) 输入，正确构造 pairwise category mask。

#### 场景: TiSASRec-Cat 完整前向
- **当** 传入 (B,L) seqs、(B,L) positions、(B,L,L) time_deltas、(B,L,L) same_cat_mask 时
- **那么** 返回 (B, num_items) 得分张量，无 NaN

### 需求: Rank 阶段 batch scoring
Rank 阶段必须对候选物品使用模型 batch scoring，而非逐用户 for 循环。

#### 场景: 候选 batch 打分
- **当** Rank.predict() 收到 CandidateList 时
- **那么** 每个候选获得一个来自模型 batch forward 的得分

### 需求: 冷启动用户 fallback
用户无序列历史时，Rank 阶段必须保留 recall 阶段的得分，不调用模型。

#### 场景: 冷用户保留 recall 分
- **当** 用户序列为空时
- **那么** 候选的 score 保持 recall 阶段的值不变