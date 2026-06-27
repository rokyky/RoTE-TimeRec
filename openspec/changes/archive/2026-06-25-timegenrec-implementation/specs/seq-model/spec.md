## 新增需求
### 需求: SASRec 前向
SASRec.forward(seqs, positions) -> (B, num_items) logits，padding + causal mask。
#### 场景: SASRec 训练
- 当 输入 (B, L) seqs 和 positions
- 那么 输出 (B, num_items) 得分，无 NaN

### 需求: TiSASRec 时间感知
TiSASRec 额外接受 time_deltas(B,L,L)，离散化为时间桶，可学习标量偏置加到 attention。
#### 场景: TiSASRec 时间偏置
- 当 额外传入 time_deltas
- 那么 attention 得分包含时间桶偏置

### 需求: TiSASRec-Cat 类目条件
额外接受 same_cat_mask(B,L,L)，同类/跨类使用独立时间桶偏置。
#### 场景: TiSASRec-Cat
- 当 传入 same_cat_mask
- 那么 同类和跨类配对使用不同时间桶偏置