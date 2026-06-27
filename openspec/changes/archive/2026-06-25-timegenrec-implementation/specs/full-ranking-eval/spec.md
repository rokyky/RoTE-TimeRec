## 新增需求
### 需求: full-ranking
对所有 item batch scoring，计算 recall/ndcg/mrr。
#### 场景: 模型评估
- 当 传入模型和 loader
- 那么 返回 recall@k 和 ndcg@k

### 需求: 分桶评估
按 popularity/entropy/length 分桶用户，分别计算指标。
#### 场景: 长尾分析
- 当 传入 bucket_info
- 那么 返回每桶 recall/ndcg