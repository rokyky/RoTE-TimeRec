## 1. 核心模型
- [x] 1.1 SASRec：self-attention + FFN + mask
- [x] 1.2 TiSASRec：时间差离散化 + 时间桶偏置
- [x] 1.3 TiSASRec-Cat：类目条件时间桶偏置
- [x] 1.4 单元测试：forward shape、mask、loss

## 2. 训练器
- [x] 2.1 train_epoch()：loader、loss、backward、梯度裁剪
- [x] 2.2 train()：多 epoch、early stopping、checkpoint、LR
- [x] 2.3 metric logging：epoch、loss、metric 打印
- [x] 2.4 trainer 单元测试

## 3. 数据与配置
- [x] 3.1 SeqRecDataset：序列、padding、负采样
- [x] 3.2 EvalDataset：留最后一项评估
- [x] 3.3 configs/default.yaml
- [x] 3.4 load_config()

## 4. 评估
- [x] 4.1 recall/ndcg/mrr 函数
- [x] 4.2 evaluate_full_sort()
- [x] 4.3 model_eval()
- [x] 4.4 evaluate_by_bucket()
- [x] 4.5 popularity 分桶

## 5. Pipeline
- [x] 5.1 base.py：Candidate/CandidateList/PipelineStage
- [x] 5.2 PopularityRecall
- [x] 5.3 ItemCFRecall
- [x] 5.4 SimplePreRank
- [x] 5.5 SequenceRanker
- [x] 5.6 MMRReRank
- [x] 5.7 PipelineRunner
- [x] 5.8 多路召回合并
- [x] 5.9 合成数据端到端测试

## 6. 脚本
- [x] 6.1 scripts/train_model.py
- [x] 6.2 scripts/run_pipeline.py
- [x] 6.3 README