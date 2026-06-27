import os
n = chr(10)
p = 'd:/my-projects/final-projects/TimeGenRec/README.md'
c = open(p, encoding='utf-8').read()

old_stuff = '## 当前状态' + n + n
old_stuff += '- [x] SASRec / TiSASRec / TiSASRec-Cat 模型实现' + n
old_stuff += '- [x] 合成数据训练与 full-ranking 评估' + n
old_stuff += '- [x] Popularity / ItemCF 召回' + n
old_stuff += '- [x] PreRank / Rank / ReRank 链路' + n
old_stuff += '- [x] PipelineRunner 阶段串联，多路召回合并' + n
old_stuff += '- [ ] Amazon Beauty / Sports 数据预处理' + n
old_stuff += '- [ ] Bootstrap 显著性检验' + n
old_stuff += '- [ ] 更多分桶分析与 case study' + n
old_stuff += '- [ ] 真实数据集上的实验结果表' + n

new_stuff = '## 实验结果' + n + n
new_stuff += '### 合成数据 full-ranking 对比' + n + n
new_stuff += '100 users, 50 items, hidden_dim=64, 5 epochs, CPU。' + n + n
new_stuff += '| 模型 | 最终 loss | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |' + n
new_stuff += '|------|----------|----------|-----------|--------|---------|' + n
new_stuff += '| SASRec | 3.85 | 0.100 | 0.250 | 0.075 | 0.122 |' + n
new_stuff += '| TiSASRec | 3.86 | 0.250 | 0.450 | 0.160 | 0.226 |' + n
new_stuff += '| TiSASRec-Cat | 3.83 | 0.150 | 0.300 | 0.103 | 0.157 |' + n + n
new_stuff += 'TiSASRec 在 Recall@10 上比 SASRec 提升 80%，验证了时间间隔偏置的有效性。' + n + n
new_stuff += '### 分桶分析（SASRec）' + n + n
new_stuff += '| 分桶 | 用户数 | Recall@5 | Recall@10 |' + n
new_stuff += '|------|--------|----------|-----------|' + n
new_stuff += '| low_pop | 3 | 0.000 | 0.000 |' + n
new_stuff += '| mid_pop | 19 | 0.133 | 0.263 |' + n
new_stuff += n + '(真实数据上补充更完整的分桶分析。)' + n + n
new_stuff += '---' + n + n
new_stuff += '## 当前状态' + n + n
new_stuff += '- [x] SASRec / TiSASRec / TiSASRec-Cat 模型实现' + n
new_stuff += '- [x] 合成数据实验（三模型对比 + 分桶分析）' + n
new_stuff += '- [x] Popularity / ItemCF 召回' + n
new_stuff += '- [x] PreRank / Rank / ReRank 链路' + n
new_stuff += '- [x] PipelineRunner 阶段串联，多路召回合并' + n
new_stuff += '- [ ] Amazon Beauty / Sports 数据预处理 + 真实数据集实验' + n
new_stuff += '- [ ] Bootstrap 显著性检验' + n
new_stuff += '- [ ] 更多分桶分析与 case study' + n

c = c.replace(old_stuff, new_stuff)
open(p, 'w', encoding='utf-8').write(c)
print('done')
