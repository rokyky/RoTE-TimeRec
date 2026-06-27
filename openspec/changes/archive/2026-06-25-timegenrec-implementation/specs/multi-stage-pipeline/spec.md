## 新增需求
### 需求: PipelineRunner 顺序执行
runner 按 recall -> pre_rank -> rank -> re_rank 执行。
#### 场景: 全链路
- 当 调用 run()
- 那么 候选经所有阶段返回最终 CandidateList

### 需求: 多路召回
多个召回结果按(user,item)去重合并。
#### 场景: pop + itemcf
- 当 配置两个召回方法
- 那么 候选合并去重

### 需求: 阶段跳过
pre_rank/re_rank 可通过配置禁用。
#### 场景: 跳过粗排
- 当 配置省略 pre_rank
- 那么 recall 直达 rank