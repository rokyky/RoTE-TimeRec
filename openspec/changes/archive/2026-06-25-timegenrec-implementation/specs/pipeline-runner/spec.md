## 新增需求

### 需求: PipelineRunner 按顺序执行阶段
系统必须按 recall -> pre_rank -> rank -> re_rank 顺序执行各阶段，在阶段间传递 CandidateList。

#### 场景: 全链路运行
- **当** PipelineRunner.run() 被调用时
- **那么** 候选数据经过所有阶段，返回最终的 CandidateList

### 需求: 支持阶段跳过
系统必须允许通过配置禁用可选阶段（如 pre_rank 或 re_rank）。DSSMRecall 默认不启用。

#### 场景: 跳过粗排阶段
- **当** 配置中省略 pre_rank 时
- **那么** 候选数据直接从 recall 进入 rank

### 需求: 多路召回合并
系统必须将多个启用召回方法的结果按 user-item 对去重合并。

#### 场景: 多路合并去重
- **当** 配置了 pop + itemcf 两个召回方法时
- **那么** 所有候选按 user-item 对去重后合并为一个 CandidateList