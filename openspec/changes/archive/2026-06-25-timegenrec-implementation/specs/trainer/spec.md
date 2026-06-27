## 新增需求
### 需求: 单 epoch 训练
遍历 loader，计算 loss，反向传播，梯度裁剪，返回平均 loss。
### 需求: early stopping
val metric 连续 patience 轮不提升则停止，恢复最佳权重。
### 需求: checkpoint
metric 提升时保存 model state_dict。
### 需求: LR 调度
ReduceLROnPlateau。