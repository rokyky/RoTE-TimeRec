## 新增需求

### 需求: 脚本下载和预处理 Amazon 数据集
系统必须提供脚本，下载 Amazon Beauty/Sports 数据集，构建用户交互序列，保留叶子类目，输出 train/val/test 切分。

#### 场景: Amazon 数据处理
- **当** 执行 prepare_data.py --dataset amazon_beauty 时
- **那么** 生成包含用户-物品序列和叶子类目映射的 train/val/test 文件

### 需求: MovieLens 作为 smoke test
系统必须支持 MovieLens-1M 预处理，但不要求叶子类目信息。MovieLens 仅用于 pipeline 可运行性检查。

#### 场景: MovieLens smoke test
- **当** 执行 prepare_data.py --dataset ml-1m 时
- **那么** 输出可加载的序列数据，但 category 字段标记为 None