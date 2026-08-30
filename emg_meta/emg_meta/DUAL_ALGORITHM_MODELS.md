# 双算法训练与在线推理

上位机支持两个互斥运行的算法：

- `meta_conv_lstm_v1`：Meta Conv1D+LSTM，远端目录 `/home/qxy/qxy/generic-neuromotor-interface`。
- `personal_mpf_tds_v1`：依据论文描述独立复现的单参与者 MPF+TDS，远端目录 `/home/qxy/qxy/emgforce-mpf-tds`。

训练页先选择算法，界面会自动切换仓库和 tmux 会话。启动脚本会检查另外两个已知训练会话，阻止两个 GPU 训练任务同时运行。

实时识别页会在每个模型前显示 `[Conv-LSTM]` 或 `[MPF+TDS]`。切换模型时会先停止旧推理线程、清空流缓冲和事件状态，再加载新后端。新模型不继承旧模型的校准，必须重新校准后才能开始识别。

模型包 v2 使用 `algorithm_id` 和 `runtime_backend` 选择运行时；没有这些字段的 v1 模型自动按 `meta_conv_lstm_v1` 加载，因此已有 CKPT 模型包无需迁移。

两个算法共用固定的数据划分：S01–S10 为训练集，S011–S012 为验证集，S013–S015 为测试集。论文修正版 MPF+TDS 使用固定 `0.35` 阈值、去抖、状态机和 Needleman–Wunsch 序列匹配，以最低验证集平均逐类别 FNR 选择 checkpoint。

MPF+TDS 的 TDS 部分不是 Meta 官方开源代码。MPF 与 Meta 已公开实现进行逐元素数值等价测试；TDS 严格保留论文公开的 192 维输入、256 维投影、六尺度级联、每尺度两个 TDS block 和三层末端全连接结构。论文没有披露的卷积核和 dropout 等细节仍属于明确记录的复现参数。
