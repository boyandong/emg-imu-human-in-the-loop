# Meta 数据统计与本工程采集数量

本目录集中保存制定 `meta_discrete_7_short_v2` 协议时使用的数量统计。它把
“Meta 官方开放数据中实际出现的数量”和“本工程最终采用的采集数量”分开记录，
便于复核和后续程序读取。

## 当前采集方案

- 每位参与者采集 **12 个独立重新佩戴的短 Session**。
- 划分为 8 个训练 Session、2 个验证 Session、2 个最终测试 Session。
- 每个 Session 的 7 个目标动作各做 12 个 Trial，共 84 个目标 Trial。
- 每个 Session 另做响指 3 次、屈指弹出 3 次、自然打字 60 秒。
- 每个 Session 共生成 91 行 Trial 记录，理论时长约 8.92 分钟。
- 12 个 Session 后，每个高层目标动作有 144 个 Trial。
- 食指保持和中指保持各自产生 press、release 两个事件，所以最终 9 个 Meta
  输出事件类别每类都有 144 个事件，共 1,296 个目标事件。

这里的 **Trial** 指界面提示参与者完成一次高层动作；**event** 指导出给 Meta
格式模型的离散事件。一次保持 Trial 会产生按下和松开两个 event，因此二者不能混用。

## 文件

- `meta_official_discrete_counts.csv`：对 Meta 官方开放数据中 100 份离散手势记录
  的逐记录 prompt 数统计摘要。
- `local_protocol_action_counts.csv`：本工程每个动作在单 Session 和 12 个 Session
  中的 Trial/event 数。
- `session_plan.json`：当前协议的机器可读汇总、训练/验证/测试划分及 null 任务数量。
- `sources.md`：论文、官方仓库、统计口径和已知限制。

工程实际运行配置仍以
[`protocols/meta_discrete_7.json`](../protocols/meta_discrete_7.json) 为准；本目录是其
设计依据与统计快照。
