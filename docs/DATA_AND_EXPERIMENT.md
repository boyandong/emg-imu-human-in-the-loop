# 数据与实验协议

## 标签定义

`Direction` 描述当前运动方向，不描述手臂位置。`NONE` 表示确实没有方向；
`UNKNOWN` 表示本应有答案但数据无效、处于未标注回程或无法可靠判断。

`Gesture` 描述当前手部姿态。`NEUTRAL` 是自然放松；`OPEN` 是主动伸展后的
张开姿态。张开后 EMG burst 下降不自动把姿态改回 Neutral，释放过程由
`P_hand=RELEASE` 表示。

## 正式采集

- 四个 session，至少跨两天；每次摘下并重新佩戴。
- 每个 session 采集全部 28 个 `Direction × Gesture` 组合，每个组合 3 次。
- 复合动作三次分别采用手势相对手臂运动 `-200/0/+200 ms` 的提示偏移。
- trial 顺序随机；去程是正式标签区，未标注回程设为 Unknown。
- 程序记录 cue 时间；稳定区间使用 `stable_mask=true`，边界、回程和操作者
  判为不合格的片段设为 false。
- 每个 session 前单独录制约 60 秒校准块：静息、舒适强度的 Pinch/Fist/Open、
  六方向中性手势移动。无需要求最大自主收缩。

数据固定分区：Session 1–2 train，Session 3 validation，Session 4 test。
Session 4 只允许在冻结全部算法后以 `--unlock-test` 打开一次。
首次评估会在数据根目录留下带模型哈希的 `SESSION4_FINAL_RESULT.json` 锁文件。

## Trial NPZ

| 字段 | 形状 | 含义 |
|---|---:|---|
| `timestamp_ms` | `[N]` | 严格递增的设备源时间戳 |
| `emg` | `[N,8]` | 原始同步 EMG |
| `accel` | `[N,3]` | 原始加速度 |
| `gyro` | `[N,3]` | 原始角速度 |
| `direction` | `[N]` 或标量 | Direction 数值 |
| `gesture` | `[N]` 或标量 | Gesture 数值 |
| `stable_mask` | `[N]` | 可用于稳定分类训练的采样 |
| `session_id` | 标量字符串 | `1`–`4` |
| `trial_id` | 标量字符串 | 全局唯一 trial ID |
| `session_date` | 标量字符串 | 本地日期 `YYYY-MM-DD`，正式审计必需 |
| `window_quality` | `[N]` | 由传输、通道和IMU可用性生成的0–1质量 |
| `missing_mask` | `[N]` | 丢包或无可靠IMU覆盖的内部帧 |
| `channel_quality` | `[8]` | 本次佩戴的逐通道质量，0–1 |
| `timestamp_valid` | `[N]` | 时间轴有效性 |
| `imu_valid` | `[N]` | 对齐后的IMU是否有可靠邻近观测 |
| `interpolated_imu` | `[N]` | 该内部帧的IMU是否来自时间插值 |
| `quality_gate_pass` | 标量 | 采集前质量闸门是否通过 |

旧 NPZ 没有这些字段时按“质量未知但兼容旧流程”读取；HDF5 3.0正式数据必须
显式生成。训练窗口同时要求稳定比例、平均质量、缺失比例、有效通道数、时间戳
和质量闸门全部合格。

通过质量筛选后，窗口并非等票训练。权重依次平衡 session 和 trial，再在 trial 内
由重叠窗口均分；因此较长 trial 或较少被筛掉窗口的 trial 不会自动主导 SVM/TCN。
Session 3 的温度校准和拒识阈值沿用同一权重口径，但最终报告仍需同时保留逐类与
逐 trial 结果，不能只看窗口级总 accuracy。

## HDF5 3.0入口

分类核心仍工作在统一200 Hz内部时间轴。适配器以HDF5中的两条
`sample_time_ns` 为准，将250 Hz EMG抗混叠降采样，并将112 Hz IMU按时间戳
对齐；不会假设两个流的sample index一一对应。原始双时间轴保存在每个trial中，
用于审计。

28组合标签被拆成独立 `direction` 和 `gesture`。提示前、onset边界、无法确认的
保持/回程均使用 Unknown/Transition；只将保守交集写入 `stable_mask`。HDF5中的
cue是界面提示时间，不解释为真实生理起始。

运行正式审计：

```powershell
emgimu validate-dataset DATASET_ROOT --formal
```

## 校准文件

`DATASET_ROOT/calibration/session_N.json` 保存滤波选择、EMG 尺度模式、逐通道
诊断尺度、全局稳健尺度、重力、陀螺仪偏置、设备到人体坐标旋转、腕带循环偏移、
8 个 shift 候选误差、shift margin、人体旋转残差和运动尺度。新校准默认用全局
尺度保留通道间相对强弱；缺少版本字段的旧 JSON 按 v1 逐通道尺度恢复。Session 4
校准不更新分类器权重。

校准文件同时区分 `emg_channel_shift_evaluated` 与 `body_rotation_fitted`。只有实际
做过相应评估且结果不合格时，运行时才分别拒绝 H 或 D；字段缺失的旧校准不会因
“没有评估”被误判为“评估失败”。

## 指标

最终报告同时给出 D/H macro-F1、联合准确率、Unknown 覆盖率、ECE、混淆矩阵、
逐类别覆盖率、risk–coverage 曲线和端到端 onset-to-stable 延迟 P50/P95。Unknown
在 macro-F1 与联合准确率中按错误计算。没有真实 Session 4 延迟标注时不得宣称
满足 300 ms。
