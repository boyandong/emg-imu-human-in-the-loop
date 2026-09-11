# OSC v2 协议

默认输出地址：

```text
/emgimu/state/v2
```

参数顺序固定为：

```text
timestamp_ms
direction_id
gesture_id
arm_phase_id
hand_phase_id
activation
motion_intensity
consistency_id
consistency_score
q_direction
q_gesture
q_arm_phase
q_hand_phase
emg_quality
imu_quality
quality_flags
```

枚举：

```text
Direction: -1 Unknown, 0 None, 1 Forward, 2 Backward, 3 Left, 4 Right, 5 Up, 6 Down
Gesture:   -1 Unknown, 0 Neutral, 1 Pinch, 2 Fist, 3 Open
Phase:     -1 Unknown, 0 Idle, 1 Onset, 2 Active, 3 Hold, 4 Release, 5 Transition
Consistency: -1 Unknown, 0 Normal, 1 Atypical
```

`consistency_score=-1` 表示条件样本不足，不能评价；其他值越接近 1 表示越偏离
训练数据中的同条件分布。第一版不得用 C 覆盖 D/H。

兼容地址 `/emgimu/state` 仍发送：

```text
timestamp_ms direction_as_legacy_gesture q_direction motion_intensity activation
```

## 带质量信息的实时输入

旧 `/emgimu/raw` 保持不变。采集端能够提供质量信息时使用
`/emgimu/raw/v2`，共31项：

```text
timestamp_ms
emg[8]
accel[3]
gyro[3]
sample_quality
missing
timestamp_valid
imu_valid
interpolated_imu
quality_gate_pass
duplicate_packet
out_of_order_packet
channel_quality[8]
```

质量闸门失败、时间戳无效或有效EMG通道少于6个时不会强制猜测；对应输出进入
Unknown，并在 `quality.flags` 中保留原因。旧输入没有这些信息时维持兼容行为。

`quality_flags` 新增分类器原因位：`1<<10` 为模型域漂移，`1<<11` 为神经融合头与
单模态辅助头高置信冲突，`1<<12` 为模型置信不足，`1<<13` 为已执行但未通过的
分类校准可靠性检查。它们复用现有整数位图，不改变 OSC v2 参数数量和顺序。
概率间隔与漂移分数记录在运行 CSV/`HumanState.to_dict()`，不追加到固定 OSC v2
参数表。

实时服务会为每次启动建立独立审计目录。`manifest.json` 记录本次模型、校准和
一致性模型哈希；`health.json` 汇总解析/处理错误、Unknown 比例、质量标志计数、
输出速率和服务内处理延迟；`states.csv` 的 `run_id`、`state_sequence` 与 UTC
记录时间用于恢复状态顺序。健康文件采用原子替换，CSV 默认每25个输出状态做一次
持久化同步，并在正常或异常关闭时最终同步。

旧协议无法表达手势、双阶段、一致性和信号质量，仅用于迁移期间维持旧前端。
