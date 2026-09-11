# 外部数据使用边界

外部数据只用于预训练、特征验证和鲁棒性实验。最终验收只使用自采 Session 4。

| 数据集 | 第一版用途 | 原始输入策略 |
|---|---|---|
| UniBo-INAIL | H 跨日/跨姿态 benchmark | 保持原生 4 通道，独立 benchmark adapter |
| EMG-IMU-EPN-100+ Myo | 分别预训练 D/H | 8 通道 200 Hz 环形输入 |
| EMG-EPN-612 | 预训练 H | 8 通道 200 Hz环形输入 |
| Electrode Shift | 腕带旋转鲁棒性 | 循环通道增强/校正 |
| NinaPro DB5 | 独立迁移实验 | 两条 Myo 分成两个 8 通道设备域 |
| Meta GNI | 时序结构参考 | 不进入共同原始输入头 |
| 3DC/Fougner/putEMG/GRABMyo/HD-sEMG | 特征基准 | 独立 source adapter |

规则：

1. 下载时重新确认许可证并记录版本、URL、哈希、设备、单位、采样率、硬件滤波、
   电极位置和官方 train/test 划分。
2. 商业模式拒绝所有 NC 数据及其衍生权重。
3. 高采样率 EMG 用抗混叠 FIR 多相重采样降到 200 Hz；不声称保留 100 Hz 以上信息。
4. 低采样率信号不通过上采样进入原始 EMG 预训练。
5. 只有等价环形 8 通道布局允许循环重排；16/24/128 通道不得插值冒充 8 通道。
6. 外部单标签动作不能伪造成 `Direction × Gesture` 复合标签。

代码中的来源登记、标签白名单和策略执行位于 `emgimu.external`。

## UniBo-INAIL 标准化

UniBo 数据不进入自采 Session 1–4 数据加载器。转换、检查和统计分别运行：

```powershell
emgimu benchmark-adapt unibo-inail PATH_TO_OFFICIAL_DATA --output PATH_TO_OUTPUT
emgimu benchmark-check PATH_TO_OUTPUT
emgimu benchmark-report PATH_TO_OUTPUT
```

输出保留 4 个命名肌肉通道且不生成 `imu` 字段。主 H benchmark 只映射
Rest、Power grip、2-finger pinch、Open hand；3-finger pinch 和 Pointing index
保留但不参与四分类。固定分区为 Day 1–5 train、Day 6 validation、Day 7–8 test。
