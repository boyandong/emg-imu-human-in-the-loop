# UniBo-INAIL benchmark 能力边界

## 可以验证

- Neutral / Fist / 2-finger Pinch / Open 的 H 四分类。
- 同一受试者跨日泛化，以及七名受试者上的总体趋势。
- 四种静态手臂姿态下的鲁棒性和跨姿态变化。
- 与通道数无关的时域特征、时序建模、归一化和域适配方法。

## 不能验证

- 手臂运动方向 D、运动阶段、动作提示到稳定输出的端到端延迟。
- D+H 组合状态、A、M、C 或 EMG-IMU 融合；数据集没有 IMU。
- 八通道环形腕带的循环移位鲁棒性；UniBo 是四块命名肌肉电极且不是环形布局。
- 当前八通道原始输入头本身。不得把四通道插值、复制或补零成假八通道。

## 主 benchmark 标签

主任务只使用 Rest、Power grip、2-finger pinch 和 Open hand。3-finger pinch 与
Pointing index 保留在标准化文件中，但整段标记为 benchmark_eligible=false，
不参与四分类，也不贡献 Neutral 尾段。
