# EMG–IMU Human-in-the-loop

面向音乐交互场景的 EMG–IMU 人体状态估计项目。系统由数据采集、HumanState 分类和实时音乐反馈三部分组成：

```text
采集端 → HumanState estimation → 前端反馈
```

## 项目结构

- `collection/`：8 通道前臂 sEMG 与 6 轴 IMU 正式四 Session 采集、质量检查和数据审计。
- `emgimu_classifier/`：D/H 双头分类、A/M 连续状态、动作阶段、置信度与拒识机制。
- `muscle_music/`：将方向、肌肉激活和手势状态映射为实时音乐反馈。
- `docs/`：项目书和科研过程复盘。

## 当前研究重点

- 真实采样率和可审计时间轴；
- 手臂方向与手部状态的解耦；
- 跨 Session、重新佩戴和跨日泛化；
- 主动 Open 的 onset–hold–release 时序；
- Human-in-the-loop 中的置信度、Unknown 和端到端延迟。

## 数据说明

真实受试者数据、参考照片、模型文件和运行日志不进入 Git 仓库。正式数据由采集端生成带 SHA-256 的 `SESSION_COLLECTION_READINESS.json`，通过门禁后再进入受控训练流程。

## 项目与贡献

本项目由 Boyan 主导研究问题、交互设计、实验协议和系统整合。采集程序由团队协作开发，并保留原始 Git 历史和提交作者，以准确反映实际工程贡献。详见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

