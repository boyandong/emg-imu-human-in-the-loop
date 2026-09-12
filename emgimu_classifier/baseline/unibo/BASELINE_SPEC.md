# UniBo H 四分类 baseline 实现规格

本文档定义 UniBo baseline 的实现与验收约束。目标是验证本项目手部状态 H 的
跨日、跨姿态能力，不是复刻完整 D+H HumanState 系统。

## 数据合同

只读取 `baseline/unibo/benchmark`：

- `manifest.json`：数据集和源文件审计信息；
- `splits.json`：唯一主划分，Day 1-5 train、Day 6 validation、Day 7-8 test；
- `trials/**/*.npz`：每个文件是一个有边界的编号试次；
- `benchmark_eligible=true` 的试次才进入主四分类；
- 训练标签只使用 `stable_mask=true` 的 `hand_label`；
- 四类整数标签来自 `label_map.json`，不要重新猜测标签语义。

禁止把同一 trial 的窗口拆到不同集合。模型选择、阈值、温度校准和超参数只能
使用 train/validation，test 只能在方案冻结后评估一次。

## 第一版必须实现的两个模型

1. RBF-SVM：使用当前项目 `src/emgimu/signal.py` 的通道无关时域特征思想；
   标准化器只在训练集拟合；类权重使用 balanced；超参数在 validation 选择。
2. 简单神经基线：小型 1D TCN 或同等容量模型，直接接受原生 4 通道窗口；不得
   调用项目固定 8 通道的生产 TCN 输入头，也不得伪造通道。

建议以 SVM 为必须完成项，TCN 为第二项。所有随机过程必须设置种子。

## 窗口与聚合

- 窗口长度、步长必须配置化，并在结果文件中记录；
- 窗口不得跨 trial；
- 训练时避免长 trial 或高窗口数 trial 获得不成比例权重，可使用 trial-balanced
  sample weight 或每 trial 等量采样；
- 同时报告 window-level 和 trial-level 结果，trial 聚合规则必须预先固定；
- 不得把 3-finger pinch、Pointing index 的尾部静息样本混入 Neutral。

## 必须报告

- accuracy、macro-F1、每类 precision/recall/F1、混淆矩阵；
- 按 subject、day、posture 分层的 macro-F1；
- validation 上拟合后的 ECE；
- 最大概率拒识下的 coverage 与 risk-coverage 曲线/表；
- 训练、验证、测试窗口数和 trial 数；
- 运行配置、随机种子、依赖版本、模型文件 SHA-256；
- 未映射样本和数据质量警告不得静默丢弃。

## 输出位置

运行产物只写入：

```text
baseline/unibo/models/<run_id>/
baseline/unibo/results/<run_id>/
```

至少生成 `config.json`、`metrics.json`、`predictions.csv`、`confusion_matrix.csv`、
`risk_coverage.csv`、`run_manifest.json`。大模型和运行结果默认不提交 Git。

## 验收条件

- `benchmark-check` 仍为 `ok`；
- 固定 split 无 subject/day 泄漏；
- 代码能从空的 models/results 目录重复运行；
- 结果不能被表述为生产模型性能，也不能证明 D、融合或 8 通道能力；
- SVM 与 TCN 必须在完全相同的数据单元、标签和划分上比较。
