# UniBo-INAIL baseline 工作区

这个目录集中保存 UniBo-INAIL 的来源记录、官方原始数据、本项目的标准化输出，
以及后续 baseline 实现约束。可复用的数据适配器仍保留在
`src/emgimu/datasets/adapters/unibo_inail.py`，避免复制两份实现导致结果漂移。

## 当前状态

- 官方来源：<https://github.com/pulp-bio/unibo-inail-semg-dataset>
- 下载日期：2026-09-12（Asia/Shanghai）
- 原始压缩包：`unibo-official-main.zip`，513,385,791 bytes
- 压缩包 SHA-256：`E32F1649B7B9B775A6B03328FB862EBF21A18C7CA6B4D9ABB50B7281CAEFBCE3`
- 官方 MAT 文件：224/224
- 转换结果：完整性检查通过，11,291 个编号试次，6,774 个四分类可用试次
- 覆盖范围：7 名受试者、8 天、4 种静态手臂姿态、4 通道 EMG
- 已完成 baseline：RBF-SVM 测试 macro-F1 0.6921，四通道 TCN 0.6729；详见
  `RESULTS_20260912.md`

## 目录

```text
baseline/unibo/
├── SOURCE_MANIFEST.json      # 下载来源、校验值和本次准备结果
├── BASELINE_SPEC.md          # 交给 baseline 编写者的实验与输出约束
├── verify.ps1                # 一键复查标准化数据和适配器测试
├── unibo-official-main.zip   # 官方压缩包，仅本地，不提交 Git
├── official/                 # 官方仓库内容，仅本地，不提交 Git
│   └── data/                 # 224 个原始 MAT 文件
└── benchmark/                # 本项目的标准化数据
    ├── manifest.json         # 每个源文件的哈希与数据集元信息
    ├── label_map.json        # 标签映射
    ├── splits.json           # Day 1-5/6/7-8 固定划分
    ├── reports/              # 能力边界、统计与完整性报告
    └── trials/               # 11,291 个 NPZ，仅本地，不提交 Git
```

## 重要实验边界

UniBo 只能验证手部状态 H 的 EMG baseline，不能验证方向 D、A/M/C、动作阶段、
EMG-IMU 融合或 8 通道环形腕带输入。不能通过复制、补零或插值把 4 个命名肌肉
通道伪装成项目的 8 通道环形输入。

主四分类只使用 Rest/Neutral、Power grip/Fist、2-finger pinch/Pinch、Open hand。
3-finger pinch 和 Pointing index 被保留用于审计，但不进入主任务。

固定主实验按 subject/day 分组：Day 1-5 训练，Day 6 验证，Day 7-8 测试。
不得把相邻或重叠窗口随机分散到不同集合。

## 已发现并处理的官方数据边界

每个官方文件都有 `gestureCounter=0` 的非编号区域。全量数据中这些区域共
445,135 个原始样本，其中 2,779 个样本仍残留主动动作标签。适配器不会猜测其
真实归属，也不会把它们改写成静息；而是全部排除，并把数量写入
`benchmark/reports/statistics.json`。编号重复本身不受影响。

## 复查

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File baseline/unibo/verify.ps1
```

如果 Python 不在 PATH：

```powershell
powershell -ExecutionPolicy Bypass -File baseline/unibo/verify.ps1 -PythonExe D:\python\python.exe
```

baseline 的实现与验收要求见 `BASELINE_SPEC.md`。下一阶段的人体表征特征分组、
消融矩阵、评价口径和 test 使用纪律见 `EXPERIMENT_PLAN.md`。

## 运行 RBF-SVM baseline

以下命令会在 Day 6 选择模型和置信度参数，随后冻结模型，再打开 Day 7–8 测试集：

```powershell
$env:PYTHONPATH = (Resolve-Path src)
python -m emgimu.datasets.unibo_baseline baseline/unibo/benchmark `
  --output-root baseline/unibo --run-id svm_rbf_v1 --evaluate-test
```

实现使用原生 4 通道数据，不复用固定 8 通道的生产输入头。运行产物写入
`models/<run_id>` 和 `results/<run_id>`，已有 run id 永不覆盖。

小型四通道因果 TCN 使用同一数据单元、窗口、权重和划分：

```powershell
$env:PYTHONPATH = (Resolve-Path src)
python -m emgimu.datasets.unibo_neural_baseline baseline/unibo/benchmark `
  --output-root baseline/unibo --run-id tcn_v1 --evaluate-test
```

`--device auto` 会在 CUDA PyTorch 可用时使用 GPU，否则使用 CPU。
