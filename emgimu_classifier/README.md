# EMG–IMU Human State

这是一个独立于采集前端的实时分类后端。它接收共享源时间轴上的
`8 路 EMG + 3 路加速度 + 3 路角速度`（默认 200 Hz），每 40 ms
输出一次方向、手部姿态、两条动作阶段、激活强度、运动强度、一致性、
分项置信度和信号质量。

系统把方向和手势作为两个标签，而不是把 `Left+Fist` 扩成一个组合类：

```text
Direction = Unknown / None / Forward / Backward / Left / Right / Up / Down
Hand      = Unknown / Neutral / Pinch / Fist / Open
```

新训练的经典模型还会建立按类别的稳健特征域守卫：概率不足或输入整体偏离训练域时
输出 Unknown，并记录概率间隔和漂移分数。神经模型在融合头与单模态辅助头高置信
冲突时同样拒识。Open 使用独立时序记忆，短暂 EMG 衰减不会立即退回 Neutral。

## 数据约定

每个 `.npz` trial 至少包含：

```text
timestamp_ms   [samples]
emg            [samples, 8]
accel          [samples, 3]
gyro           [samples, 3]
direction      [samples] 或标量
gesture        [samples] 或标量
session_id     标量字符串
trial_id       标量字符串
```

方向表示当前运动方向，不表示手臂所在位置。动作回程若未单独标注，必须设为
`Direction.UNKNOWN`，不能沿用去程标签。`Neutral` 是自然放松；`Open` 是主动
伸展后保持的张开姿态。

固定数据划分为 Session 1–2 训练、Session 3 验证、Session 4 最终测试。
工具会拒绝 trial 同时出现在多个分区中。

## 快速使用

```powershell
pip install -e .[dev]
emgimu validate-dataset path\to\dataset --formal
emgimu train-baseline path\to\dataset --output artifacts\baseline.pkl
emgimu evaluate path\to\dataset --model artifacts\baseline.pkl --split validation
emgimu evaluate path\to\dataset --model artifacts\baseline.pkl --split validation `
  --output artifacts\validation_evidence.json
emgimu deployment-check path\to\dataset --model artifacts\baseline.pkl `
  --validation-report artifacts\validation_evidence.json `
  --output artifacts\deployment_readiness.json
emgimu freeze-model path\to\dataset --model artifacts\baseline.pkl `
  --validation-report artifacts\validation_evidence.json `
  --output artifacts\baseline_formal.pkl
emgimu package-deployment --model artifacts\baseline_formal.pkl `
  --freeze-receipt artifacts\baseline_formal.pkl.freeze.json `
  --calibration calibration\session_3.json --output artifacts\deployment_v1
emgimu verify-deployment artifacts\deployment_v1\deployment.json
emgimu serve-deployment artifacts\deployment_v1\deployment.json
emgimu train-neural path\to\dataset --output artifacts\dual_branch.pt
```

若一次佩戴的有标签引导校准已经提取为特征 NPZ，可生成独立、可回滚的 SVM
会话适配产物。输入必须包含 `emg_features[N,48]`、`imu_features[N,36]`、
`direction[N]` 和 `gesture[N]`，并覆盖模型中的全部类别：

```powershell
emgimu adapt-baseline --model artifacts\baseline.pkl `
  --input calibration\session_features.npz --output artifacts\baseline_session.pkl
```

适配只拟合有界的校准后输出 bias，不修改 SVM、标准化器或基础模型文件；缺类、
输出路径等于原模型或目标已经存在时会拒绝执行。

正式 SVM 与 TCN 训练不再让重叠窗口较多的 trial 自动获得更多票数。训练权重按
“session 等权 → session 内 trial 等权 → trial 内窗口等分”计算；Session 3 的温度
校准、拒识阈值和汇总指标也沿用相同口径。混淆矩阵保留原始窗口整数计数。类别权重
仍用于头部类别不平衡。LDA 只是不加权的最低诊断基线，不参与正式模型冻结判断。

程序不会自动下载外部生理数据。`emgimu.external` 保存设备、许可证和允许的
适配策略；带 `NC` 的数据在商业模式下会被拒绝。

UniBo-INAIL 公开跨日数据使用独立 benchmark 层，不影响自采正式数据：

```powershell
emgimu benchmark-adapt unibo-inail PATH_TO_UNIBO_DATA --output PATH_TO_BENCHMARK
emgimu benchmark-check PATH_TO_BENCHMARK
emgimu benchmark-report PATH_TO_BENCHMARK
```

新版自采 HDF5 3.0 先检查，再转换到分类器内部的 200 Hz trial。转换保留
250 Hz EMG 与 112 Hz IMU 原始时间轴、packet audit、质量闸门和佩戴元数据：

```powershell
emgimu hdf5-v3-check path\to\session.h5
emgimu hdf5-v3-adapt path\to\session_root --output path\to\formal_dataset
emgimu validate-dataset path\to\formal_dataset
```

适配器使用 UI cue 加保守边界生成训练标签；cue 不是生理 onset。正式训练仍需
每次佩戴独立的约 60 秒引导校准，适配器不会使用 Session 4 测试 trial 偷拟合校准。

## 实时接口

`HumanStateEstimator.push_sample(...)` 接收一个源时间戳采样；累积到 200 ms
窗口后，以 25 Hz 返回 `HumanState`。新 OSC 地址为 `/emgimu/state/v2`。
旧 `/emgimu/state` 可由 `OscPublisher(publish_legacy=True)` 同时发送。

已有采集端只需向 `/emgimu/raw` 发送 `timestamp + 8 EMG + 3 accel + 3 gyro`：

```powershell
emgimu serve --model artifacts\baseline.pkl --calibration calibration\session_3.json `
  --publish-legacy
```

实时服务默认在 `artifacts/runtime/<UTC时间-随机ID>/` 创建互不覆盖的运行审计包：
`manifest.json` 固定模型、校准、可选一致性模型的 SHA-256 和网络配置，
`health.json` 原子更新输入/输出计数、Unknown 比例、质量原因、错误和处理延迟
P50/P95，`states.csv` 保存逐状态记录并定期执行持久化同步。显式 `--run-id`
不得复用已有默认审计目录；需要自定义位置时可使用 `--audit-dir`、
`--log-csv`、`--health-json` 和 `--run-manifest`。
实时窗口中的非有限传感器值会被安全替换后再送入数值计算，同时仍按缺失/坏通道
降低质量并使受影响的 D 或 H 头输出 Unknown；坏包不应直接终止现场服务。
新训练产物标记为 `development_unvalidated`；没有声明状态的旧模型在manifest中标记为
`legacy_unspecified`。只有未来完成正式冻结流程的产物才允许出现
`formal_claim_allowed=true`，服务能运行不等于模型已经通过正式验证。
`deployment-check` 会重新计算完整trial/校准数据指纹，检查四Session、28组合、
onset条件、四份校准、完整7/4类别、验证报告模型哈希及v1指标门槛。旧参考数据会
明确返回 `blocked`，该检查通过也只代表“可以进入冻结步骤”，不会自行宣布模型正式。
`freeze-model` 会再次运行同一组门禁而不是信任一份可编辑的readiness JSON；仅在
全部通过后另存 `formal_frozen` 模型并生成 `.freeze.json` 收据。收据绑定源模型、
冻结模型、数据集和验证报告四组SHA-256，源模型与已有输出均不会被覆盖。
`package-deployment` 只接受与冻结收据匹配的 `formal_frozen` 模型，并把模型、
校准、验证证据及可选C模型复制到不可覆盖的新目录；`verify-deployment` 重新检查每个
成员哈希、模型状态、冻结收据、验证报告权重口径以及模型—校准—C模型的加载契约。
模型元数据、收据与报告三处必须对同一源模型/数据集/报告哈希达成一致，单独重写清单
或收据不能替换证据。输入/输出地址、端口和旧协议发布开关也固定在部署清单内。正式
现场使用 `serve-deployment`：它先完整复验部署包，
再仅从清单加载这些固定配置，并把部署清单 SHA-256 和 `formal_bundle` 启动模式写入
本次运行审计。普通 `serve --model ... --calibration ...` 保留用于开发调试，审计中
标记为 `development_loose_files`，不应作为正式演示入口。部署包目录中的任一成员被
替换后都不能通过验证。

观察数据足够后可拟合仅展示、不干预 D/H 的一致性模型：

```powershell
emgimu fit-consistency artifacts\runtime\RUN_ID\states.csv --output artifacts\consistency.json
```

最终验收不能用模拟数据替代：冻结模型后，只运行一次未查看过的 Session 4，
并把 Unknown 计作错误，检查 D/H 宏 F1、联合准确率和端到端 P95 延迟。
首次 `--unlock-test` 会写入 `SESSION4_FINAL_RESULT.json`；此后程序拒绝再次测试，
文件同时保存模型 SHA-256，避免测试后偷偷换模型。

评估报告包含每类别覆盖率和 risk–coverage 曲线。模型文件加载时记录 SHA-256；
新模型声明其采样率与校准版本/尺度模式，不兼容时实时服务拒绝启动。

在接入真实采集程序前，可用 `emgimu make-smoke-dataset TEMP_PATH` 生成明确标注为
非生理数据的管线测试集。完整字段、正式采集和协议说明见 `docs/`。

建议依次阅读：`docs/DATA_AND_EXPERIMENT.md`、`docs/OSC_V2.md`、
`docs/ARCHITECTURE_AND_LIMITS.md`。外部数据边界见 `docs/EXTERNAL_DATASETS.md`。
