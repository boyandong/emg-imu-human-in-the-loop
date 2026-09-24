# EMG Data Collection

正式采集上位机：连接 8 通道、250 Hz、24-bit EMG + 112 Hz 6-axis IMU
设备，按 JSON Protocol 展示 Prompt，并将连续原始数据、事件与 Trial 索引写入 HDF5。

正式采集协议冻结为 `jilv_music_28_v2`。采集结束后自动生成
`SESSION_COLLECTION_READINESS.json`；只有 `status=passed` 的 Session 才能进入分类流程。
历史 Meta 七动作与 2000 Hz 对齐代码只用于旧数据复现，不属于当前正式采集主线。

## 安装与启动

```powershell
conda env create -f environment.yml
conda activate emgforce
python main.py
```

Windows 也可以直接双击 `run_emgforce.bat`。脚本会自动查找 `emgforce`
Conda 环境；如果环境在其他目录，可先把 `EMGFORCE_PYTHON` 设为该环境的
`python.exe` 完整路径。

界面包含“设备监测 / 实验采集 / 数据检查 / 数据上传 / 训练模型 / 实时识别”六页。可在设备页选择串口，也可用
“连接模拟设备”在没有硬件时完成全流程。

实时识别页有显式“开始诊断记录 / 结束诊断记录”按钮。连接设备、加载模型后点击开始，
本机 `data/live_diagnostics/<UTC时间>-<随机ID>/` 会保存原始八通道 EMG、IMU、
采样索引/接收时间、逐帧各类概率和模型 SHA-256。停止、断开设备、切换模型或退出时
会关闭记录。诊断文件不会自动上传，也没有动作真值标签；请在测试时另行记下动作、
开始/结束时间及电极佩戴情况。页内“标记动作开始/结束”会将人工按键时刻写入
`manual_annotations.csv`，可辅助对齐，但不能替代真实肌肉动作起止标注。只有获得
可信的动作真值后才可以计算实时识别准确率。结束记录时会生成 `analysis.json`，
给出各类峰值预测占比、人工区间内的预测一致率、每段是否曾显示正确、结束时是否正确，
以及从手动开始标记到首次显示正确的名义采样时间，便于排查“总判静息/很少张开”；
还会统计开始标记前 400 毫秒有无活跃判决。手动标记可能晚于或早于实际肌肉动作，
因此这些时间和命中数只是诊断数据，不是生理起始延迟或正式准确率；
生成前会核对三个原始记录文件的 SHA-256、样本数量与索引递增性；
同时给出每通道原始幅值、整秒平线窗口、接近 ADC 上限的比例和设备报告丢包数。
这些信号质量统计覆盖动作与静息，只用于诊断，不宣称正式准确率。

## 新 Session

唯一正式实验协议为 `jilv_music_28_v2`。手臂状态为静止、上、下、左、右、前、后，
手部状态为 Neutral（自然放松）、食指捏合、舒适力度握拳和主动 Open，共 28 个组合。
每个 Session 有 144 个正式 trial：4 个静止组合各 18 次，24 个运动组合各 3 次，
静止与运动数据各占一半。正式 trial 分为 12 个 block，每个 block 内固定包含 6 个静止
和 6 个运动组合后再随机；每个 block 后强制休息 60 秒并记录疲劳/不适。
`-200/0/+200 ms` onset 条件在每个标签内平衡，不再无约束地逐 trial 随机抽取。

1. 连接真实设备或模拟器，在设备页确认 8 通道波形。
   正式音乐体验前，在“音乐力度控制”卡片开始约 17 秒的个人标定：先自然松手约
   4 秒，手指保持松散张开且不要用力撑开；其中前 1 秒只等待信号稳定，不计入基线。
   随后按提示以主观 7/10 力度稳定握拳 3 次，每次约 2.6 秒。手指需要完全合拢，能够稳定重复，
   且不颤抖、不疼痛。该握拳参考映射为 0.85，给演奏时更强的自然发力
   留出约 15% 余量；输出仍限制在 0–1，不要求极限用力。
2. 在实验采集页填写 Participant ID 和 Experiment Name。离开 Participant ID 输入框后，
   Session ID 会严格按已有数据自动更新。每人固定四轮：S01/S02 为 train，S03 为 validation，
   S04 为 final test。S02 缺失时不能创建 S03；S04 必须确认模型、阈值和校准算法已经冻结。
3. 填写结构化佩戴元数据和统一角度参考照片路径；照片 SHA-256 会写入 HDF5。
4. 保持自然放松至少 8 秒并执行质量检查。失败时优先重新检查；人工忽略必须填写原因，
   且质量失败的 Session 不会通过最终分类门禁。
5. 点击开始后先完成约 64 秒 Session 校准：自然静息、两次 Pinch、两次 Fist、两次 Open、
   六个方向和最终静息复核。每个校准块的原始数据与样本区间都会保存；失败块自动补采。
6. 正式采集中持续检查饱和、断线、坏通道和丢包。异常 trial 保留为 invalid 并追加补采，
   不会无记录覆盖。Open 明确显示 Hold 与 Release，提示时刻不冒充人体动作 onset。
7. 最后一个 Trial 完成后会自动安全停止；需要提前结束时也可点击 Stop Session。原始文件保存到
   `data/<Participant>/<date>_<Session>/session.h5`。
8. 同目录生成 `SESSION_COLLECTION_READINESS.json`，检查 HDF5 v3、采样率、28 组合、Open、
   offset 平衡、有效次数、校准、质量、包审计、重录原因、元数据和文件完整性。

## 新 Protocol

在 `protocols/` 新建 JSON，无需修改 Python。例如：

```json
{
  "name": "pinch_v1",
  "labels": ["thumb_up", "index_pinch", "middle_pinch"],
  "trials_per_class": 20,
  "countdown_sec": 3.0,
  "prompt_duration_sec": 1.0,
  "rest_min_sec": 1.0,
  "rest_max_sec": 2.0,
  "randomize": true
}
```

重启软件或重新进入后即可发现该 Protocol。

### 历史 Meta 式 null 数据（非正式采集协议）

`meta_discrete_7_short_v2` 每个 Session 采集七个高层动作各 12 个 Trial（共 12 条
平衡导航路线），并在目标手势完成后追加论文明确描述的两类负样本：定时执行的
响指/屈指弹出，以及持续进行的自然键盘打字。定时 null 使用与目标动作相同的滚动
指示线；连续 null 只显示行为要求并持续记录指定时长。

以下仅说明旧数据复现安排，不适用于新的 `jilv_music_28_v2`。历史方案建议每位参与者
采集 12 个独立重新佩戴的短 Session，并按 8 个训练、2 个验证、
2 个最终测试 Session 划分。这样每个高层动作累计 144 个 Trial；保持动作分别产生
按下和松开事件，因此九个 Meta 输出事件也各累计 144 次。单个 Session 理论时长
约 9 分钟（不含重新佩戴和信号检查）。

null 数据始终写入连续 EMG、`trials`、`events` 和带起止时间的 `stages`，但不会写入
`cue_events`，不会进行动作起点对齐，也不会进入最终九类 `prompts`。因此训练时这些
时间自然对应九个输出通道全为零，而不是新增第十个 null 类。相关参数为：

```json
{
  "timed_null_actions": ["null_finger_snap", "null_finger_flick"],
  "timed_null_repetitions": 3,
  "timed_null_prompt_duration_sec": 1.0,
  "continuous_null_blocks": [
    {
      "name": "null_typing",
      "instruction": "请在键盘上自然、连续地打字",
      "duration_sec": 60.0
    }
  ]
}
```

## 数据检查

在数据检查页打开 `session.h5`，可查看 metadata、时长、EMG/IMU 样本数、
Trial 有效性、丢包事件，并按 Trial 回放 8 通道连续 EMG；红线对应 PROMPT_START。

HDF5 原始文件主结构：`/meta`、`/streams/emg`、`/streams/imu`、`/events`、
`/trials`、`/calibration_blocks`、`/cue_events` 和 `/packet_audit`。`cue_events` 仅表示 UI 发出动作提示的样本位置，
不冒充真实生理动作起点，原始文件中不写入最终 `prompts`。
EMG `raw` 为 `[N,8] int32`，显示滤波数据不会进入正式文件。
EMG 与 IMU 分别保存标称率、实测率和独立时间轴。在固件没有硬件时间戳时，
`timestamp_source=pc_reconstructed`，因此只能用于工程延迟估算，不能解释毫秒级生理时延。

Meta V3 对齐文件包含 `/data`、Pandas `/prompts(name,time)`、
`/stages(start,end,name)`、`/alignment_events`、`/alignment_templates` 和
`/alignment_match_scores`。对齐复用与官方实现数值一致的 8 通道、192 维矩阵对数 MPF，
使用 rERP 回归分离重叠动作贡献，再在整个 Session 内交替估计九类手势模板与事件时间，
并用 beam search 保持重叠动作序列的顺序。
`alignment_events` 同时保存 cue、Session 对齐位置、全局重居中位置、模板相关性、次优
相关性、搜索边界和置信度。该方法遵循论文公开的 Session 模板强制对齐结构；论文未公开
的协议约束由明确标注的本地参数替代。跨参与者全局模板是可选的版本化输入；未提供时
明确标记为未应用，因此不伪造全局重中心化。

### Meta 训练信号与 corpus

原始 `session.h5/streams/emg/raw` 始终保存设备的 24-bit ADC 计数，不执行训练滤波。
独立的 `session_meta_aligned.hdf5/data.emg` 使用版本化的 `meta_8ch_v1` 预处理：

1. 每通道去中位数基线；
2. 4 阶 40 Hz 零相位高通；
3. 50–450 Hz、Q=30 的工频谐波陷波；
4. 使用单个 Session 全局中位绝对幅值执行确定性缩放，使输出中位绝对幅值为 16。

设备协议目前只提供 ADC count，尚无经过设备厂商参数验证的 count-to-µV 换算，因此导出
明确标记为 `normalized_device_counts`，不会冒充微伏。实际滤波、陷波、缩放参数和原始单位
均保存在 `/data` 属性中。旧版未带 `preprocessing_version=meta_8ch_v1` 的导出不会通过
“训练就绪”检查，应从同目录原始 `session.h5` 重新生成，不需要重新采集。

`data/discrete_gestures_corpus.csv` 使用相对路径，可连同整个 `data/` 目录上传服务器。
同一 Session 始终只登记一次，人工审核重建 `prompts` 后会同步刷新 prompt 数量。
Meta 官方代码根据 CSV 的 `split` 列建立 DataLoader；8 通道训练还必须添加 Hydra 参数：

```text
+lightning_module.network.input_channels=8
```

`training_manifest.json` 汇总 Session 数、各 split 数量、prompt 总数、采样率、通道数和
预处理版本，上传前可用于快速复核训练包。

### 一键上传服务器

“数据上传”页与“数据检查”平级，默认使用 SSH 别名 `my-gpu-server` 和远端目录
`/home/qxy/qxy/emg_data/emgforce_dataset`。页面会重新读取 corpus、刷新 manifest、验证
所有训练文件和 train/val 划分，只上传 corpus 登记的对齐 HDF5，不上传原始
`session.h5`。传输使用系统 `ssh/scp` 和已有 SSH 密钥，不在工程内保存密码。

每个文件先上传为远端临时文件，SHA-256 一致后才替换正式文件；已有且哈希相同的文件会
跳过，已有但内容不同的文件会先改名为带时间戳的 `.backup-*` 备份。上传完成后，服务器
训练参数中的 `data_location` 应指向页面显示的远端数据集目录。

如需手动重新对齐：

```powershell
python -m emgforce.processing.meta_alignment path\to\session.h5
```

生成仅用于 v2/v3 对照、不会登记到训练 corpus 的文件：

```powershell
python -m emgforce.processing.meta_alignment path\to\session.h5 `
  --output path\to\session_meta_aligned_v3.hdf5 --no-register-corpus
```

构建参与者均衡的全局模板并在第二遍对齐时应用：

```powershell
python scripts\build_global_alignment_template.py data global_alignment_v1.h5
python -m emgforce.processing.meta_alignment path\to\session.h5 `
  --output path\to\session_meta_aligned_v3.hdf5 `
  --global-template global_alignment_v1.h5
```

`meta_discrete_7_posture_palm_up_v1` 是独立的可选姿态协议；原
`meta_discrete_7_short_v2` 及其 Null 次数、时长和顺序保持不变。没有姿态字段的历史数据
按 `unspecified` 兼容处理。

`prompts.time` 是对齐后的事件时间；Meta 训练使用的 `+80–120 ms` 目标脉冲
应在训练 transform 中动态生成，不写回 HDF5。

“数据检查 → 标签对齐质检”可查看去工频后的 8 通道活动和 V3 模板匹配分数，灰线为
原始 cue、绿线为模板 aligned 时间；支持筛选低置信度、接受、按 1/10/100 ms 微调、
波形点击覆盖及排除整个试次。人工通常只需接受或排除，模板明显失配时才覆盖。
审核只写入 `session_meta_aligned.hdf5/alignment_reviews`；只有一个试次的所有事件
都通过时才会进入 `prompts`，因此保持动作不会只留下 press 或 release。

## 测试

```powershell
pytest
```

## 本地实时手势识别

“实时识别”页从 `models/*/manifest.json` 发现模型包，只加载 PyTorch Lightning 原生
`.ckpt`，并在加载前校验 SHA-256。页面可连接训练服务器、列出每次训练的 epoch、验证准确率、
测试 CLER 与文件大小，再下载所选 CKPT 和该次运行的 Hydra 配置。当前已验证模型为
`models/emg_20260821_113915_e45`，源 checkpoint 是 `epoch=45-step=1426.ckpt`，
`val_accuracy=55.1%`、`test_cler=0.0892`。大模型二进制保留在本机但由 `.gitignore` 排除。

本机加载 CKPT 依赖 `emgforce` Conda 环境中的 Torch 2.4.1、Hydra 1.3.2、
PyTorch Lightning 1.8.6，以及 Meta 官方仓库。仓库固定在服务器训练使用的提交：

```powershell
git -C third_party/generic-neuromotor-interface checkout b6bf250e2be5a67b23488104335373cdb87a15c9
D:\Users\qxy\anaconda3\envs\emgforce\python.exe -m pip install --no-deps -e third_party/generic-neuromotor-interface
```

使用步骤：

1. 连接真实设备或模拟设备，进入“实时识别”。
2. 选择模型并点击“加载模型”。
3. 点击“开始推理校准”，按页面提示在默认 24 秒内完成静息和七种目标动作。
4. 校准完成后点击“开始实时识别”，观察九路概率和超过在线阈值 0.50 的事件。
5. 在线事件按论文执行 50 ms 防抖及 press/release 状态机；食指和中指保持至少 500 ms 才算有效 hold。
6. 也可选择任意 `session_meta_aligned.hdf5` 以论文离线阈值 0.35 运行回放。

训练预处理使用零相位滤波，严格的零延迟在线实现不存在。因此当前实时页使用 4 秒滚动
缓冲、2 秒模型窗口，并输出距最新数据约 250 ms 的固定延迟结果；模型推理运行在独立线程，
不会阻塞串口采集或原始 HDF5 保存。显示页的 20 Hz 组合滤波不会进入模型。

当前 epoch 45 模型的验证准确率约 55.1%，测试 CLER 为 0.0892。CLER 较低表示在已匹配到的
测试事件中分类混淆较少，但不等同于低漏检率；实时页因此同时显示九路概率、阈值、模型来源、
hold 状态和事件日志，便于区分“没有触发”和“触发了错误类别”。
# UniBo 8-channel live adapter

The **Realtime recognition** page automatically discovers the tracked UniBo
`E0`, `E5`, and `E6b` Day 1–5 to Day 6 artifacts. `E5` is listed first because
it is the strongest model that does not require an external posture label.

The adapter accepts the collector's eight-channel 250 Hz bipolar ADC stream,
selects four distinct channels for `ECU`, `EDC`, `FCR`, and `FCU`, removes the
rest offset, applies a 20–90 Hz band-pass, full-wave rectification, and a 3 Hz
envelope low-pass, then resamples each 200 ms window to the UniBo model's 200 Hz
rate with an anti-aliasing polyphase filter. It displays `Neutral`, `Pinch`,
`Fist`, and `Open` probabilities. Keep the hand
relaxed during the eight-second calibration so the adapter can match the four
selected channel amplitudes to the training fold. `E6b` also requires the
posture selector.

The ring electrodes do not reproduce the anatomical UniBo electrode geometry.
The UI and runtime metadata therefore mark this path as an experimental domain
adapter; its live output is not a reproduction of the reported Day 6 metrics.

# Song real 8-channel local model (exploratory)

The realtime page also discovers a locally trained `Song 8ch` model from
`models/song_real8_f0/`. It uses all eight current electrodes at 250 Hz, a
causal 40 Hz high-pass plus 50/100 Hz notch, 200 ms windows every 100 ms,
source-fitted F0 features and a four-class logistic model. Its labels are
Neutral, Index Pinch, Fist and Open Hand. Select **Song 8ch**, load it, connect
the device and click **Start recognition**. This zero-shot model does not use
the page's 24-second calibration step; the button is disabled for this model.
Sample discontinuities and packet-loss reports reset its filter and window.
The current debounced class remains visible during a sustained action; the
event log records class changes.

On this machine the model was built from `E:/qxy/emg_meta/emg_meta/data/Song/`
using S01/S02 for training and S03 for validation. To regenerate it from the
classifier directory, set `PYTHONPATH=src;.` in PowerShell and run
`python benchmarks/export_song_live_model.py --source E:/qxy/emg_meta/emg_meta/data/Song --output ../collection/emg_meta/emg_meta/models/song_real8_f0`.
The local `models/` directory is Git-ignored; no participant HDF5 or learned
weights are uploaded by the source-code commit. The model JSON is checked
against its SHA-256 manifest before loading.

S04 cue-labelled stable-trial evaluation with this causal preprocessing reached
90.97% four-state accuracy; the 28-state study is separate and is not exposed
as a live model. S01–S03 failed collection readiness, and all sessions are
from one participant on one day. Continuous live-event accuracy, onset
detection and end-to-end data age have **not** been measured. The UI marks
this model as exploratory and displays data age as unmeasured.
The full S04 cue-timeline replay is recorded in
`emgimu_classifier/benchmarks/song_real8/CONTINUOUS_REPLAY_AUDIT.json`.
