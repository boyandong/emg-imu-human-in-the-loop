# EMG Data Collection

纯 sEMG 科研数据采集上位机：连接 8 通道、2000 Hz、24-bit EMG + 6-axis IMU
设备，按 JSON Protocol 展示 Prompt，并将连续原始数据、事件与 Trial 索引写入 HDF5。

采集主线不包含在线训练或识别。Meta 七动作协议在原始文件安全关闭后，
会运行离线 Session 模板强制对齐并生成独立的 Meta 格式导出；对齐永远不修改原始采集文件。

## 安装与启动

```powershell
conda env create -f environment.yml
conda activate emgforce
python main.py
```

界面包含“设备监测 / 实验采集 / 数据检查 / 数据上传 / 训练模型 / 实时识别”六页。可在设备页选择串口，也可用
`Connect Simulator` 在没有硬件时完成全流程。

## 新 Session

默认实验协议为 `jilv_music_9_v3`，九项采集内容与首版音乐交互分工一致：
自然松手（手指松散张开且不用力撑开）、主观 7/10 稳定握拳、前、后、左、右、上、下、拇指与食指捏合。
张手和握拳样本用于力度标定，不映射为音乐开关或新声部。
每轮保持 108 个试次：六个手臂方向各 4 次，张手放松、7/10 稳定握拳和食指捏合
各 28 次。相较原来的均匀分配，方向数据各减少 2/3，腾出的试次平均补给后三类。
旧的 Meta 拇指手势协议仍保留用于原研究复现，但不再作为新采集的默认协议。

1. 连接真实设备或模拟器，在设备页确认 8 通道波形。
   正式音乐体验前，在“音乐力度控制”卡片开始约 17 秒的个人标定：先自然松手约
   4 秒，手指保持松散张开且不要用力撑开；其中前 1 秒只等待信号稳定，不计入基线。
   随后按提示以主观 7/10 力度稳定握拳 3 次，每次约 2.6 秒。手指需要完全合拢，能够稳定重复，
   且不颤抖、不疼痛。该握拳参考映射为 0.85，给演奏时更强的自然发力
   留出约 15% 余量；输出仍限制在 0–1，不要求极限用力。
2. 在实验采集页填写 Participant ID 和 Experiment Name。离开 Participant ID 输入框后，
   Session ID 会根据该受试者跨日期已有数据自动更新为下一轮 `Sxx`。每人固定采集 11 轮；
   自动模式将 S01–S08 设为 train、S09–S10 设为 val、S11 设为 test。
3. 点击 Signal Check，确认 8 个通道的 RMS、Peak-to-Peak 与饱和状态。
4. 选择 Protocol，点击 Start Session。参与者窗口可以拖到第二显示器。
5. 实验期间可 Pause/Resume、Skip、Repeat、Mark Bad、Manual Mark，并切换 Donning/Stage。
6. 最后一个 Trial 完成后会自动安全停止；需要提前结束时也可点击 Stop Session。原始文件保存到
   `data/<Participant>/<date>_<Session>/session.h5`。
7. Meta 七动作协议会额外生成 `session_meta_aligned.hdf5`，并在 `data/` 根目录原子更新
   `discrete_gestures_corpus.csv` 和 `training_manifest.json`。

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

### Meta 式 null 数据

`meta_discrete_7_short_v2` 每个 Session 采集七个高层动作各 12 个 Trial（共 12 条
平衡导航路线），并在目标手势完成后追加论文明确描述的两类负样本：定时执行的
响指/屈指弹出，以及持续进行的自然键盘打字。定时 null 使用与目标动作相同的滚动
指示线；连续 null 只显示行为要求并持续记录指定时长。

建议每位参与者采集 12 个独立重新佩戴的短 Session，并按 8 个训练、2 个验证、
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
`/trials`、`/cue_events`。`cue_events` 仅表示 UI 发出动作提示的样本位置，
不冒充真实生理动作起点，原始文件中不写入最终 `prompts`。
EMG `raw` 为 `[N,8] int32`，显示滤波数据不会进入正式文件。

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
