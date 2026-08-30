# Generic LSTM 模型与在线识别系统核验与整改报告（基于 Meta 2025 论文）

本报告对本地代码库中的 **通用离散手势模型（Conv1D + 3层 LSTM，`generic-neuromotor-interface` 模块 / `meta_conv_lstm_v1`）** 与 **实时识别系统（`emgforce/inference` 模块）** 进行了审查，并与 Meta 2025 论文原文（*Kaifosh & Reardon, 2025: "A generic non-invasive neuromotor interface for human-computer interaction"*，对应 `experiment_statistics/Kaifosh和Reardon-2025-A-generic-non-invasive-neuromotor-interface-for-human-computer-interactio.md`）逐项比对。

> 2026-08-23 核验说明：原报告包含若干正确发现，也混入了未经测试的推断。本文已按当前本地代码整改结果修订。远端服务器上的代码版本未在本次整改中读取，因此所有“当前状态”均指本地仓库。

---

## 目录
- [一、 论文技术规范与实现对照总览](#一-论文技术规范与实现对照总览)
- [二、 核心缺陷与问题清单 (Issues)](#二-核心缺陷与问题清单-issues)
  - [Issue 1: 在线识别与离线评测中去抖动（Debouncing）存在分支漏判](#issue-1-在线识别与离线评测中去抖动debouncing存在分支漏判)
  - [Issue 2: 在线流式推理中 LSTM 无状态（Stateless）滑动窗口与瞬态热身效应](#issue-2-在线流式推理中-lstm-无状态stateless滑动窗口与瞬态热身效应)
  - [Issue 3: 同一采样时刻多类别过阈值时缺少最大概率（Argmax）优先仲裁](#issue-3-同一采样时刻多类别过阈值时缺少最大概率argmax优先仲裁)
  - [Issue 4: 训练标签平移（+100 ms）与在线推理 Fixed-Lag（250 ms）时间参数不匹配](#issue-4-训练标签平移100-ms与在线推理-fixed-lag250-ms时间参数不匹配)
  - [Issue 5: 16 通道论文原文与 8 导环形手环硬件的配置解耦与适配风险](#issue-5-16-通道论文原文与-8-导环形手环硬件的配置解耦与适配风险)
  - [Issue 6: 上位机离线回放模式缺少基于真值对齐的标准 CLER / FNR 评估](#issue-6-上位机离线回放模式缺少基于真值对齐的标准-cler--fnr-评估)
- [三、 已确认保留的实现](#三-已确认保留的实现)
- [四、 本次实施结果](#四-本次实施结果)

---

## 一、 论文技术规范与实现对照总览

| 模块 / 环节 | Meta 2025 论文原文标准 (Ground Truth) | 当前代码库实现状态 | 状态评估 |
| :--- | :--- | :--- | :--- |
| **网络架构** | Conv1D (k=21, s=10, 2000Hz\(\to\)200Hz) \(\to\) Dropout (0.1) \(\to\) LayerNorm \(\to\) 3层 LSTM (d=512, drop=0.1) \(\to\) LayerNorm \(\to\) Linear (9类) | `third_party/generic-neuromotor-interface/generic_neuromotor_interface/networks.py` 中 `DiscreteGesturesArchitecture` 严格实现 | ✅ **完全一致** |
| **非线性输入压缩** | 40 Hz 高通滤波后，经 Reinhard 算子 \(f(x) = 64 \times \frac{x}{32 + \|x\|}\) 抑制异常值 | `ReinhardCompression(range=64.0, midpoint=32.0)` 严格实现 | ✅ **完全一致** |
| **优化器与学习率** | Adam；前 5 epoch 线性预热，此后每 25 epoch 衰减 0.5；梯度裁剪；多标签 BCE | 已配置周期性 milestones；梯度裁剪移至 Lightning Trainer | ✅ **已整改** |
| **释放损失掩码** | 手指未处于 press 状态时不计算 release 负样本损失 | 已修复“窗口内有 press、无 release”时掩码错误；窗口起始状态仍受窗口标签可观测性限制 | ⚠️ **边界有限制** |
| **验证指标与 checkpoint** | \([-50\text{ ms}, +150\text{ ms}]\) 内执行 argmax Proxy CLER | `collect_metric` 实现该代理指标，ModelCheckpoint 选择最佳权重；没有 EarlyStopping | ✅ **指标一致，非早停** |
| **离线序列评测** | 0.35 阈值、50 ms 去抖、双轮 NW、状态机、CLER | 去抖已按论文文字修正；上位机回放现可输出 CLER 与 Mean FNR | ✅ **已整改** |
| **在线识别状态机** | 阈值 0.50；移除孤立释放；未完成释放时自动合成 Release；500 ms Hold 判定 | `emgforce/inference/engine.py` 中 `OnlineGestureStateMachine` 完整实现 | ✅ **完全一致** |
| **去抖动算法** | 50 ms 去抖（仅当前事件为 release 且前序手势不同才豁免） | 在线、官方 CLER 和 MPF-TDS 三条路径已统一 | ✅ **已整改** |
| **在线流式机制** | 论文没有公开足以复刻内部流式状态管理的细节 | 当前为 2 秒无状态滑窗；尚无证据证明改为持久化 \((h_t,c_t)\) 会提升现有模型 | ⚠️ **待基准测试** |

---

## 二、 核心缺陷与问题清单 (Issues)

### Issue 1: 在线识别与离线评测中去抖动存在分支漏判（已修复）
> 以下代码与复现描述记录修复前行为。当前在线引擎、官方 CLER 和 MPF-TDS 评测已统一采用论文规则，并有回归测试覆盖 `release → 非 release`。
* **涉及文件**：`emgforce/inference/engine.py` (行 104-112) 与 `mpf_tds/metrics.py` (行 110-116)
* **论文原文证据**（*Line 427 & Line 505*）：
  > *"In debouncing, whenever a gesture was predicted within 50 ms of another gesture, the second gesture was removed. The sole exception was release events, which were not debounced when preceded by a different gesture, to ensure the inclusion of quick index/middle taps (that is, a press immediately followed by a release)."*
* **规则要求**：
  在 50 ms 时间窗口内，任何后续手势都必须被消除；**唯一的例外是：当前事件是 release 且前一个事件是不同事件（例如 press 紧接着 release）时才保留**。
* **问题分析**：
  查看当前 `engine.py` 中的去抖实现：
  ```python
  if last_event_index is not None and sample_index - last_event_index < debounce_samples:
      previous_is_release = last_event_name in release_names
      current_is_release = name in release_names
      if not current_is_release and not previous_is_release:
          continue
      if current_is_release and last_event_name == name:
          continue
  ```
  **漏判分支**：
  若前一个事件是 `index_release`（`previous_is_release = True`），当前事件是 `thumb_click` 或 `thumb_swipe`（`current_is_release = False`），且时间间隔 \(<50\text{ ms}\)：
  - `not current_is_release and not previous_is_release` 判定为 `False`；
  - `current_is_release and last_event_name == name` 判定为 `False`；
  - **代码跳过了 `continue`，导致非释放事件穿透了去抖过滤器！**
* **修正逻辑**：
  ```python
  if last_event_index is not None and sample_index - last_event_index < debounce_samples:
      current_is_release = name in release_names
      # 论文规则：50ms 窗口内仅保留非重复的释放事件，其余所有冲突一律去抖消除
      if not (current_is_release and name != last_event_name):
          continue
  ```

---

### Issue 2: 在线流式推理中 LSTM 无状态（Stateless）滑动窗口与瞬态热身效应
> 核验结论：无状态重算和重叠计算属实；“需要 0.5~1.0 秒热身”“概率会显著偏低”未得到当前模型实验支持。持久化 LSTM 状态会改变训练/推理分布，暂不作为默认修复。
* **涉及文件**：
  - `emgforce/inference/engine.py` (行 293-315)
  - `emgforce/inference/model_bundle.py` (行 187-203)
* **问题分析**：
  1. **LSTM 的隐状态丢失**：
     LSTM 是具备时序记忆的状态模型。当前系统在线识别每次定时触发时，截取一段长度为 `model_window_seconds`（默认 2.0 秒）的肌电片段，直接调用 `model(tensor)`，每次都从全零隐状态 \((h_0=0, c_0=0)\) 重新前向传播。
  2. **瞬态热身（Transient Warmup）影响**：
     LSTM 从零状态启动需要约 0.5~1.0 秒的历史输入来建立稳定的时序表征。`engine.py` 只截取窗口末端的输出帧，因此 2.0 秒窗口能够较好地吸收初始瞬态。但如果参数 `model_window_seconds` 被调小（如 1.0 秒以内），会导致输出概率显著偏低或延迟增大。
  3. **重复计算开销**：
     每次推理重复计算了过去 2.0 秒的数据（计算量是增量计算的 20 倍）。对于算力受限的设备，理想方案是维护持续递推的 \((h_t, c_t)\) 隐状态。

---

### Issue 3: 同一采样时刻多类别过阈值时缺少最大概率优先仲裁（已修复）
> 当前实现先按概率选择同一输出帧的唯一事件，概率相同时稳定回退到标签顺序，再执行跨时间去抖。论文没有规定此仲裁方式，这是为消除固定类别顺序偏置而采用的本地策略。
> 以下代码与问题分析记录修复前行为。
* **涉及文件**：`emgforce/inference/engine.py` (行 93-103)
* **问题分析**：
  在 `detect_threshold_events()` 中：
  ```python
  for column, sample_index in enumerate(indices):
      current = probs[:, column]
      crossed = np.flatnonzero((current >= thresholds) & (previous < thresholds))
      for label_index in crossed:
          candidates.append((
              int(sample_index), labels[int(label_index)], float(current[label_index])))
  ```
  如果某一时刻由于肌肉协同激活，多个手势的预测概率同时超过阈值（如 `index_press=0.60`, `middle_press=0.85`）：
  - 代码直接按标签固定顺序依次加入候选；
  - 排序仅根据 `sample_index`，同一时刻事件的先后顺序取决于该类别的下标；
  - 导致先被遍历到的弱手势（0.60）优先被采纳，并将在 50ms 窗口内的强手势（0.85）消除。
* **修正逻辑**：
  在候选列表排序时增加预测概率降序作为第二排序键：`sorted(candidates, key=lambda item: (item[0], -item[2]))`，确保同一时刻置信度最高的手势优先触发。

---

### Issue 4: 标签平移与 Fixed-Lag 被错误地视为同一参数（原结论不成立）
* 训练标签的 +100 ms 平移决定模型学习在哪个时点输出事件。
* 250 ms Fixed-Lag 为零相位实时滤波提供未来上下文，两者用途不同，不要求数值相等。
* 可见延迟近似由标签上下文、Fixed-Lag、推理调度和计算耗时累加，不能用 `250-100=150 ms` 推导。
* 默认 Fixed-Lag 保持 250 ms。界面现显示模型输出数据龄，后续应基于 CLER/FNR 与延迟实测决定是否缩短。

---

### Issue 5: 16 通道论文硬件与本地 8 通道适配（当前已配置）
网络类构造函数仍保留官方的 16 通道默认值，但有效训练 YAML、算法规格、导出逻辑和当前模型包均为 8 通道。远程训练从该 YAML 读取配置，不再需要额外 Hydra override。旧的 UI 与训练 manifest 提示已经清理。

---

### Issue 6: 上位机离线回放缺少 CLER / FNR（已修复）
`OfflineReplayWorker` 现在读取 `prompts`，两种算法均输出 Mean FNR，Conv-LSTM 额外输出官方 CLER。分块推理增加历史重叠，丢弃重叠输出，以避免 Conv1D 边界缺口并减轻 LSTM 每块冷启动。没有 `prompts` 时界面会明确说明未计算指标。

---

## 三、 已确认保留的实现

1. Reinhard 压缩、Conv1D、三层 LSTM、LayerNorm 和输出层结构与开源实现一致；本项目将输入通道适配为 8。
2. Proxy CLER 的时间窗口和 argmax 逻辑与论文描述一致，验证指标用于选择最佳 checkpoint，但训练不会提前终止。
3. 在线状态机正确过滤孤立 release，能在 press 被其他事件打断时合成 release，并计算 500 ms hold。
4. 本地预处理额外使用陷波与 session 级稳健缩放。这是明确的工程适配，不应表述为与论文固定缩放“完全相同”。

---

## 四、 本次实施结果

- 修复三条路径的 50 ms 去抖分支。
- 增加同帧最高概率仲裁及稳定的概率相同兜底。
- 离线回放增加历史重叠、CLER、Mean FNR 和无真值提示。
- 离线回放增加 0.10–0.50 阈值扫描、宏 F1、误报率、分类别 FNR 与推荐阈值。
- 学习率改为每 25 epoch 衰减，梯度裁剪移至 Trainer。
- 修复 press 后无 release 的损失掩码边界。
- 清理过时的 8 通道 override 提示。
- 在线界面显示输出数据龄；Fixed-Lag 默认值未作未经验证的调整。
- 本地测试结果：完整测试 92 项通过。
