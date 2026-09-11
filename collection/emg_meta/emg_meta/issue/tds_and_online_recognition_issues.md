# TDS 代码与在线识别系统对比审查报告 (基于 Meta 2025 论文)

本报告对当前服务器与本地代码库中的 **单参与者 MPF+TDS 模型实现（`mpf_tds` 模块）** 与 **实时识别系统（`emgforce/inference` 模块）** 进行了系统性审查，并与 Meta 2025 论文原文（*Kaifosh & Reardon, 2025: "A generic non-invasive neuromotor interface for human-computer interaction"*，对应 `experiment_statistics/Kaifosh和Reardon-2025-A-generic-non-invasive-neuromotor-interface-for-human-computer-interactio.md`）进行了逐项技术细节比对。

---

## 目录
- [一、 论文技术规范与实现对照总览](#一-论文技术规范与实现对照总览)
- [二、 核心缺陷与问题清单 (Issues)](#二-核心缺陷与问题清单-issues)
  - [Issue 1: 训练阶段缺失信号幅值缩放与 40 Hz 高通滤波（严重分布失配）](#issue-1-训练阶段缺失信号幅值缩放与-40-hz-高通滤波严重分布失配)
  - [Issue 2: 去抖动（Debouncing）算法存在逻辑漏判分支](#issue-2-去抖动debouncing算法存在逻辑漏判分支)
  - [Issue 3: MPF 矩阵对数（Matrix Logarithm）奇异特征值截断破坏物理单调性](#issue-3-mpf-矩阵对数matrix-logarithm奇异特征值截断破坏物理单调性)
  - [Issue 4: 同一时间步多手势过阈值时缺少最大概率（Argmax）仲裁](#issue-4-同一时间步多手势过阈值时缺少最大概率argmax仲裁)
  - [Issue 5: 训练标签平移（+100 ms）与在线推理 Fixed-Lag（250 ms）时间参数不匹配](#issue-5-训练标签平移100-ms与在线推理-fixed-lag250-ms时间参数不匹配)
  - [Issue 6: 8 导环形手环特征维度适配与 16 导论文原文的通道扩展兼容性](#issue-6-8-导环形手环特征维度适配与-16-导论文原文的通道扩展兼容性)
- [三、 与论文一致的实现亮点](#三-与论文一致的实现亮点)
- [四、 逐文件修复建议与代码补丁](#四-逐文件修复建议与代码补丁)

---

## 一、 论文技术规范与实现对照总览

| 模块 / 环节 | Meta 2025 论文原文标准 (Ground Truth) | 当前代码库实现状态 | 状态评估 |
| :--- | :--- | :--- | :--- |
| **手势定义** | 9 类离散手势：食指/中指按下与释放、拇指点击、拇指上下左右滑动 | `LABELS` 包含 9 类手势 | ✅ 一致 |
| **MPF 基础预处理** | 乘以 \(2.46 \times 10^{-6}\) 归一化噪声；40 Hz 4阶 Butterworth 高通滤波 | 训练 `data.py` 缺失；在线流预处理具备 | ❌ **严重不一致** |
| **MPF 特征构建** | STFT (64 点, hop 10), 6 频带, CSD 互谱外积, 矩阵对数, 4 条循环对角线 | `mpf_tds/features.py` 实现完整，但特征值对数截断有缺陷 | ⚠️ **需修复截断** |
| **TDS 网络架构** | 1 FC + LeakyReLU \(\to\) 6 尺度级联（每尺度 2 个 TDS block + AvgPool + Dilation \(2^s\)）\(\to\) 3 FC | `mpf_tds/model.py` 严格按照公式级联 | ✅ 一致 |
| **优化器与学习率** | Adam, 1~5 epoch 预热至 \(10^{-3}\), 25 epoch 衰减至 \(5\times 10^{-4}\), 共 300 epochs | `mpf_tds/train.py` 严格实现 | ✅ 一致 |
| **标签时间平移** | 经过对齐后的手势标签向后平移 +100 ms 提供因果上下文 | `mpf_tds/data.py` 使用了 +0.10s 平移 | ✅ 一致 |
| **离线去抖动与评测** | 阈值 0.35；50 ms 去抖（仅不同前序的 release 豁免）；双轮 NW 匹配与状态机过滤；Mean FNR | `mpf_tds/metrics.py` 实现了 NW 与 FNR，但去抖有逻辑漏洞 | ⚠️ **去抖逻辑有误** |
| **在线识别状态机** | 阈值 0.50；移除孤立释放；未完成释放时自动合成 Release；500 ms Hold 判定 | `emgforce/inference/engine.py` 实现了状态机与合成释放 | ⚠️ **去抖与仲裁需优化** |

---

## 二、 核心缺陷与问题清单 (Issues)

### Issue 1: 训练阶段缺失信号幅值缩放与 40 Hz 高通滤波（严重分布失配）
* **涉及文件**：`mpf_tds/data.py` (行 31-39)
* **论文原文证据**（*Line 359*）：
  > *"To obtain these features, we first rescaled the sEMG by \(2.46 \times 10^{-6}\), to normalize the s.d. of the noise to 1.0 (this value was determined empirically). Motivated by the need to remove motion artifacts, we then applied a 40 Hz high-pass filter (fourth-order Butterworth) to the sEMG recordings sampled at 2 kHz."*
* **问题分析**：
  `mpf_tds/data.py` 中的 `load_split()` 函数在从 HDF5 中读取 `signal = handle["data"]["emg"]` 之后，直接执行了：
  ```python
  raw = torch.from_numpy(signal[start:start + required_samples].T[None])
  features = extractor(raw)[0]
  ```
  这里**既没有进行 \(2.46 \times 10^{-6}\)（或基于会话的归一化）缩放，也没有进行 40 Hz 高通滤波**。
  - 如果 HDF5 保存的是原始 ADC counts，数值量级为 \(10^3 \sim 10^5\)，直接计算傅里叶外积会导致特征数值过大；
  - 低频电极基线漂移和运动伪影未滤除，污染了 0~62.5 Hz 的第 1 频段；
  - 更重要的是，实时推理引擎（`emgforce/inference/preprocessing.py`）对在线流执行了滤波和缩放，导致**训练特征与推理特征的数值尺度相差几个数量级**。

---

### Issue 2: 去抖动（Debouncing）算法存在逻辑漏判分支
* **涉及文件**：
  - 离线评测：`mpf_tds/metrics.py` (行 110-116)
  - 实时识别：`emgforce/inference/engine.py` (行 104-112)
* **论文原文证据**（*Line 427*）：
  > *"In debouncing, whenever a gesture was predicted within 50 ms of another gesture, the second gesture was removed. The sole exception was release events, which were not debounced when preceded by a different gesture, to ensure the inclusion of quick index/middle taps (that is, a press immediately followed by a release)."*
* **问题分析**：
  论文规则为：**50 ms 窗口内，任何手势都必须被消除；唯一的例外是：当前事件是 release 且前一个事件是不同事件（如 press \(\to\) release）时才保留。**
  但看当前代码实现：
  ```python
  if last_event_index is not None and sample_index - last_event_index < debounce_samples:
      previous_is_release = last_event_name in release_names
      current_is_release = name in release_names
      if not current_is_release and not previous_is_release:
          continue
      if current_is_release and last_event_name == name:
          continue
  ```
  **漏判情况**：
  当 `last_event_name` 是 `index_release`（`previous_is_release = True`），而当前事件是 `thumb_click`（`current_is_release = False`），且时间差 \(<50\text{ ms}\) 时：
  - `not current_is_release and not previous_is_release` 计算为 `False`；
  - `current_is_release and last_event_name == name` 计算为 `False`；
  - **代码未执行 `continue`，导致该 `thumb_click` 错误地穿透了去抖器！**
* **修正逻辑**：
  ```python
  if last_event_index is not None and sample_index - last_event_index < debounce_samples:
      current_is_release = name in release_names
      # 唯一的幸存条件：当前是 release，且与上一个事件不同
      if not (current_is_release and name != last_event_name):
          continue
  ```

---

### Issue 3: MPF 矩阵对数（Matrix Logarithm）奇异特征值截断破坏物理单调性
* **涉及文件**：`mpf_tds/features.py` (行 83-85)
* **论文原文证据**（*Line 359*）：
  > *"This produced a set of 6 symmetric and positive definite 16 × 16 square matrices that update every 40 samples... we applied a log-matrix operation on each of these matrices."*
* **问题分析**：
  现有代码计算矩阵对数时：
  ```python
  eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
  eigenvalues = eigenvalues.log().nan_to_num(nan=0.0, neginf=0.0)
  matrix_log = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
  ```
  当处于肌电静息期时，某些通道的能量非常小，矩阵的特征值可能接近 0 或因单精度误差略小于 0。
  - 正常的对数映射应该是一个绝对值很大的负数（例如 \(\ln(10^{-6}) \approx -13.8\)）；
  - 代码使用 `nan_to_num(nan=0.0, neginf=0.0)` 把负无穷强制替换成了 `0.0`（对应 \(\ln(1.0) = 0\)）；
  - **这导致极小能量/噪声反而被映射成了比中等能量（如 \(\ln(0.01) = -4.6\)）更大的值（0.0），造成了严重的非单调能量失真。**
* **修正逻辑**：
  应在取对数前进行数值下限截断（`clamp`）：
  ```python
  eigenvalues = torch.clamp(eigenvalues, min=self.config.matrix_log_epsilon).log()
  ```

---

### Issue 4: 同一时间步多手势过阈值时缺少最大概率（Argmax）仲裁
* **涉及文件**：`emgforce/inference/engine.py` (行 93-103)
* **问题分析**：
  在 `detect_threshold_events()` 中，如果同一时刻模型输出的多个通道概率均超过阈值（如 `index_press` 为 0.60，`middle_press` 为 0.85）：
  - 代码按 `labels` 的固定遍历顺序生成候选；
  - 排序仅根据 `sample_index`，同一时刻事件的相对顺序取决于类别在列表中的索引；
  - 导致先被遍历的弱置信度手势优先被采纳，并将后遍历的高置信度手势 debounce 消除。
* **修正逻辑**：
  同一时间步内若有多个过阈值事件，应按**置信度/概率降序排列**，优先触发最大概率的手势。

---

### Issue 5: 训练标签平移（+100 ms）与在线推理 Fixed-Lag（250 ms）时间参数不匹配
* **涉及文件**：
  - 训练：`mpf_tds/data.py` (行 48: `center = float(prompt.time) + 0.10`)
  - 推理：`emgforce/inference/engine.py` (行 24: `fixed_lag_seconds: float = 0.25`)
* **问题分析**：
  - 论文中模型训练时将标签平移了 +100 ms，意味着模型是学习在动作起始后的 100 ms 因果窗口处达到概率峰值；
  - 在线推理引擎中如果设置了 250 ms 的 Fixed-Lag（即窗口右边界延迟当前采样点 250 ms），事件触发点与真实动作之间会引入额外的 150 ms 固定延迟；
  - 建议将实时配置中的 `realtime_fixed_lag_seconds` 统一调整为 `0.10` ~ `0.15` 秒。

---

### Issue 6: 8 导环形手环特征维度适配与 16 导论文原文的通道扩展兼容性
* **涉及文件**：`mpf_tds/model.py` (行 12) & `mpf_tds/features.py` (行 12)
* **说明**：
  - Meta 原文使用 16 通道硬件，保留 4 条对角线，6 频带，特征维度为 \(6 \times 4 \times 16 = 384\) 维；
  - 本地系统适配了 8 通道硬件，特征维度为 \(6 \times 4 \times 8 = 192\) 维；
  - 该适配在数学结构上是完全正确的，但在配置文件与网络初始化中，应确保 `input_dim` 能够根据 `channels` 自动计算（`channels * 24`），以保持对 16 导数据的向下兼容。

---

## 三、 与论文一致的实现亮点

在审查中，当前代码在以下核心算法模块上高度准确地复现了 Meta 2025 原文：

1. **六尺度级联 TDS 网络结构** (`mpf_tds/model.py`)：
   - 严格实现了 6 个 scale（\(s=0\dots5\)），每个 scale 包含两个 dilation 为 \(2^s\) 的 TDS block；
   - 实现了基于左侧填充（causal padding）的 \(2^s\) 时间尺度 `AveragePool`，无未来时间泄露；
   - 跨尺度残差相加（`current = current + previous`）及 3 层输出 FC 完全忠实于论文第 393 行。
2. **学习率调度（LR Warm-up & Decay）** (`mpf_tds/train.py`)：
   - 1~5 epoch 线性预热到 \(1\times 10^{-3}\)，第 25 epoch 后单次衰减到 \(5\times 10^{-4}\)，总共 300 轮训练。
3. **在线手势状态机与自动合成释放（Synthetic Release）** (`emgforce/inference/engine.py`)：
   - 完整实现了论文第 505 行规定的机制：当处于 index/middle hold 状态但用户接着做其他手势时，状态机**先自动合成一个对应释放事件**，保证长按闭环状态不会卡死。
4. **离线双轮 Needleman-Wunsch 序列对齐与 Mean FNR 指标** (`mpf_tds/metrics.py`)：
   - 严格实现了 \([-50\text{ ms}, +250\text{ ms}]\) 容差窗口的 Needleman-Wunsch 序列匹配，经过状态机过滤后执行第二轮匹配计算逐类别 FNR。

---

## 四、 逐文件修复建议与代码补丁

### 1. 修复 `mpf_tds/data.py` (加入预处理滤波与缩放)
```diff
--- a/mpf_tds/data.py
+++ b/mpf_tds/data.py
@@ -6,6 +6,7 @@ from pathlib import Path
 import h5py
 import numpy as np
 import pandas as pd
+from scipy.signal import butter, sosfiltfilt
 import torch
 from torch.utils.data import TensorDataset
 
@@ -14,6 +15,12 @@ from .features import MPFConfig, MultiBandMatrixPowerFeatures
 LABELS = ("index_press", "index_release", "middle_press", "middle_release",
           "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up")
 
+def _preprocess_emg(signal: np.ndarray, sample_rate: float = 2000.0) -> np.ndarray:
+    scaled = np.asarray(signal, dtype=np.float64) * 2.46e-6
+    sos = butter(4, 40.0, btype="highpass", fs=sample_rate, output="sos")
+    filtered = sosfiltfilt(sos, scaled, axis=0)
+    return filtered.astype(np.float32)
+
 def load_split(data_root: str | Path, split: str, sequence_frames: int = 400) -> TensorDataset:
@@ -31,6 +38,7 @@ def load_split(data_root: str | Path, split: str, sequence_frames: int = 400) -
         with h5py.File(path, "r") as handle:
             signal = np.asarray(handle["data"]["emg"], dtype=np.float32)
             sample_times = np.asarray(handle["data"]["time"], dtype=np.float64)
+        signal = _preprocess_emg(signal)
         prompts = pd.read_hdf(path, "prompts")
```

### 2. 修复 `mpf_tds/features.py` (特征值截断)
```diff
--- a/mpf_tds/features.py
+++ b/mpf_tds/features.py
@@ -82,7 +82,7 @@ class MultiBandMatrixPowerFeatures(nn.Module):
             matrix = cross_spectral_density[:, :, mask].mean(dim=2)
             eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
-            eigenvalues = eigenvalues.log().nan_to_num(nan=0.0, neginf=0.0)
+            eigenvalues = torch.clamp(eigenvalues, min=self.config.matrix_log_epsilon).log()
             matrix_log = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
```

### 3. 修复 `emgforce/inference/engine.py` (去抖漏判与概率排序)
```diff
--- a/emgforce/inference/engine.py
+++ b/emgforce/inference/engine.py
@@ -102,15 +102,12 @@ def detect_threshold_events(
     release_names = {"index_release", "middle_release"}
     events: list[DetectedEvent] = []
-    for sample_index, name, probability in sorted(candidates, key=lambda item: item[0]):
+    for sample_index, name, probability in sorted(candidates, key=lambda item: (item[0], -item[2])):
         if last_event_index is not None and sample_index - last_event_index < debounce_samples:
-            previous_is_release = last_event_name in release_names
             current_is_release = name in release_names
-            # Meta: suppress two nearby non-release events, and repeated releases
-            # of the same kind. A press/release pair and different releases survive.
-            if not current_is_release and not previous_is_release:
-                continue
-            if current_is_release and last_event_name == name:
+            # 论文规则：50ms 窗口内仅保留非重复的释放事件，其余所有冲突一律去抖消除
+            if not (current_is_release and name != last_event_name):
                 continue
         events.append(DetectedEvent(
```

### 4. 修复 `mpf_tds/metrics.py` (离线去抖修正)
```diff
--- a/mpf_tds/metrics.py
+++ b/mpf_tds/metrics.py
@@ -110,9 +110,8 @@ def _events_from_probabilities(probabilities: np.ndarray, labels: tuple[str, ...
         if filtered and event["time"] - filtered[-1]["time"] < 0.05:
             previous = filtered[-1]
-            if event["name"] not in releases and previous["name"] not in releases:
-                continue
-            if event["name"] in releases and event["name"] == previous["name"]:
+            is_release = event["name"] in releases
+            if not (is_release and event["name"] != previous["name"]):
                 continue
         filtered.append(event)
```
