# 数据采集与打标签系统对比审查报告 (基于 Meta 2025 论文)

本报告对当前服务器与本地代码库中的 **数据采集部分（`emgforce/experiment` 协议引擎、采集流程、数据存储）** 与 **打标签与时间对齐部分（`emgforce/processing` 时间对齐、模板生成、标签转换）** 进行了系统性审查，并与 Meta 2025 论文原文（*Kaifosh & Reardon, 2025: "A generic non-invasive neuromotor interface for human-computer interaction"*，对应 `experiment_statistics/Kaifosh和Reardon-2025-A-generic-non-invasive-neuromotor-interface-for-human-computer-interactio.md`）进行了逐项技术细节比对。

---

## 目录
- [一、 论文技术规范与实现对照总览](#一-论文技术规范与实现对照总览)
- [二、 核心缺陷与问题清单 (Issues)](#二-核心缺陷与问题清单-issues)
  - [Issue 1: 负样本（Null Data）比例严重偏低且排布集中在末尾](#issue-1-负样本null-data比例严重偏低且排布集中在末尾)
  - [Issue 2: 采集阶段缺失受试者手臂姿态（Postures）与动态运动引导](#issue-2-采集阶段缺失受试者手臂姿态postures与动态运动引导)
  - [Issue 3: 模板估计采用中位数平均（Median Average），未实现 rERP 线性回归去时域重叠](#issue-3-模板估计采用中位数平均median-average未实现-rerp-线性回归去时域重叠)
  - [Issue 4: 全局重中心化（Global Recentering）缺失带来 Session 间时间基准漂移](#issue-4-全局重中心化global-recentering缺失带来-session-间时间基准漂移)
  - [Issue 5: 对齐特征构建（log1p + Session z-score）与 Meta 标准 MPF 特征定义存在偏差](#issue-5-对齐特征构建log1p--session-z-score与-meta-标准-mpf-特征定义存在偏差)
  - [Issue 6: 软件 UI 定时器与硬件采样点绑定的时延与线程调度抖动隐患](#issue-6-软件-ui-定时器与硬件采样点绑定的时延与线程调度抖动隐患)
  - [Issue 7: 标签平移（+100 ms）在对齐导出与模型加载阶段的传递一致性](#issue-7-标签平移100-ms在对齐导出与模型加载阶段的传递一致性)
- [三、 与论文一致的优秀实现亮点](#三-与论文一致的优秀实现亮点)
- [四、 逐模块改进建议与代码方案](#四-逐模块改进建议与代码方案)

---

## 一、 论文技术规范与实现对照总览

| 模块 / 环节 | Meta 2025 论文原文标准 (Ground Truth) | 当前代码库实现状态 | 状态评估 |
| :--- | :--- | :--- | :--- |
| **手势定义与映射** | 7 类提示手势映射为 9 类基础事件（含 press/release），拇指左右划动按解剖学（in/out）区分左右手 | `session.py` 严格实现了解剖学镜像映射（左手/右手反转） | ✅ **完全一致** |
| **负样本 (Null Data)** | 约 **1/3 (33%)** 为 Null 数据，包含定时空动作（响指、屈指）与自然连续打字，分散在各阶段 | `meta_discrete_7.json` 中 Null 占比不足 **15%**，且被集中排在实验最末尾 | ❌ **严重偏少且排布不均** |
| **姿态与动态动作** | 要求覆盖 **5 种不同手臂姿态**（悬空、掌心朝内/外/上、大腿上、手臂下垂）及手臂平移/旋转 | 协议和 UI 仅包含单一静态坐姿手势提示，缺失姿态多样性引导 | ⚠️ **缺失姿态泛化维度** |
| **时间对齐生成模型** | 基于特征生成模型 \(x(t) = \sum \phi_k(t - t_k) + n(t)\) 求解时域重叠贡献 | 采用中位数切片，未求解重叠去卷积 | ⚠️ **近似实现** |
| **模板估计方法** | 采用 **rERP (Regression-based ERP)** 回归估计器解耦连续手势的时域重叠 | `template_alignment.py` 采用中位数平均（Median Average） | ⚠️ **密集动作时有干扰** |
| **全局重中心化** | 与跨受试者全局平均模板（Grand Average Global Template）对齐消除 Session 间相位漂移 | 因官方未公开全局模板，代码中标记为 `not_applied` 直接跳过 | ⚠️ **Session 间存在微小时序漂移** |
| **对齐特征定义** | 严格基于 50 Hz MPF（矩阵对数 Log-Matrix）特征计算互相关 | `template_alignment.py` 采用了 `log1p` + Session 局部 z-score 近似特征 | ⚠️ **特征定义存在细微偏差** |
| **标签时间平移** | 最终导出标签向后平移 **+100 ms** 为因果网络提供肌肉激活上下文 | 平移在 `data.py` 和 `transforms.py` 中执行，但各模块传递需防重复/遗漏 | ⚠️ **需严格统一平移位置** |

---

## 二、 核心缺陷与问题清单 (Issues)

### Issue 1: 负样本（Null Data）比例严重偏低且排布集中在末尾
* **涉及文件**：`protocols/meta_discrete_7.json` 与 `emgforce/experiment/prompt_engine.py` (行 100-101)
* **论文原文证据**（*Line 349*）：
  > *"About one-third of the training corpus was composed of a range of null data in which participants were either asked to generate specifically timed null gestures (such as snaps, flicks) or to engage in more loosely prompted longer-form null behaviours (such as typing on a keyboard)."*
* **问题分析**：
  1. **数量严重不足**：协议中目标手势共 \(12 \times 7 = 84\) 次（约 300~400 秒），但响指（snap）和屈指（flick）仅各有 3 次（共 6 次，约 15 秒），打字仅 60 秒。Null 数据总时长不足总数据的 15%，远低于 Meta 原文的 **33% (1/3)**。
  2. **阶段排布不合理**：在 `prompt_engine.py` 中，Null 动作被机械地追加到了所有目标手势的**最末尾**。受试者在采集后半段可能发生电极微移或出汗，导致 Null 数据的肌肉基线与前半段目标手势产生分布漂移（Distribution Shift）。
* **改进建议**：
  - 将 `timed_null_repetitions` 从 3 次增加至 8~10 次；
  - 增加长段自然打字时长（例如 120 秒）；
  - 在 `prompt_engine.py` 中将 Null 阶段穿插在常规手势组之间。

---

### Issue 2: 采集阶段缺失受试者手臂姿态（Postures）与动态运动引导
* **涉及文件**：`emgforce/ui/prompt_window.py` 与 `protocols/meta_discrete_7.json`
* **论文原文证据**（*Line 349*）：
  > *"During data collection for these stages, the participants were asked to hold their hand and arm in one of a range of postures (hand in front, palm facing in/out/up, hand in lap, arm hanging by side, forearm pronated inwards) or to translate/rotate their arms while completing gestures."*
* **问题分析**：
  - 当前上位机仅在屏幕上显示手势名称和图标，受试者通常全程保持同一种桌面坐姿；
  - Meta 论文强调肌电手环会随着手臂下垂、掌心旋转或放在大腿上发生前臂肌肉形态形变与重力位移；若采集时无姿态引导，训练出的模型在实际站立或手臂悬垂交互时会出现大幅性能衰减。
* **改进建议**：
  - 在协议中增加姿态阶段划分（如：Block 1 悬空掌心朝内，Block 2 掌心向上，Block 3 自然下垂，Block 4 放在大腿上），并在 `prompt_window.py` 中展示姿态引导插图。

---

### Issue 3: 模板估计采用中位数平均（Median Average），未实现 rERP 线性回归去时域重叠
* **涉及文件**：`emgforce/processing/template_alignment.py` (行 156-168)
* **论文原文证据**（*Line 377*）：
  > *"Templates were estimated by an EMG analogue of the regression-based estimator of the event-related potential (rERP), to disentangle overlapping contributions of gestures performed in a fast sequence."*
* **问题分析**：
  - 当前代码在每个事件候选中心截取 \([-0.20\text{s}, +0.20\text{s}]\) 的特征片段并计算**中位数（Median Average）**：
    ```python
    templates[name] = np.median(np.stack(snippets), axis=0).astype(np.float32)
    ```
  - 当手势之间间隔较长（如 \(>1.0\text{s}\)）时该方法有效；但对于快速连续手势（如快速 tap，或 index hold 中 press 与 release 靠得很近时），前后动作的肌肉激活特征在时域上相互重叠。简单中位数无法解耦前序动作对后序动作特征的污染，导致模板发生形变。

---

### Issue 4: 全局重中心化（Global Recentering）缺失带来 Session 间时间基准漂移
* **涉及文件**：`emgforce/processing/template_alignment.py` (行 35, 275-283)
* **论文原文证据**（*Line 385*）：
  > *"Direct application of the above procedure produced timestamps that were referenced to the session template, and there was an indeterminacy as to the timing offset within the gesture, which can vary due to initial conditions. To better standardize alignment of template timing across individuals, we performed a global recentring step at the end of timestamp estimation. Specifically, we found the time of maximal correlation between the session template (that is, for a particular participant) and a global template (grand average of all templates across participants)."*
* **问题分析**：
  - 论文指出，每个 Session 自行 EM 迭代出来的 Session 模板其中心点（\(t=0\)）存在平移不确定性（即不同 Session 的对齐中心可能对在动作启动点、动作峰值点或动作结束点）；
  - Meta 采用与所有受试者的全局大平均模板做互相关极值平移，将所有 Session 的标签基准统一定标到同一生理时钟相位；
  - 当前代码由于没有全局模板，标记为 `RECENTER_NAME = "global_template_unavailable_not_applied_v2"` 并直接跳过。
* **影响分析**：
  - 这会导致不同 Session 之间的真值标签中心存在 \(\pm 20 \sim 50\text{ ms}\) 的随机相位漂移，增大了跨 Session 训练时的标签时序噪声。
* **改进建议**：
  - 在当前受试者的多个 Session 间构建一个受试者级全局模板（Participant Grand Average Template），并在第二轮对齐时执行重中心化。

---

### Issue 5: 对齐特征构建（log1p + Session z-score）与 Meta 标准 MPF 特征定义存在偏差
* **涉及文件**：`emgforce/processing/template_alignment.py` (行 103-146)
* **论文原文证据**（*Line 359*）：
  > *"Building on robust results in the EEG space for this class of features, we applied a log-matrix operation on each of these matrices. Finally, the diagonal and the first three off-diagonals... were preserved and half-vectorized... producing a single 384 dimensional vector"*
* **问题分析**：
  - 在 `template_alignment.py` 的 `mpf_like_features` 中：
    1. 频带协方差后做的是元素级 `np.log1p(np.abs(covariance))`，而非 Meta 论文规定的**矩阵对数（Matrix Logarithm \(\log m\)）**；
    2. 特征最后还额外执行了 Session 级别的中位数与 MAD 标准化（z-score 裁剪至 \([-12, 12]\)）；
  - 这导致对齐算法提取的特征分布与模型训练时（`mpf_tds/features.py`）提取的 MPF 特征分布不一致。

---

### Issue 6: 软件 UI 定时器与硬件采样点绑定的时延与线程调度抖动隐患
* **涉及文件**：`emgforce/experiment/session.py` (行 249-256)
* **问题分析**：
  - `_gesture_cued` 是在 Qt UI 线程的定时器超时后触发的；
  - `current_sample_index` 是上位机串口/蓝牙接收线程中已经接收到的最新采样点编号；
  - **隐患**：如果 Windows 发生线程调度延迟或 Qt 主事件循环轻微卡顿（例如 20~50 ms），记录的 `cue_sample_index` 会对应到卡顿发生时的数据点，产生非生物学的时间漂移。
  - 虽然 `template_alignment` 留有搜索窗，但如果搜索窗偏紧（`SEARCH_START_SEC=0.08s`），可能导致真实动作刚好落在搜索窗左边界外。

---

### Issue 7: 标签平移（+100 ms）在对齐导出与模型加载阶段的传递一致性
* **涉及文件**：
  - `emgforce/processing/meta_alignment.py` (行 265-268)
  - `mpf_tds/data.py` (行 48)
  - `third_party/generic-neuromotor-interface/generic_neuromotor_interface/transforms.py` (行 61)
* **论文原文证据**（*Line 425*）：
  > *"To facilitate gesture detection, we then shifted these labels forward in time by 100 ms to provide the model with a 100 ms longer context of sEMG signal before making a prediction."*
* **问题分析**：
  - 在 `meta_alignment.py` 导出中，`prompts["time"]` 记录的是对齐到的物理动作发生时刻（未平移）；
  - 在 `mpf_tds/data.py` 中，执行了 `center = float(prompt.time) + 0.10`（+100 ms 平移）；
  - 在 `transforms.py` 中，使用的是 `pulse_window: [0.0, 0.040]`（未带 +100 ms 平移）。
  - **严重风险**：若通用 LSTM 训练使用了未做 +100 ms 偏移的提示时间，而 TDS 使用了 +100 ms 偏移的时间，会导致两个模型的因果时间基准不一致。必须确保所有模型的数据加载管道中统一定义 +100 ms 平移。

---

## 三、 与论文一致的优秀实现亮点

1. **解剖学左右手镜像映射** (`session.py` 行 257-273)：
   - 准确区分了左右手在屏幕左右划动与解剖学拇指内收（`thumb_in`）/外展（`thumb_out`）的镜像关系。
2. **长按试次的成对排除机制（Paired Events Exclusion）** (`template_alignment.py` 行 338-355)：
   - 对于 `index_hold` 和 `middle_hold`，若 press 或 release 中任一事件对齐置信度不足，严格将整个 trial 两个事件同步标记为 `paired_event_failed` 联动排除，防止破坏长按闭环状态。
3. **基于 Beam Search 的时序约束序列对齐** (`template_alignment.py` 行 210-240)：
   - 实现了带最小事件间隔（`MIN_EVENT_GAP_SEC = 0.040s`）的束搜索（Beam Search），有效避免了重叠搜索窗口中事件时序颠倒的问题。

---

## 四、 逐模块改进建议与代码方案

### 1. 协议中扩充 Null Data 比例 (`protocols/meta_discrete_7.json`)
```json
{
  "timed_null_actions": [
    "null_finger_snap",
    "null_finger_flick"
  ],
  "timed_null_repetitions": 8,
  "continuous_null_blocks": [
    {
      "name": "null_typing",
      "instruction": "请在键盘上自然、连续地打字",
      "duration_sec": 120.0
    }
  ]
}
```

### 2. 受试者级全局模板重中心化思路 (`template_alignment.py`)
```python
def recenter_session_templates_with_participant_average(
    session_templates: dict[str, np.ndarray],
    participant_grand_template: dict[str, np.ndarray],
) -> dict[str, int]:
    """计算当前 Session 模板与受试者大平均模板的最大互相关偏移并进行时间重对齐"""
    offsets = {}
    for name, template in session_templates.items():
        if name in participant_grand_template:
            ref = participant_grand_template[name]
            correlation = np.correlate(template.flatten(), ref.flatten(), mode="full")
            shift = int(np.argmax(correlation) - (len(ref.flatten()) - 1))
            offsets[name] = shift
        else:
            offsets[name] = 0
    return offsets
```
