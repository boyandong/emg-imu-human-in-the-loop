# UniBo 人体表征消融实验计划

## 1. 研究问题与当前基线

主研究问题是：在固定的跨日协议下，加入区域募集、人体解剖关系和肌肉协同表示，
能否减少 `PINCH`、`FIST`（UniBo 原标签 `power_grip`）和 `OPEN` 之间的混淆？

2026-09-12 的冻结参考结果为：

- RBF-SVM 四类 window macro-F1：`0.6921`；
- 主动手势 macro-F1：`0.5909`；
- `FIST` F1：`0.4855`；
- `NEUTRAL` F1：`0.9958`。

第一优先指标是：

```text
active_gesture_macro_f1 = mean(F1_PINCH, F1_FIST, F1_OPEN)
```

总体 accuracy 不是主要优化目标，因为它会被接近满分的 `NEUTRAL` 主导。

## 2. 信息层次

实验必须区分四类信息：

1. `G0`–`G5` 是可送入手势分类器的 EMG 特征；
2. `G6 posture` 是条件化 EMG 解释的上下文，不是 EMG 特征；
3. `D/H/A/M/P/C/Q` 是 HumanState 输出或内部状态，不是本实验的输入特征；
4. signal quality、坏通道、漂移和校准残差是质量/拒识依据，不混入生理特征消融。

内部概念结构为：

```text
Raw EMG
  -> regional activation proxies
  -> recruitment/anatomy/co-contraction/synergy/temporal representation
  -> H
  -> qH
```

UniBo 的四个通道是 ECU、EDC、FCR、FCU 附近的表面区域。它们只能解释为
muscle-region activation proxy，不能写成没有串扰的单块肌肉真值。

## 3. 特征组

### G0：冻结参考特征

当前真实实现每通道提取 RMS、MAV、standard deviation、waveform length、IQR 和
relative energy，共 24 维。首轮不得悄悄改变这组特征。

UniBo 输入接近整流/包络信号。ZC、SSC、MDF、WAMP、AR 和频谱熵不进入首轮；
只有确认其在该信号语义下有效后，才能作为单独的探索组验证。

### G1：区域募集分布

- 四区域 RMS proportion；
- spatial entropy；
- max channel share；
- second/max ratio；
- active-region count。

G0 已包含基于 RMS 平方的 relative energy。G1 必须记录自己新增了什么，不能把
已有 relative energy 重复包装成新贡献。active-region count 的静息阈值只能由
训练数据或显式校准数据拟合，并随模型产物保存。

### G2：解剖关系

- extensor-related activity：`ECU + EDC`；
- flexor-related activity：`FCR + FCU`；
- `log((F + eps) / (E + eps))`；
- 归一化屈伸平衡；
- `EDC/ECU`、`FCR/FCU` 等区域关系。

这些变量只作为软证据。禁止写成 `F > E => FIST` 一类硬规则。

### G3：共同收缩与空间关系

- `CCI = 2 * min(E, F) / (E + F + eps)`；
- 共同收缩幅值，并与总激活同时保留；
- 区域集中程度；
- 探索性的通道相关性或共同升高特征。

相关性可用于分类，但不能直接解释成神经协同，因为表面电极串扰也能产生相关性。

### G4：NMF muscle synergy

- 先在非负的窗口级区域激活上比较 `k=2` 与 `k=3`；
- NMF basis 只能用训练数据拟合；
- validation/test 只能投影到冻结 basis，禁止重新拟合；
- 至少运行多个初始化，报告 basis 稳定性和分类结果波动；
- synergy 的生理名称只能在检查权重且证据稳定后给出；
- 必须验证 NMF 是否提供了超出显式比例特征的新信息。

### G5：时间募集

第一轮保持 200 ms window 和 200 ms hop，只增加：

- early/late RMS；
- early-late 差值与稳定比例；
- activation slope；
- activation/synergy change。

更短 hop、onset 检测和更长历史属于单独协议实验，因为它们会同时改变样本相关性、
延迟和实时输出频率。

### G6：posture context

比较 `P(H | EMG)` 与 `P(H | EMG, posture)`。至少包含：

- `G0 + posture`；
- `best EMG representation + posture`。

UniBo posture 标签是无误差的 oracle context，只能表示姿态信息的理论上限。生产系统
还需要用 IMU 估计 posture，并另行做错误敏感性验证。

## 4. 实验矩阵与顺序

第一阶段使用同一 RBF-SVM、窗口、训练采样、权重和超参数选择规则：

| Run | 输入 | 要回答的问题 |
|---|---|---|
| E0 | G0 | 重现冻结参考基线 |
| E1 | G0+G1 | 相对区域募集是否有效 |
| E2 | G0+G2 | 解剖关系是否有效 |
| E3 | G0+G3 | 共同收缩/空间关系是否有效 |
| E123 | G0+G1+G2+G3 | 静态人体表示是否互补 |

第二阶段只在第一阶段代码和口径验证通过后进行：

| Run | 输入 | 要回答的问题 |
|---|---|---|
| E4-k2 | E123+G4(k=2) | 两个协同是否增加新信息 |
| E4-k3 | E123+G4(k=3) | 三个协同是否增加新信息 |
| E5 | E123+G5 | 时间过程是否有效 |
| E45 | best-static+best-G4+G5 | 协同与时间是否互补 |

第三阶段验证姿态条件化：

| Run | 输入 | 要回答的问题 |
|---|---|---|
| E6a | G0+G6 | posture 的独立贡献 |
| E6b | best-EMG+G6 | posture 对最佳表示的附加贡献 |

完整组合只有在逐组实验完成后才有解释价值。只有完整组合确实改善，才运行
`All-G1`、`All-G2` 等 leave-one-group-out 消融。

## 5. 数据纪律

主 benchmark 继续使用 `splits.json`：Day 1–5 train、Day 6 validation、Day 7–8
test。trial 不能跨集合，窗口不能跨 trial，权重继续使用：

```text
equal subject/day -> trial -> truth-label segment -> window
```

所有 scaler、静息阈值、NMF basis、温度和拒识阈值只能在允许的 train/validation
范围拟合。比例特征必须在不破坏跨通道比例的尺度上计算，禁止先逐通道独立归一化。

Day 7–8 已在参考 baseline 中打开。新特征开发不得反复查看或针对 Day 7–8 调参；
新模型在该集合上的结果只能作为连续性比较，不能重新宣称完全盲测。开发时优先在
Day 1–5 内增加按日期的内层验证，Day 6 作为确认。若要形成新的强结论，需要预注册
新的 subject-held-out/chronological protocol 或获得新的未查看数据。

## 6. 必须报告的证据

每个实验至少报告：

- all-class window macro-F1；
- active-gesture macro-F1；
- Pinch/Fist/Open 的 precision、recall、F1；
- Fist->Open 和 Fist->Pinch 混淆；
- trial-label-segment macro-F1；
- subject/day/posture 分层结果和最差组；
- all-class 与 active-only ECE、coverage、risk-coverage；
- 特征维数、特征名称、拟合状态、随机种子、运行时长和模型 SHA-256；
- 以 trial 或 subject/day 为重采样单元的差值置信区间，不能把相关窗口当独立样本。

一个特征组只有在多个日期/受试者上的方向一致、提升不只来自 Neutral 或单个被试、
且没有明显恶化最差组时，才可以进入下一阶段。没有足够证据时写“未确认”，不得把
符合生理直觉的特征直接写成已验证的生理机制。

## 7. 第一批实现与验收

第一批只要求：

1. 在现有指标中加入 active-gesture macro-F1、active-only ECE/risk-coverage；
2. 实现可开关、具名且数值稳定的 G1/G2/G3；
3. 为特征维数、有限值、训练专用阈值、确定性和无测试拟合增加测试；
4. 运行 E0/E1/E2/E3/E123，默认不读取 test；
5. 生成一份结果总结，区分直接证据、可能解释和未验证原因；
6. 不覆盖 2026-09-12 的参考模型与结果目录。

若第一阶段没有稳定收益，应先分析特征冗余、跨日漂移和四通道可观测性，而不是直接
增加网络规模。TCN 多随机种子与更大模型属于后续工作，第一阶段不需要高端 GPU。
