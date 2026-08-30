# 来源与统计口径

## 主要来源

1. Kaifosh, Reardon and CTRL-labs at Reality Labs, *A generic non-invasive
   neuromotor interface for human-computer interaction*, Nature (2025).
   DOI: <https://doi.org/10.1038/s41586-025-09255-w>
2. Meta 官方代码与开放数据仓库：
   <https://github.com/facebookresearch/generic-neuromotor-interface>

## 官方开放数据统计方法

`meta_official_discrete_counts.csv` 来自官方约 31.1 GB 数据归档中的全部 100 份
离散手势 HDF5 记录。统计时逐文件读取 prompt/event 标签并按类别计数，然后计算
跨记录中位数，同时记录数据中明显存在的两种采集规模：

- 46 份高重复记录：每种食指/中指 press、release 各 400 次，每种拇指动作 60 次；
- 54 份较短记录：press/release 通常约 100–115 次，拇指点击约 100–123 次，
  拇指方向动作约 140–170 次。

CSV 中“典型范围”用于描述这一组记录的规模，不表示论文规定的硬性上下限。

## 本工程采用 12 × 12 的原因

本工程没有直接复制官方单次长记录，而是把数据分散到 12 次独立重新佩戴的短
Session。这样每个目标事件类别累计 144 次，处在官方较短记录的常见规模附近，
同时额外覆盖跨 Session、跨佩戴位置的变化。每类统一为每 Session 12 个 Trial，
也能保持现有七动作平衡导航路线的结构。

## 限制

- 论文将独立重新佩戴/调整腕带视为不同 Session，因此本工程的 12 个 Session
  也必须真正重新佩戴，不能只是在软件里连续改 Session ID。
- 官方开放数据没有把响指、屈指弹出等 null 行为分别作为目标 prompt 发布，
  所以无法从开放标签得到它们的官方精确次数。本工程采用每 Session 各 3 次，
  属于明确标注的本地实验设计。
- 自然打字同样作为连续 null 数据采集，不构成第十个分类标签。
- 本目录记录的是 2026-08-19 的设计快照。若修改协议 JSON，应同步更新此目录。

## 上位机训练导出约定

从 `meta_8ch_v1` 起，原始 `session.h5` 继续保留设备 ADC count；训练用
`session_meta_aligned.hdf5/data.emg` 单独执行 40 Hz 高通、50 Hz 谐波陷波和明确记录参数的
Session 全局幅值归一化。数据根目录同时生成 `discrete_gestures_corpus.csv`，按照
S01–S08 / S09–S10 / S11–S12 分别登记为 train / val / test。旧的未预处理导出不得与
新导出混合训练。
