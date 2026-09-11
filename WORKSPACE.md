# EMG–IMU Human-in-the-loop 工作区

## 核心模块

- `collection/`：数据采集程序，来源于 `doublesong1/EMG`，已连同原提交历史并入总仓库；导入版本为 `collection-v2.1.0`。
- `emgimu_classifier/`：HumanState 分类与实时服务。
- `muscle_music/`：音乐交互与前端反馈。
- `docs/research_reviews/`：科研过程复盘。
- `docs/project_proposals/`：项目书及修订稿。

## 数据位置

旧采集数据暂时保留在：

`collection/emg_meta/emg_meta/data/`

该目录包含既有 HDF5 实验数据。整理工作区时未删除、移动或改写这些数据。

## 临时材料

- `work/`、`tmp/`、`output/`：既有课程报告和临时产物，本次未移动，避免破坏其他任务路径。
