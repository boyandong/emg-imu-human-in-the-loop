# TouchDesigner 可视化接入

建立一个 OSC In DAT，Network Port 设为 9001，Callbacks DAT 指向 `touchdesigner/osc_callbacks.py`。再建立一个名为 `jilv_state` 的 Table DAT，回调会自动把最新状态写入该表。

画面建议由中心节拍环和六个声部环组成。六个环按顺时针排列，分别代表鼓、环境、和弦、旋律、琶音和贝斯。每个环至少读取以下值：

```text
layer_0_active
layer_0_energy
layer_0_generation
layer_0_stage
```

其余声部只需替换编号。`active` 控制明暗，`energy` 控制半径或粒子速度，`generation` 和 `stage` 控制图形复杂度。当前手势可读取 `gesture_id`，节拍读取 `bar`、`beat` 和 `subdivision`。

作品个性化结果位于 `personality_label`、`personality_motion`、`personality_muscle`、`personality_legato` 和 `personality_complexity`。标签适合放在中心，四个连续值可以控制中心图形的速度、亮度、拖尾和层数。作品结束后保留这些值，形成一张可拍照的个人音乐画像。

新手势确认后，`gesture_id` 会立即变化；音乐可能要等到下一小节才更新。因此画面应先播放一次 150 至 250 毫秒的确认闪光，再让对应声部环在小节边界稳定亮起。食指捏合结束时，六个环一起收束，并保留本次作品的调性、声部数和演化层级。下一次捏合开始新作品时再清空画面。

展会现场不要把原始 EMG 波形作为主体。可以在画面底部保留一条较细的肌电曲线或能量条，帮助观众理解信号来源；主体仍应表现声部、节拍和个人演奏强度，否则画面容易像医学监护仪。
