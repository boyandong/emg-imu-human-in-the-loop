# Ableton Live 12 配置

先在 loopMIDI 中建立 `JILV OUT` 端口。打开 Ableton 的 MIDI 设置，在 Input Ports 中启用该端口的 Track 和 Remote。

新建六条 MIDI 轨道，输入均选择 `JILV OUT`，监听设为 In。六条轨道的 MIDI 通道必须按下面顺序设置：

| MIDI 通道 | 声部 | 建议音色 |
| --- | --- | --- |
| 1 | 鼓与节奏 | Drum Rack，电子鼓组 |
| 2 | 环境纹理 | Wavetable 或 Drift 长音色 |
| 3 | 和弦铺底 | 柔和 Pad |
| 4 | 主旋律 | 清晰但不过亮的 Lead |
| 5 | 高音琶音 | Pluck 或 Bell |
| 6 | 贝斯 | 单声道 Sub Bass |

每条轨道都接收 CC11 和 CC74。CC11 用于表达强度，CC74 用于滤波亮度。最简单的配置是在 MIDI Effects 中加入 Expression Control，把 Expression 映射到 Utility Gain 或乐器宏，把 Brightness 映射到滤波器频率。映射范围不要从完全静音开始，否则轻动作会让声部消失；建议保留约 -18 dB 的最低电平。

所有轨道发送到同一个短混响和一个四分音符延迟。贝斯和和弦可以由鼓轨做轻度侧链。主总线至少保留 6 dB 余量，并使用 Limiter 防止展会现场连续变奏造成削波。

首次调试时先用很普通的音色验证六个 MIDI 通道。确认每个方向只控制一条轨道后，再替换成最终音色。音色设计不应依赖随机切换预设，个性化主要由乐句、力度、密度和滤波变化完成，这样每次演示更稳定。

暂停或程序退出时会发送 All Notes Off 和 All Sound Off。如果仍有卡音，检查乐器是否忽略 MIDI CC123，并在 Ableton 顶部点击 Stop 两次完成手动 Panic。
