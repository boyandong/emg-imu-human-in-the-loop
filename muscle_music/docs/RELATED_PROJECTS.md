# 相关项目与采用的经验

Atau Tanaka 与 R. Benjamin Knapp 早在 2002 年就把 EMG 与相对位置传感结合用于音乐控制。他们指出，离散手势识别可以触发事件，但是否具有充分的音乐创造性并不确定；EMG 的整体动态能量更适合作为连续表达控制。运动传感器擅长测量手臂运动，EMG 擅长测量没有明显位移的肌肉张力，两者应互相补充。本项目据此让 IMU 方向选择声部，让 EMG 与运动强度连续影响力度、音色和密度，而不是把全部信号只用于九分类。原文：[Multimodal Interaction in Music Using the Electromyogram and Relative Position Sensing](https://www.nime.org/proceedings/2002/nime2002_171.pdf)

2024 年发表的 EAVI-EMG 是一套从肌电采集、手势特征映射到颗粒合成的完整数字乐器。它采用多维映射和交互式机器学习，并通过公开演出检验非专业音乐人能否使用。欧盟 BioMusical Instrument 项目也强调把多维肌肉特征映射到声音元数据，而不是固定的一手势一音符。本项目借鉴了它的高层映射方法，但首版采用约束式 MIDI，降低两个月原型的延迟和调试风险。资料：[EAVI-EMG 论文记录](https://napier-repository.worktribe.com/output/3597697)、[欧盟项目总结](https://cordis.europa.eu/project/id/789825/reporting)

NIME 2025 的 EMMA 同时使用 EMG 与运动传感器，把手指、手掌和手臂动作作为实时作曲输入。论文把动作与声音之间的即时性，以及演奏者与乐器之间能否形成稳定关系，列为数字乐器设计的关键问题。本项目因此让画面立即确认手势，而将较大的音乐变化量化到小节边界；用户能够立刻知道动作被接收，同时不会破坏节拍。[EMMA 项目页](https://nime.org/proc/nime2025_35/index.html)

早期的 EEG 生成音乐系统没有把脑电波逐点变成音符，而是根据主要频段选择不同风格的生成规则，再用信号复杂度控制速度和动态。2014 年的 The Space Between Us 也把估计的效价和唤醒度映射到具有相应情绪内容的乐句，并定期选择下一乐句。它们说明生物信号适合控制作曲规则和结构层级，而不是直接承担全部作曲工作。本项目的四级主题演化和动作强度映射来自这一思路。资料：[Brain-Computer Music Interface for Generative Music](https://citeseerx.ist.psu.edu/document?doi=e474adb95ec299c94798853b03364db9679a440c&repid=rep1&type=pdf)、[The Space Between Us](https://www.nime.org/proceedings/2014/nime2014_418.pdf)

Brainrave 把 EEG 注意或放松程度同时映射到音乐、灯光和一个可以被参与者直接观察的注意力环。它适合公众展览的重要原因不是参数多，而是观众能很快理解自己的状态怎样影响现场。本项目用节拍环、六个声部环和即时确认动画保留这种可理解的反馈。[Brainrave 项目页](https://www.lateinteractive.com/brainrave)

2026 年一项使用额前 EEG 控制情绪音乐的初步研究没有发现目标情绪或训练时间能可靠改变控制信号，个体差异反而解释了更多变化。这提醒我们不要把不可稳定控制的生理指标包装成精确意图。本项目只用 EMG 与 IMU 支持用户明确做出的动作，并把个体差异转化为会话内的音乐风格参数，不宣称能够识别情绪或心理状态。资料：[A Minimalist Brain-Computer Musical Interface for Real-Time Emotion-Driven Sonification](https://arxiv.org/abs/2606.01473)

综合这些项目，本项目采用三个原则：方向负责清楚的结构选择，连续生理与运动量负责表达，同一动作应保持稳定含义但允许主题逐步发展。这样既保留第一次使用时的可学性，也为重复使用提供足够变化。
