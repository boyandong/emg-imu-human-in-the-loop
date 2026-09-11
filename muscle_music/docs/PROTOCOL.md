# 纯 IMU 手环输入协议

手环识别程序通过 UDP 向 `127.0.0.1:9000` 持续发送 OSC 消息。默认地址为：

```text
/emgimu/state
timestamp_ms gesture_id confidence motion_energy reserved_expression
```

五个参数依次是毫秒时间戳、手势编号、识别置信度、运动强度和保留表达量。后三项范围均为 0 至 1。当前设备以 200 Hz 采集三轴加速度和三轴角速度，识别程序不必把 200 个原始采样逐条发给页面。建议每秒向 9000 端口汇总发送 25 至 50 帧识别结果，即使状态没有变化也持续发送，便于系统发现掉线并保持连续控制。第五个参数暂时固定发送 `0.0`，这样以后加入 EMG 或其他连续传感量时不必更改协议结构。

手势编号固定如下：

```text
0 无动作
1 前
2 后
3 左
4 右
5 上
6 下
7 开始或结束（兼容保留）
8 保留状态，当前不触发任何功能
```

系统确认新状态前要求其稳定 120 毫秒，且置信度不低于 0.55。同一方向持续发送只会产生保持事件，不会反复生成乐句。从上直接切到左同样可以被识别，不要求中间出现无动作。超过约 1.5 秒没有收到数据时，页面会把控制状态恢复为无动作，但正在播放的音乐不会停止。

当前手环只有一个六轴 IMU，不能可靠判断食指捏合、握拳力度或肌肉发力。因此第一版由用户点击页面上的“开始演奏”启用声音，方向手势只负责加入和发展声部。状态 7 留作接口兼容，不作为本阶段必须实现的动作。状态 8 收到后不会改变音乐。

`motion_energy` 可以由加速度与角速度在短时间窗内的能量归一化得到。它不代表真实空间位移，也不代表肌肉力量。首版只要求同一次佩戴内保持基本一致：挥动越快、角速度或加速度越大，运动强度越高。

浏览器通过 `serve_demo.py` 接收 UDP 数据并转成页面可以读取的实时流。HTTP 页面使用 `127.0.0.1:8765`，手环输入使用 `127.0.0.1:9000`。浏览器不能直接监听 UDP，所以不应跳过这层转发。同一台电脑上不要同时让浏览器桥和独立音乐引擎占用 9000 端口。

TouchDesigner 输出默认发送到 `127.0.0.1:9001`，消息如下：

```text
/music/gesture   gesture_id confidence motion_energy reserved_expression
/music/transport state countdown_steps bpm
/music/beat      bar beat subdivision
/music/layer     layer_id active generation evolution_stage energy change_label
/music/harmony   root_name mode_name
/music/personality label motion reserved legato complexity
/music/variation kind
/music/reset     1
/music/error     message
```

六个 `layer_id` 依次为鼓、环境纹理、和弦、主旋律、琶音和贝斯，编号为 0 至 5。
