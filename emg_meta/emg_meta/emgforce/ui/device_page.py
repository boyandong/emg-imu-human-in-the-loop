from __future__ import annotations

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from serial.tools import list_ports

from emgforce.config import BAUDRATE, EMG_CHANNELS, SAMPLING_RATE
from emgforce.filters import EmgDisplayFilterBank
from emgforce.protocol import ParserStats
from emgforce.quality.monitor import SignalQualityMonitor


EMG_COLORS = [
    "#2563eb", "#7c3aed", "#0891b2", "#059669",
    "#d97706", "#dc2626", "#db2777", "#4f46e5",
]
AXIS_COLORS = ["#2563eb", "#e11d48", "#059669"]


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(); frame.setObjectName("card")
    layout = QVBoxLayout(frame); layout.setContentsMargins(16, 14, 16, 14); layout.setSpacing(10)
    return frame, layout


def section_label(text: str) -> QLabel:
    label = QLabel(text); label.setObjectName("sectionTitle"); return label


class Metric(QWidget):
    def __init__(self, caption: str, value: str) -> None:
        super().__init__(); layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0); layout.setSpacing(1)
        self.value = QLabel(value); self.value.setObjectName("metricValue")
        caption_label = QLabel(caption); caption_label.setObjectName("muted")
        layout.addWidget(self.value); layout.addWidget(caption_label)


class SignalPlotCard(QFrame):
    def __init__(self, title: str, colors: list[str], legends: list[str] | None = None,
                 compact: bool = False, show_time_axis: bool = True) -> None:
        super().__init__(); self.setObjectName("signalPlotRow")
        outer = QHBoxLayout(self); outer.setContentsMargins(0, 2, 0, 2); outer.setSpacing(8)
        self.visible_box = QCheckBox(title); self.visible_box.setChecked(True)
        self.visible_box.setStyleSheet("font-weight: 700; color: #344054;")
        self.visible_box.setFixedWidth(70 if compact else 88)
        self.value_label = QLabel("--"); self.value_label.setObjectName("muted")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setFixedWidth(82 if compact else 150)
        outer.addWidget(self.visible_box)
        self.plot = pg.PlotWidget(background="#ffffff")
        height = 78 if compact else 130
        self.plot.setMinimumHeight(height); self.plot.setMaximumHeight(height)
        self.plot.showGrid(x=False, y=False); self.plot.hideAxis("left")
        if show_time_axis:
            self.plot.setLabel("bottom", "相对当前时刻", units="s")
            self.plot.getAxis("bottom").setTextPen("#667085")
            self.plot.getAxis("bottom").setPen("#d0d5dd")
        else:
            self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False); self.plot.setMouseEnabled(x=False, y=False)
        self.plot.disableAutoRange(); self.plot.getViewBox().setDefaultPadding(0.0)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#e5e7eb")))
        outer.addWidget(self.plot, 1); outer.addWidget(self.value_label)
        legend = self.plot.addLegend(offset=(-8, 5)) if legends else None
        if legend:
            legend.setBrush(pg.mkBrush(255, 255, 255, 220)); legend.setPen(pg.mkPen("#e2e8f0"))
        self.curves: list[pg.PlotCurveItem] = []
        for index, color in enumerate(colors):
            curve = pg.PlotCurveItem(pen=pg.mkPen(QColor(color), width=2.0))
            self.plot.getPlotItem().addItem(curve)
            if legend: legend.addItem(curve, legends[index])
            self.curves.append(curve)
        self.visible_box.toggled.connect(self._toggle_plot)

    def _toggle_plot(self, visible: bool) -> None:
        self.plot.setVisible(visible); self.value_label.setVisible(visible)
        self.setMaximumHeight(16_777_215 if visible else 34)


class DevicePage(QWidget):
    """Reference-compatible live monitor backed by the new acquisition controller."""

    connect_requested = Signal(str, bool)
    disconnect_requested = Signal()
    calibration_requested = Signal()
    MAX_DISPLAY_SECONDS = 10
    DEFAULT_DISPLAY_SECONDS = 5

    def __init__(self) -> None:
        super().__init__(); self.setObjectName("devicePage")
        self.is_connected = False; self.highpass_enabled = False
        self._filter = EmgDisplayFilterBank(EMG_CHANNELS, SAMPLING_RATE)
        self._qc_raw: deque[np.ndarray] = deque(maxlen=SAMPLING_RATE * 5)
        self.emg_time: deque[float] = deque(maxlen=SAMPLING_RATE * self.DEFAULT_DISPLAY_SECONDS)
        self.emg_values = [deque(maxlen=SAMPLING_RATE * self.DEFAULT_DISPLAY_SECONDS) for _ in range(8)]
        self.imu_time: deque[float] = deque(maxlen=1000 * self.DEFAULT_DISPLAY_SECONDS)
        self.gyro_values = [deque(maxlen=1000 * self.DEFAULT_DISPLAY_SECONDS) for _ in range(3)]
        self.accel_values = [deque(maxlen=1000 * self.DEFAULT_DISPLAY_SECONDS) for _ in range(3)]
        self.last_stats = ParserStats(); self.last_rate_time = time.monotonic()
        self.rate_emg_frames = 0; self.rate_imu_frames = 0
        pg.setConfigOptions(antialias=False, foreground="#667085")

        self.header = self._build_header()
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        body = QHBoxLayout(); body.setContentsMargins(18, 16, 18, 16); body.setSpacing(16)
        body.addWidget(self._build_sidebar()); body.addWidget(self._build_dashboard(), 1)
        body_widget = QWidget(); body_widget.setLayout(body); root.addWidget(body_widget, 1)
        self.update_axis_ranges(); self.refresh_ports()
        self.render_timer = QTimer(self); self.render_timer.setInterval(50)
        self.render_timer.timeout.connect(self.render_plots); self.render_timer.start()

    def _build_header(self) -> QFrame:
        header = QFrame(); header.setObjectName("header")
        layout = QHBoxLayout(header); layout.setContentsMargins(24, 14, 24, 14)
        title_box = QVBoxLayout(); title_box.setSpacing(1)
        title = QLabel("EMG Data Collection"); title.setObjectName("appTitle")
        subtitle = QLabel("8 通道肌电 · 惯性测量 · 实时科研采集工作站"); subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title); title_box.addWidget(subtitle); layout.addLayout(title_box); layout.addStretch()
        self.connection_metric = Metric("设备状态", "未连接")
        self.emg_rate_metric = Metric("EMG 包速率", "0 Hz")
        self.loss_metric = Metric("检测丢包", "0")
        self.record_metric = Metric("Session 记录", "未记录")
        for metric in (self.connection_metric, self.emg_rate_metric, self.loss_metric, self.record_metric):
            layout.addWidget(metric)
        return header

    def _build_sidebar(self) -> QScrollArea:
        sidebar = QWidget(); sidebar.setObjectName("sidebarContent")
        # The Windows scrollbar is an overlay at some DPI settings, so keep a
        # right gutter rather than allowing controls underneath it.
        layout = QVBoxLayout(sidebar); layout.setContentsMargins(0, 0, 18, 0); layout.setSpacing(12)
        # Never vertically compress cards below their content size. On a
        # high-DPI full-screen desktop the logical height can be smaller than
        # the default window even though there are more physical pixels.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        connect_card, connect_layout = card(); connect_layout.addWidget(section_label("设备连接"))
        self.port = QComboBox(); self.port.setToolTip("选择串口，也可使用模拟设备检查界面")
        self.port.setFixedHeight(36)
        self.port.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.port.setMinimumContentsLength(28)
        connect_layout.addWidget(self.port)
        buttons = QHBoxLayout(); buttons.setSpacing(8)
        self.refresh = QPushButton("刷新端口")
        self.connect = QPushButton("连接"); self.connect.setObjectName("primary")
        self.disconnect = QPushButton("断开"); self.disconnect.setObjectName("danger"); self.disconnect.setEnabled(False)
        for button in (self.refresh, self.connect, self.disconnect): buttons.addWidget(button, 1)
        connect_layout.addLayout(buttons)
        self.simulate = QPushButton("连接模拟设备")
        for button in (self.refresh, self.connect, self.disconnect, self.simulate):
            button.setFixedHeight(36)
        connect_layout.addWidget(self.simulate)
        params = QLabel(f"{BAUDRATE} baud  ·  8-N-1  ·  无流控"); params.setObjectName("muted")
        detail = QLabel(f"{SAMPLING_RATE} Hz  ·  {EMG_CHANNELS} 通道  ·  signed int24"); detail.setObjectName("muted")
        connect_layout.addWidget(params); connect_layout.addWidget(detail); layout.addWidget(connect_card)
        self.refresh.clicked.connect(self.refresh_ports)
        self.connect.clicked.connect(
            lambda: self.connect_requested.emit(str(self.port.currentData() or ""), False)
        )
        self.simulate.clicked.connect(lambda: self.connect_requested.emit("__SIM__", True))
        self.disconnect.clicked.connect(self.disconnect_requested)

        display_card, display_layout = card(); display_layout.setSpacing(12)
        display_layout.addWidget(section_label("显示控制"))
        settings = QGridLayout(); settings.setHorizontalSpacing(12); settings.setVerticalSpacing(10)
        settings.setColumnStretch(1, 1)
        settings.addWidget(QLabel("时间窗口"), 0, 0)
        self.window_spin = QSpinBox(); self.window_spin.setRange(1, self.MAX_DISPLAY_SECONDS)
        self.window_spin.setValue(self.DEFAULT_DISPLAY_SECONDS); self.window_spin.setSuffix(" 秒")
        self.window_spin.setFixedHeight(36); self.window_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.window_spin.valueChanged.connect(self.update_display_window)
        settings.addWidget(self.window_spin, 0, 1)
        settings.addWidget(QLabel("EMG 固定 Y 轴"), 1, 0)
        self.emg_range_spin = QSpinBox(); self.emg_range_spin.setRange(50, 50_000)
        self.emg_range_spin.setSingleStep(50); self.emg_range_spin.setValue(15_000)
        self.emg_range_spin.setPrefix("± "); self.emg_range_spin.setSuffix(" μV")
        self.emg_range_spin.setFixedHeight(36); self.emg_range_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emg_range_spin.valueChanged.connect(self.update_axis_ranges)
        settings.addWidget(self.emg_range_spin, 1, 1)
        display_layout.addLayout(settings)
        self.highpass_button = QPushButton("EMG 组合滤波：关闭"); self.highpass_button.setCheckable(True)
        self.highpass_button.setFixedHeight(36)
        self.highpass_button.setToolTip("均值去除 · 50/100 Hz 陷波 · 20–850 Hz 带通 · 仅影响显示")
        self.highpass_button.toggled.connect(self.toggle_highpass); display_layout.addWidget(self.highpass_button)
        display_buttons = QHBoxLayout(); display_buttons.setSpacing(8)
        self.pause_button = QPushButton("暂停显示"); self.pause_button.setCheckable(True)
        self.clear_button = QPushButton("清空曲线")
        self.pause_button.setFixedHeight(36); self.clear_button.setFixedHeight(36)
        display_buttons.addWidget(self.pause_button); display_buttons.addWidget(self.clear_button)
        self.pause_button.toggled.connect(lambda checked: self.pause_button.setText("继续显示" if checked else "暂停显示"))
        self.clear_button.clicked.connect(self.clear_buffers); display_layout.addLayout(display_buttons); layout.addWidget(display_card)

        music_card, music_layout = card(); music_layout.addWidget(section_label("音乐力度控制"))
        explanation = QLabel("自然松手与主观 7/10 稳定握拳用于个人标定。松手时让手指松散张开，不要用力撑开；握拳应可重复保持且不颤抖不疼痛。")
        explanation.setObjectName("muted"); explanation.setWordWrap(True)
        explanation.setStyleSheet("font-size: 14px;"); music_layout.addWidget(explanation)
        self.calibration_instruction = QLabel("连接手环后开始标定")
        self.calibration_instruction.setWordWrap(True)
        self.calibration_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_instruction.setMinimumHeight(76)
        self.calibration_instruction.setStyleSheet(
            "font-size: 20px; font-weight: 800; color: #0f172a; padding: 10px 6px;"
        )
        music_layout.addWidget(self.calibration_instruction)
        self.calibration_progress = QProgressBar(); self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0); self.calibration_progress.setFixedHeight(24)
        music_layout.addWidget(self.calibration_progress)
        self.calibration_button = QPushButton("开始个人力度标定（约 17 秒）")
        self.calibration_button.setObjectName("primary"); self.calibration_button.setFixedHeight(44)
        self.calibration_button.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.calibration_button.setEnabled(False); self.calibration_button.clicked.connect(self.calibration_requested)
        music_layout.addWidget(self.calibration_button)
        self.music_values_label = QLabel("EMG 力度：0.00  ·  IMU 运动：0.00")
        self.music_values_label.setObjectName("muted")
        self.music_values_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        music_layout.addWidget(self.music_values_label)
        self.music_output_label = QLabel("音乐输出：设备未连接")
        self.music_output_label.setObjectName("muted"); self.music_output_label.setWordWrap(True)
        music_layout.addWidget(self.music_output_label); layout.addWidget(music_card)

        stats_card, stats_layout = card(); stats_layout.addWidget(section_label("接收统计"))
        self.total_frames_label = QLabel("有效帧：0"); self.emg_frames_label = QLabel("EMG 帧：0")
        self.imu_rate_label = QLabel("IMU 包速率：0 Hz"); self.duplicate_label = QLabel("重复帧：0")
        self.discarded_label = QLabel("丢弃字节：0"); self.received_bytes_label = QLabel("接收字节：0")
        for label in (self.received_bytes_label, self.total_frames_label, self.emg_frames_label,
                      self.imu_rate_label, self.duplicate_label, self.discarded_label):
            label.setObjectName("muted"); stats_layout.addWidget(label)
        layout.addWidget(stats_card)

        qc_card, qc_layout = card(); qc_layout.addWidget(section_label("实时 Signal Quality"))
        self.qc_label = QLabel("等待 EMG 数据"); self.qc_label.setObjectName("muted"); self.qc_label.setWordWrap(True)
        qc_layout.addWidget(self.qc_label); layout.addWidget(qc_card); layout.addStretch()
        scroll = QScrollArea(); scroll.setObjectName("sidebarScroll")
        scroll.setWidgetResizable(True); scroll.setWidget(sidebar)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Reserve enough logical width for translated labels plus the vertical
        # scrollbar at 125–200% Windows scaling.
        scroll.setMinimumWidth(430); scroll.setMaximumWidth(470)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll = scroll
        return scroll

    def _build_dashboard(self) -> QScrollArea:
        content = QWidget(); content.setObjectName("dashboard")
        layout = QVBoxLayout(content); layout.setContentsMargins(0, 0, 4, 4); layout.setSpacing(14)
        heading = QHBoxLayout(); heading.addWidget(section_label("肌电信号 · 8 通道 · 2 kHz")); heading.addStretch()
        hint = QLabel("固定时间窗 · 固定幅值 · 已关闭缩放与拖动"); hint.setObjectName("muted"); heading.addWidget(hint)
        layout.addLayout(heading)
        grid = QVBoxLayout(); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(2)
        self.emg_cards: list[SignalPlotCard] = []
        for index in range(8):
            plot_card = SignalPlotCard(f"CH{index + 1}", [EMG_COLORS[index]], compact=True,
                                       show_time_axis=index == 7)
            self.emg_cards.append(plot_card); grid.addWidget(plot_card)
        layout.addLayout(grid); layout.addWidget(section_label("惯性测量 · 实时三轴数据"))
        imu_grid = QVBoxLayout(); imu_grid.setContentsMargins(0, 0, 0, 0); imu_grid.setSpacing(4)
        self.gyro_card = SignalPlotCard("角速度", AXIS_COLORS, ["X", "Y", "Z"])
        self.accel_card = SignalPlotCard("加速度", AXIS_COLORS, ["X", "Y", "Z"])
        imu_grid.addWidget(self.gyro_card); imu_grid.addWidget(self.accel_card); layout.addLayout(imu_grid); layout.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dashboard_scroll = scroll; return scroll

    def refresh_ports(self) -> None:
        current = self.port.currentData() if hasattr(self, "port") else ""
        if hasattr(self, "port"):
            self.port.clear()
            for info in list_ports.comports():
                label = self._port_label(info)
                self.port.addItem(label, info.device)
                self.port.setItemData(
                    self.port.count() - 1,
                    self._port_tooltip(info),
                    Qt.ItemDataRole.ToolTipRole,
                )
            if current:
                index = self.port.findData(current)
                if index >= 0:
                    self.port.setCurrentIndex(index)

    @staticmethod
    def _port_label(info: object) -> str:
        device = str(getattr(info, "device", ""))
        description = str(getattr(info, "description", "") or "").strip()
        if description.lower() in {"", "n/a", device.lower()}:
            manufacturer = str(getattr(info, "manufacturer", "") or "").strip()
            description = "" if manufacturer.lower() == "n/a" else manufacturer
        # pyserial descriptions often already end in "(COMx)".
        if description.endswith(f"({device})"):
            description = description[:-(len(device) + 2)].strip()
        return f"{device}  —  {description}" if description else device

    @staticmethod
    def _port_tooltip(info: object) -> str:
        lines = [f"端口：{getattr(info, 'device', '')}"]
        for title, attribute in (("设备", "description"), ("制造商", "manufacturer"),
                                 ("产品", "product"), ("序列号", "serial_number")):
            value = getattr(info, attribute, None)
            if value and str(value).lower() != "n/a":
                lines.append(f"{title}：{value}")
        vid, pid = getattr(info, "vid", None), getattr(info, "pid", None)
        if vid is not None and pid is not None:
            lines.append(f"VID:PID：{vid:04X}:{pid:04X}")
        hwid = getattr(info, "hwid", None)
        if hwid:
            lines.append(f"HWID：{hwid}")
        return "\n".join(lines)

    def set_connected(self, name: str, connected: bool) -> None:
        self.is_connected = connected
        self.connection_metric.value.setText(name if connected else "未连接")
        self.connection_metric.value.setStyleSheet("color: #059669;" if connected else "")
        self.connect.setEnabled(not connected); self.simulate.setEnabled(not connected)
        self.disconnect.setEnabled(connected); self.port.setEnabled(not connected); self.refresh.setEnabled(not connected)
        self.calibration_button.setEnabled(connected)
        # A disabled-state stylesheet repolish can otherwise leave stale row
        # geometry on Windows fractional DPI scaling.
        QTimer.singleShot(0, self._activate_layout)

    def _activate_layout(self) -> None:
        for widget in (self.port, self.refresh, self.connect, self.simulate, self.disconnect,
                       self.window_spin, self.emg_range_spin, self.highpass_button,
                       self.pause_button, self.clear_button, self.calibration_button):
            widget.updateGeometry()
        if self.layout() is not None:
            self.layout().invalidate(); self.layout().activate()

    def set_recording(self, recording: bool) -> None:
        self.record_metric.value.setText("记录中" if recording else "未记录")
        self.record_metric.value.setStyleSheet("color: #e11d48;" if recording else "")

    def set_music_values(self, effort: float, motion: float) -> None:
        self.music_values_label.setText(f"EMG 力度：{effort:.2f}  ·  IMU 运动：{motion:.2f}")

    def set_music_calibration(self, instruction: str, progress: int, ready: bool) -> None:
        self.calibration_instruction.setText(instruction)
        self.calibration_progress.setValue(max(0, min(100, int(progress))))
        self.calibration_button.setText("重新标定个人力度" if ready else "重新开始力度标定")

    def set_music_output(self, status: str) -> None:
        self.music_output_label.setText(f"音乐输出：{status}")

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.clear_buffers(reset_filter=False)
        if not self.pause_button.isChecked():
            if not self.render_timer.isActive():
                self.render_timer.start()
            self.render_plots()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self.render_timer.stop()

    def append_emg(self, raw: np.ndarray, index: np.ndarray) -> None:
        paused = self.pause_button.isChecked()
        visible = self.isVisible()
        for values, sample_index in zip(raw, index):
            values = np.asarray(values, dtype=np.int32); self._qc_raw.append(values.copy())
            self.rate_emg_frames += 1
            if paused or not visible: continue
            filtered = self._filter.process(values)
            display = filtered if self.highpass_enabled else values
            self.emg_time.append(float(sample_index) / SAMPLING_RATE)
            for channel_values, value in zip(self.emg_values, display): channel_values.append(value)
        self._update_rates_if_due()

    def update_imu(self, gyro: np.ndarray, accel: np.ndarray, monotonic_ns: np.ndarray) -> None:
        paused = self.pause_button.isChecked(); visible = self.isVisible()
        self.rate_imu_frames += len(gyro)
        if not paused and visible:
            for gyro_row, accel_row, stamp in zip(gyro, accel, monotonic_ns):
                self.imu_time.append(float(stamp) / 1_000_000_000)
                for values, value in zip(self.gyro_values, gyro_row): values.append(value)
                for values, value in zip(self.accel_values, accel_row): values.append(value)
        self._update_rates_if_due()

    def _update_rates_if_due(self) -> None:
        now = time.monotonic(); elapsed = now - self.last_rate_time
        if elapsed < .5: return
        emg_rate = self.rate_emg_frames / elapsed; imu_rate = self.rate_imu_frames / elapsed
        self.rate_emg_frames = 0; self.rate_imu_frames = 0; self.last_rate_time = now
        self.emg_rate_metric.value.setText(f"{emg_rate:,.0f} Hz")
        self.imu_rate_label.setText(f"IMU 包速率：{imu_rate:,.1f} Hz")

    def update_stats(self, stats: ParserStats) -> None:
        self.last_stats = stats; self.loss_metric.value.setText(f"{stats.lost_frames:,}")
        self.received_bytes_label.setText(f"接收字节：{stats.received_bytes:,}")
        self.total_frames_label.setText(f"有效帧：{stats.frames:,}")
        self.emg_frames_label.setText(f"EMG 帧：{stats.emg_frames:,}  ·  IMU 帧：{stats.imu_frames:,}")
        self.duplicate_label.setText(f"重复帧：{stats.duplicate_frames:,}")
        self.discarded_label.setText(f"丢弃字节：{stats.discarded_bytes:,}")

    def recent_raw(self, seconds: float = 5.0) -> np.ndarray:
        count = min(len(self._qc_raw), round(seconds * SAMPLING_RATE))
        return np.asarray(list(self._qc_raw)[-count:], dtype=np.int32)

    def render_plots(self) -> None:
        if not self.isVisible() or self.pause_button.isChecked() or not self.emg_time: return
        seconds = self.window_spin.value(); emg_x = np.asarray(self.emg_time, dtype=np.float64); emg_x -= emg_x[-1]
        start = int(np.searchsorted(emg_x, -seconds))
        for plot_card, values in zip(self.emg_cards, self.emg_values):
            if not plot_card.visible_box.isChecked() or not values or not self._card_is_in_viewport(plot_card): continue
            y = np.asarray(values, dtype=np.float64)[start:]
            x_draw, y_draw = self._peak_preserving_decimate(emg_x[start:], y, max(400, int(plot_card.plot.width()) * 2))
            plot_card.curves[0].setData(x_draw, y_draw, skipFiniteCheck=True)
            plot_card.value_label.setText(f"{y[-1]:,.0f} μV")
        if self.imu_time:
            imu_x = np.asarray(self.imu_time, dtype=np.float64); imu_x -= imu_x[-1]
            imu_start = int(np.searchsorted(imu_x, -seconds))
            for plot_card, sources, decimals in ((self.gyro_card, self.gyro_values, 3), (self.accel_card, self.accel_values, 3)):
                if not self._card_is_in_viewport(plot_card): continue
                for curve, values in zip(plot_card.curves, sources):
                    y = np.asarray(values, dtype=np.float64)[imu_start:]
                    if not len(y): continue
                    x_draw, y_draw = self._peak_preserving_decimate(imu_x[imu_start:], y, max(400, int(plot_card.plot.width()) * 2))
                    curve.setData(x_draw, y_draw, skipFiniteCheck=True)
                if all(sources): plot_card.value_label.setText(" / ".join(f"{values[-1]:.{decimals}f}" for values in sources))
        if self._qc_raw:
            quality = SignalQualityMonitor().calculate(self.recent_raw(1.0))
            self.qc_label.setText("\n".join(f"CH{i+1}  {q.status:<7}  RMS {q.rms:,.0f}  P2P {q.peak_to_peak:,.0f}" for i, q in enumerate(quality)))

    @staticmethod
    def _peak_preserving_decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        count = y.size
        if count <= max_points or count < 4: return x, y
        bins = max(2, max_points // 2); step = int(np.ceil(count / bins)); padded = int(np.ceil(count / step) * step)
        padded_y = np.pad(y, (0, padded - count), constant_values=np.nan).reshape(-1, step)
        minimum = np.nanargmin(padded_y, axis=1); maximum = np.nanargmax(padded_y, axis=1); base = np.arange(padded_y.shape[0]) * step
        indices = np.column_stack((base + minimum, base + maximum)); indices.sort(axis=1); indices = indices.ravel(); indices = indices[indices < count]
        return x[indices], y[indices]

    def _card_is_in_viewport(self, plot_card: QWidget) -> bool:
        if not plot_card.isVisible(): return False
        viewport = self.dashboard_scroll.viewport(); top_left = plot_card.mapTo(viewport, QPoint(0, 0))
        return QRect(top_left, plot_card.size()).intersects(viewport.rect().adjusted(0, -80, 0, 80))

    def toggle_highpass(self, enabled: bool) -> None:
        self.highpass_enabled = enabled; self.clear_buffers(reset_filter=False)
        self.highpass_button.setText(f"EMG 组合滤波：{'开启' if enabled else '关闭'}")

    def update_display_window(self, seconds: int) -> None:
        emg_target = max(100, SAMPLING_RATE * int(seconds)); self.emg_time = deque(self.emg_time, maxlen=emg_target)
        self.emg_values = [deque(values, maxlen=emg_target) for values in self.emg_values]
        imu_target = max(500, 1000 * int(seconds)); self.imu_time = deque(self.imu_time, maxlen=imu_target)
        self.gyro_values = [deque(values, maxlen=imu_target) for values in self.gyro_values]
        self.accel_values = [deque(values, maxlen=imu_target) for values in self.accel_values]; self.update_axis_ranges()

    def update_axis_ranges(self, _value: int | None = None) -> None:
        seconds = self.window_spin.value(); limit = self.emg_range_spin.value()
        for plot_card in self.emg_cards:
            plot_card.plot.setXRange(-seconds, 0, padding=0); plot_card.plot.setYRange(-limit, limit, padding=0)
        self.gyro_card.plot.setXRange(-seconds, 0, padding=0); self.gyro_card.plot.setYRange(-5, 5, padding=0)
        self.accel_card.plot.setXRange(-seconds, 0, padding=0); self.accel_card.plot.setYRange(-20, 20, padding=0)

    def clear_buffers(self, _checked: bool = False, reset_filter: bool = True) -> None:
        if reset_filter: self._filter.reset()
        self.emg_time.clear(); self.imu_time.clear()
        for values in (*self.emg_values, *self.gyro_values, *self.accel_values): values.clear()
        for plot_card in (*self.emg_cards, self.gyro_card, self.accel_card):
            for curve in plot_card.curves: curve.clear()
            plot_card.value_label.setText("--")
