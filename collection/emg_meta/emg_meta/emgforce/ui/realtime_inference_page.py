from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from emgforce.algorithms import ALGORITHMS, META_CONV_LSTM, algorithm_display_name, get_algorithm
from emgforce.inference.engine import PredictionFrame
from emgforce.inference.analyze_live_diagnostic import analyze as analyze_live_diagnostic
from emgforce.inference.live_diagnostic import LiveDiagnosticRecorder
from emgforce.inference.model_bundle import ModelBundle, discover_model_bundles
from emgforce.inference.song_local import (
    DISPLAY as SONG_DISPLAY, SongLocalRuntime, SongModelInfo, discover_song_models, song_key_path,
)
from emgforce.inference.song_worker import SongRealtimeWorker
from emgforce.inference.unibo_adapter import (
    DEFAULT_CHANNEL_MAP, MUSCLE_NAMES, POSTURE_NAMES, UniBoModelInfo,
    UniBoRealtimeWorker, discover_unibo_models, unibo_key_path,
)
from emgforce.inference.worker import OfflineReplayWorker, RealtimeInferenceWorker
from emgforce.transfer.dataset_upload import CommandResult
from emgforce.transfer.remote_models import (
    RemoteModelInfo, download_remote_model, list_remote_models,
)


CALIBRATION_GUIDANCE = (
    ("自然静息", "手掌自然放松，暂时不要做动作", "rest"),
    ("拇指轻点", "用拇指连续轻点 2 次，然后自然放松", "thumb_click"),
    ("拇指左滑", "用拇指向左滑动 2 次，然后自然放松", "thumb_in"),
    ("拇指右滑", "用拇指向右滑动 2 次，然后自然放松", "thumb_out"),
    ("拇指上滑", "用拇指向上滑动 2 次，然后自然放松", "thumb_up"),
    ("拇指下滑", "用拇指向下滑动 2 次，然后自然放松", "thumb_down"),
    ("食指保持", "食指按住约 1 秒后松开", "index_press"),
    ("中指保持", "中指按住约 1 秒后松开", "middle_press"),
)


class GestureIconCanvas(QWidget):
    """Vector-rendered gesture icon matching the experiment prompt design."""

    def __init__(self, parent: QWidget | None = None, size: int = 120, hand: str = "right") -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.action = "idle"
        self.hand = str(hand or "right")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;"
        )

    def set_hand(self, hand: str) -> None:
        self.hand = str(hand or "right")
        self.update()

    def set_action(self, action: str) -> None:
        self.action = str(action or "idle")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        center = QPointF(rect.center())
        w = rect.width()
        scale = w / 260.0

        action = self.action
        is_left = (self.hand == "left")
        if action in {"thumb_click", "thumb_tap", "click"}:
            self._draw_click_marker(painter, center, scale)
        elif action in {"thumb_up", "thumb_swipe_up"}:
            self._draw_arrow(painter, center, (0, -1), scale, QColor("#2563eb"))
        elif action in {"thumb_down", "thumb_swipe_down"}:
            self._draw_arrow(painter, center, (0, 1), scale, QColor("#2563eb"))
        elif action in {"thumb_in"}:
            direction = (1, 0) if is_left else (-1, 0)
            self._draw_arrow(painter, center, direction, scale, QColor("#2563eb"))
        elif action in {"thumb_out"}:
            direction = (-1, 0) if is_left else (1, 0)
            self._draw_arrow(painter, center, direction, scale, QColor("#2563eb"))
        elif action in {"thumb_swipe_left", "left_swipe"}:
            self._draw_arrow(painter, center, (-1, 0), scale, QColor("#2563eb"))
        elif action in {"thumb_swipe_right", "right_swipe"}:
            self._draw_arrow(painter, center, (1, 0), scale, QColor("#2563eb"))
        elif action in {"index_press", "index_hold", "index_pinch"}:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ef1741"))
            painter.drawEllipse(center, 54 * scale, 54 * scale)
        elif action in {"index_release"}:
            painter.setPen(QPen(QColor("#ef1741"), 12 * scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 48 * scale, 48 * scale)
        elif action in {"middle_press", "middle_hold", "middle_pinch"}:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#7c3aed"))
            painter.drawEllipse(center, 54 * scale, 54 * scale)
        elif action in {"middle_release"}:
            painter.setPen(QPen(QColor("#7c3aed"), 12 * scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 48 * scale, 48 * scale)
        elif action in {"rest", "idle"}:
            painter.setPen(QPen(QColor("#cbd5e1"), 4 * scale, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 42 * scale, 42 * scale)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#94a3b8"))
            painter.drawEllipse(center, 12 * scale, 12 * scale)
        else:
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Microsoft YaHei UI", max(9, int(13 * scale)), QFont.Weight.Medium))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, action)

    @staticmethod
    def _draw_click_marker(painter: QPainter, center: QPointF, scale: float) -> None:
        outer = QColor("#93c5fd")
        inner = QColor("#2563eb")
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(outer, 9 * scale))
        painter.drawEllipse(center, 78 * scale, 78 * scale)
        painter.setPen(QPen(inner, 11 * scale))
        painter.drawEllipse(center, 48 * scale, 48 * scale)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner)
        painter.drawEllipse(center, 18 * scale, 18 * scale)

    @staticmethod
    def _draw_arrow(painter: QPainter, center: QPointF, direction: tuple[int, int],
                    scale: float, color: QColor) -> None:
        dx, dy = direction
        px, py = -dy, dx
        local_points = (
            (-82, -20), (16, -20), (16, -54), (94, 0),
            (16, 54), (16, 20), (-82, 20),
        )
        polygon = QPolygonF([
            QPointF(center.x() + (along * dx + across * px) * scale,
                    center.y() + (along * dy + across * py) * scale)
            for along, across in local_points
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(polygon)


class RemoteModelWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action: str, host: str, repository: str,
                 models_root: Path, model: RemoteModelInfo | None = None,
                 algorithm_id: str = META_CONV_LSTM) -> None:
        super().__init__()
        self.action = action
        self.host = host
        self.repository = repository
        self.models_root = Path(models_root)
        self.model = model
        self.algorithm_id = algorithm_id
        self._cancelled = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def run(self) -> None:
        try:
            if self.action == "list":
                result = list_remote_models(
                    self.host, self.repository, runner=self._run_command,
                    algorithm_id=self.algorithm_id)
            elif self.action == "download" and self.model is not None:
                result = download_remote_model(
                    self.model, self.host, self.models_root,
                    self.repository, runner=self._run_command)
            else:
                raise ValueError("缺少服务器模型或操作类型错误")
            if not self._cancelled.is_set():
                self.succeeded.emit(result)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self._cancelled.is_set():
                self.failed.emit(str(exc))

    def _run_command(self, args: list[str], timeout: float) -> CommandResult:
        if self._cancelled.is_set():
            raise InterruptedError
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", creationflags=flags,
        )
        with self._process_lock:
            self._process = process
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(f"远端模型操作超时：{args[0]}")
        finally:
            with self._process_lock:
                self._process = None
        if self._cancelled.is_set():
            raise InterruptedError
        return CommandResult(process.returncode, stdout, stderr)


class RealtimeInferencePage(QWidget):
    """Load a local model bundle and run replay or fixed-lag live inference."""

    music_gesture_ready = Signal(int, float)

    def __init__(self, models_root: Path, *, prefer_song_spd: bool = False) -> None:
        super().__init__()
        self.models_root = Path(models_root)
        self.prefer_song_spd = bool(prefer_song_spd)
        self.worker: RealtimeInferenceWorker | UniBoRealtimeWorker | SongRealtimeWorker | None = None
        self.replay_worker: OfflineReplayWorker | None = None
        self.remote_model_worker: RemoteModelWorker | None = None
        self.bundle: ModelBundle | None = None
        self._diagnostic: LiveDiagnosticRecorder | None = None
        self._calibration_timing: tuple[float, int, str, str] | None = None
        self._last_live_sample_index: int | None = None
        self._connected = False
        self._probability_bars: dict[str, QProgressBar] = {}
        self._probability_values: dict[str, QLabel] = {}
        self._unibo_models: dict[str, UniBoModelInfo] = {}
        self._song_models: dict[str, SongModelInfo] = {}
        self._gesture_reset_timer = QTimer(self)
        self._gesture_reset_timer.setSingleShot(True)
        self._gesture_reset_timer.timeout.connect(self._reset_current_gesture)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(22, 18, 22, 24)
        page.setSpacing(16)

        model_card, model_layout = self._make_card(
            "本地识别模型", "支持 Conv-LSTM、MPF+TDS、UniBo 适配和 Song 真实 8 通道实验模型；每次只加载一个后端")
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(40)
        self.refresh_models_button = QPushButton("刷新模型")
        self.browse_model_button = QPushButton("浏览外部...")
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.setObjectName("primary")
        for button in (self.refresh_models_button, self.browse_model_button, self.load_model_button):
            button.setMinimumHeight(40)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.refresh_models_button)
        model_row.addWidget(self.browse_model_button)
        model_row.addWidget(self.load_model_button)
        model_layout.addLayout(model_row)
        self.model_status = QLabel("尚未扫描模型")
        self.model_status.setWordWrap(True)
        self.model_status.setObjectName("dataStatusIdle")
        self.model_details = QLabel("--")
        self.model_details.setWordWrap(True)
        self.model_details.setObjectName("muted")
        model_layout.addWidget(self.model_status)
        model_layout.addWidget(self.model_details)
        self.unibo_adapter_panel = QFrame()
        self.unibo_adapter_panel.setObjectName("qcMetricNeutral")
        unibo_grid = QGridLayout(self.unibo_adapter_panel)
        unibo_grid.setContentsMargins(14, 10, 14, 10)
        unibo_grid.setHorizontalSpacing(10)
        unibo_title = QLabel("UniBo 肌肉映射（实验适配）")
        unibo_title.setStyleSheet("font-weight: 700; color: #1d4ed8;")
        unibo_grid.addWidget(unibo_title, 0, 0, 1, 2)
        self.unibo_channel_combos: list[QComboBox] = []
        for column, (muscle, default_channel) in enumerate(zip(MUSCLE_NAMES, DEFAULT_CHANNEL_MAP)):
            combo = QComboBox()
            for channel in range(8):
                combo.addItem(f"CH{channel + 1}", channel)
            combo.setCurrentIndex(default_channel)
            combo.setMinimumHeight(36)
            self.unibo_channel_combos.append(combo)
            unibo_grid.addWidget(QLabel(muscle), 1, column * 2)
            unibo_grid.addWidget(combo, 1, column * 2 + 1)
        self.unibo_posture = QComboBox()
        for posture, name in POSTURE_NAMES.items():
            self.unibo_posture.addItem(name, posture)
        self.unibo_posture.setMinimumHeight(36)
        unibo_grid.addWidget(QLabel("E6b 姿态"), 2, 0)
        unibo_grid.addWidget(self.unibo_posture, 2, 1, 1, 3)
        warning = QLabel(
            "当前腕带环形电极与 UniBo 的 ECU/EDC/FCR/FCU 解剖位置不同；"
            "这里会先做 250→200 Hz 抗混叠重采样，再按上方映射推理。结果用于现场试验，不等同于 UniBo Day 6 验证指标。")
        warning.setWordWrap(True)
        warning.setObjectName("muted")
        unibo_grid.addWidget(warning, 3, 0, 1, 8)
        self.unibo_adapter_panel.setVisible(False)
        model_layout.addWidget(self.unibo_adapter_panel)
        page.addWidget(model_card)

        remote_card, remote_layout = self._make_card(
            "服务器训练模型",
            "按算法读取独立训练仓库，下载模型包后在本机校验 SHA-256")
        remote_settings = QGridLayout()
        remote_settings.setHorizontalSpacing(12)
        remote_settings.setVerticalSpacing(8)
        self.remote_host = QLineEdit("my-gpu-server")
        self.remote_algorithm = QComboBox()
        for spec in ALGORITHMS.values():
            self.remote_algorithm.addItem(spec.display_name, spec.algorithm_id)
        self.remote_repository = QLineEdit(
            "/home/qxy/qxy/generic-neuromotor-interface")
        for row, (title, widget) in enumerate((
                ("训练算法", self.remote_algorithm),
                ("SSH 主机", self.remote_host),
                ("训练仓库", self.remote_repository))):
            label = QLabel(title)
            label.setObjectName("fieldLabel")
            widget.setMinimumHeight(38)
            remote_settings.addWidget(label, row, 0)
            remote_settings.addWidget(widget, row, 1)
        remote_settings.setColumnStretch(1, 1)
        remote_layout.addLayout(remote_settings)
        remote_row = QHBoxLayout()
        self.remote_model_combo = QComboBox()
        self.remote_model_combo.setMinimumHeight(40)
        self.remote_model_combo.setPlaceholderText("点击刷新读取服务器最佳模型")
        self.refresh_remote_button = QPushButton("刷新服务器模型")
        self.download_remote_button = QPushButton("下载所选模型")
        self.download_remote_button.setObjectName("primary")
        self.download_remote_button.setEnabled(False)
        for button in (self.refresh_remote_button, self.download_remote_button):
            button.setMinimumHeight(40)
        remote_row.addWidget(self.remote_model_combo, 1)
        remote_row.addWidget(self.refresh_remote_button)
        remote_row.addWidget(self.download_remote_button)
        remote_layout.addLayout(remote_row)
        self.remote_model_status = QLabel("尚未连接服务器读取训练模型")
        self.remote_model_status.setWordWrap(True)
        self.remote_model_status.setObjectName("dataStatusIdle")
        remote_layout.addWidget(self.remote_model_status)
        page.addWidget(remote_card)

        live_card, live_layout = self._make_card(
            "实时识别", "按所选模型执行对应预处理；校准只对本次电极佩戴有效")
        settings = QGridLayout()
        settings.setHorizontalSpacing(12)
        settings.setVerticalSpacing(10)
        self.calibration_seconds = QSpinBox()
        self.calibration_seconds.setRange(5, 60)
        self.calibration_seconds.setValue(24)
        self.calibration_seconds.setSuffix(" 秒")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.50)
        self.hand_combo = QComboBox()
        self.hand_combo.addItem("右手 (Right)", "right")
        self.hand_combo.addItem("左手 (Left)", "left")
        for widget in (self.calibration_seconds, self.threshold, self.hand_combo):
            widget.setMinimumHeight(38)
        settings.addWidget(QLabel("校准时长"), 0, 0)
        settings.addWidget(self.calibration_seconds, 0, 1)
        settings.addWidget(QLabel("在线事件阈值"), 0, 2)
        settings.addWidget(self.threshold, 0, 3)
        settings.addWidget(QLabel("佩戴手臂"), 0, 4)
        settings.addWidget(self.hand_combo, 0, 5)
        settings.setColumnStretch(1, 1)
        settings.setColumnStretch(3, 1)
        settings.setColumnStretch(5, 1)
        live_layout.addLayout(settings)

        actions = QHBoxLayout()
        self.calibrate_button = QPushButton("开始推理校准")
        self.start_button = QPushButton("开始实时识别")
        self.start_button.setObjectName("primary")
        self.pause_button = QPushButton("暂停识别")
        for button in (self.calibrate_button, self.start_button, self.pause_button):
            button.setMinimumHeight(42)
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch()
        live_layout.addLayout(actions)
        diagnostic_row = QHBoxLayout()
        self.start_diagnostic_button = QPushButton("开始诊断记录")
        self.stop_diagnostic_button = QPushButton("结束诊断记录")
        self.start_diagnostic_button.setEnabled(False)
        self.stop_diagnostic_button.setEnabled(False)
        diagnostic_row.addWidget(self.start_diagnostic_button)
        diagnostic_row.addWidget(self.stop_diagnostic_button)
        diagnostic_row.addStretch()
        live_layout.addLayout(diagnostic_row)
        self.diagnostic_status = QLabel("按“开始诊断记录”保存本次原始 EMG/IMU 与逐帧概率；文件仅保存在本机 data/live_diagnostics")
        self.diagnostic_status.setWordWrap(True)
        self.diagnostic_status.setObjectName("muted")
        live_layout.addWidget(self.diagnostic_status)
        annotation_row = QHBoxLayout()
        self.annotation_action = QComboBox()
        self.mark_action_start_button = QPushButton("标记动作开始")
        self.mark_action_end_button = QPushButton("标记动作结束")
        self.mark_action_start_button.setEnabled(False)
        self.mark_action_end_button.setEnabled(False)
        annotation_row.addWidget(QLabel("当前尝试动作"))
        annotation_row.addWidget(self.annotation_action, 1)
        annotation_row.addWidget(self.mark_action_start_button)
        annotation_row.addWidget(self.mark_action_end_button)
        live_layout.addLayout(annotation_row)
        annotation_note = QLabel("手动标记只记录按键时刻，不能当作肌肉真实起止时间；用于之后对齐原始信号与预测。")
        annotation_note.setWordWrap(True)
        annotation_note.setObjectName("muted")
        live_layout.addWidget(annotation_note)
        guidance_box = QFrame()
        guidance_box.setObjectName("qcMetricNeutral")
        guidance_layout = QHBoxLayout(guidance_box)
        guidance_layout.setContentsMargins(18, 12, 18, 12)
        guidance_layout.setSpacing(16)
        self.calibration_icon = GestureIconCanvas(size=80)
        guidance_layout.addWidget(self.calibration_icon)
        guidance_text_layout = QVBoxLayout()
        guidance_text_layout.setSpacing(4)
        guidance_title = QLabel("校准动作提示")
        guidance_title.setObjectName("muted")
        self.calibration_instruction = QLabel("等待开始校准")
        self.calibration_instruction.setWordWrap(True)
        self.calibration_instruction.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #1d4ed8;")
        self.calibration_detail = QLabel(
            "开始后将按进度依次显示静息和七个目标动作")
        self.calibration_detail.setWordWrap(True)
        self.calibration_detail.setObjectName("muted")
        guidance_text_layout.addWidget(guidance_title)
        guidance_text_layout.addWidget(self.calibration_instruction)
        guidance_text_layout.addWidget(self.calibration_detail)
        guidance_layout.addLayout(guidance_text_layout, 1)
        live_layout.addWidget(guidance_box)
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setRange(0, 1)
        self.calibration_progress.setValue(0)
        self.live_status = QLabel("请先加载模型并连接设备")
        self.live_status.setWordWrap(True)
        self.live_status.setObjectName("dataStatusIdle")
        live_layout.addWidget(self.calibration_progress)
        live_layout.addWidget(self.live_status)
        page.addWidget(live_card)

        recognition_row = QHBoxLayout()
        recognition_row.setSpacing(16)

        current_card, current_layout = self._make_card(
            "当前识别事件", "展示当前触发的手势动作、状态机判定及实时延迟")
        current_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        current_box = QFrame()
        current_box.setObjectName("qcMetricNeutral")
        current_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        current_box_layout = QHBoxLayout(current_box)
        current_box_layout.setContentsMargins(22, 16, 22, 16)
        current_box_layout.setSpacing(22)
        self.gesture_icon = GestureIconCanvas(size=150)
        current_box_layout.addWidget(self.gesture_icon, 0, Qt.AlignmentFlag.AlignCenter)

        current_info = QVBoxLayout()
        current_info.setSpacing(8)
        current_info.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.current_gesture = QLabel("等待识别")
        self.current_gesture.setObjectName("inferenceGesture")
        self.current_gesture.setStyleSheet(
            "font-size: 30px; font-weight: 800; color: #1e293b;")
        self.current_gesture_status = QLabel("手部自然放松")
        self.current_gesture_status.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #475569; padding: 4px 10px; background-color: #f1f5f9; border-radius: 6px;")
        self.current_probability = QLabel("最高概率 -- · 推理耗时 --")
        self.current_probability.setObjectName("muted")
        self.current_probability.setStyleSheet("font-size: 13px; color: #64748b;")
        current_info.addWidget(self.current_gesture)
        current_info.addWidget(self.current_gesture_status)
        current_info.addWidget(self.current_probability)
        current_box_layout.addLayout(current_info, 1)
        current_layout.addWidget(current_box, 1)

        probability_card, probability_layout = self._make_card(
            "模型输出概率", "实时显示当前模型的全部输出；超过所选阈值时更新识别事件")
        probability_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.probability_grid = QGridLayout()
        self.probability_grid.setHorizontalSpacing(10)
        self.probability_grid.setVerticalSpacing(8)
        probability_layout.addLayout(self.probability_grid, 1)

        recognition_row.addWidget(current_card, 1)
        recognition_row.addWidget(probability_card, 1)
        page.addLayout(recognition_row)

        replay_card, replay_layout = self._make_card(
            "离线 HDF5 回放", "使用 meta_8ch_v1 对齐文件检查本地模型，不重新训练")
        replay_row = QHBoxLayout()
        self.replay_path = QLineEdit()
        self.replay_path.setPlaceholderText("选择 session_meta_aligned.hdf5")
        self.browse_replay_button = QPushButton("选择文件")
        self.run_replay_button = QPushButton("运行离线回放")
        for widget in (self.replay_path, self.browse_replay_button, self.run_replay_button):
            widget.setMinimumHeight(40)
        replay_row.addWidget(self.replay_path, 1)
        replay_row.addWidget(self.browse_replay_button)
        replay_row.addWidget(self.run_replay_button)
        replay_layout.addLayout(replay_row)
        self.replay_progress = QProgressBar()
        self.replay_progress.setRange(0, 1)
        self.replay_progress.setValue(0)
        self.replay_result = QLabel("尚未运行离线回放")
        self.replay_result.setWordWrap(True)
        self.replay_result.setObjectName("muted")
        replay_layout.addWidget(self.replay_progress)
        replay_layout.addWidget(self.replay_result)
        page.addWidget(replay_card)

        log_card, log_layout = self._make_card(
            "识别事件日志", "记录动作、概率和模型输出采样位置")
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(170)
        self.event_log.setPlaceholderText("尚无超过阈值的识别事件")
        log_layout.addWidget(self.event_log)
        page.addWidget(log_card)
        page.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)

        self.refresh_models_button.clicked.connect(self.refresh_models)
        self.browse_model_button.clicked.connect(self.browse_external_model)
        self.load_model_button.clicked.connect(self.load_selected_model)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.refresh_remote_button.clicked.connect(self.refresh_remote_models)
        self.download_remote_button.clicked.connect(self.download_selected_remote_model)
        self.remote_algorithm.currentIndexChanged.connect(self._remote_algorithm_changed)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.start_button.clicked.connect(self.start_recognition)
        self.pause_button.clicked.connect(self.pause_recognition)
        self.start_diagnostic_button.clicked.connect(self.start_diagnostic)
        self.stop_diagnostic_button.clicked.connect(self.stop_diagnostic)
        self.mark_action_start_button.clicked.connect(lambda: self._mark_diagnostic_action("start"))
        self.mark_action_end_button.clicked.connect(lambda: self._mark_diagnostic_action("end"))
        self.threshold.valueChanged.connect(self._threshold_changed)
        self.hand_combo.currentIndexChanged.connect(self._on_hand_changed)
        self.browse_replay_button.clicked.connect(self.browse_replay)
        self.run_replay_button.clicked.connect(self.run_replay)
        QTimer.singleShot(0, self.refresh_models)

    def refresh_models(self) -> None:
        previous = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for bundle in discover_model_bundles(self.models_root):
            training = bundle.metadata.get("training", {})
            source = bundle.metadata.get("source_checkpoint", {})
            algorithm = algorithm_display_name(bundle.algorithm_id, short=True)
            val = training.get("val_accuracy", training.get("val_event_macro_f1"))
            val_fnr = training.get("val_mean_fnr")
            cler = training.get("test_cler")
            val_text = (f"val FNR {float(val_fnr):.1%}" if val_fnr is not None else
                        (f"val {float(val):.1%}" if val is not None else "val --"))
            test_f1 = training.get("test_event_macro_f1")
            cler_text = (f"event-F1 {float(test_f1):.3f}" if test_f1 is not None else
                         (f"CLER {float(cler):.3f}" if cler is not None else "test --"))
            epoch = source.get("epoch", "--")
            trained_at = training.get("trained_at", "时间未知")
            label = (f"[{algorithm}] {bundle.model_id} · {trained_at} · epoch {epoch} · "
                     f"{val_text} · {cler_text}")
            self.model_combo.addItem(label, str(bundle.root))
        self._unibo_models = {
            info.combo_key: info for info in discover_unibo_models(self.models_root)
        }
        for info in self._unibo_models.values():
            accuracy = (f"ACC {info.validation_accuracy:.4f}"
                        if info.validation_accuracy is not None else "ACC --")
            macro_f1 = (f"macro-F1 {info.validation_macro_f1:.4f}"
                        if info.validation_macro_f1 is not None else "macro-F1 --")
            active_f1 = (f"active-F1 {info.validation_active_f1:.4f}"
                         if info.validation_active_f1 is not None else "active-F1 --")
            context = " · 需选择姿态" if info.posture_required else ""
            self.model_combo.addItem(
                f"[UniBo 4→8] {info.experiment} · {info.fold} · {accuracy} · "
                f"{macro_f1} · {active_f1}{context}",
                info.combo_key,
            )
        self._song_models = {info.combo_key: info for info in discover_song_models(self.models_root)}
        for info in self._song_models.values():
            self.model_combo.addItem(
                f"[Song 8ch] {info.model_id} · S03 val ACC {info.validation_accuracy:.3f} · "
                f"macro-F1 {info.validation_macro_f1:.3f} · 同人单日实验", info.combo_key)
        if previous:
            index = self.model_combo.findData(previous)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        elif self.prefer_song_spd:
            # The Song-specific launcher chooses a valid native 8-channel
            # bundle; a deliberate selection survives later refreshes.
            for info in self._song_models.values():
                try:
                    if SongLocalRuntime(info.directory).with_spd:
                        self.model_combo.setCurrentIndex(
                            self.model_combo.findData(info.combo_key))
                        break
                except (OSError, KeyError, TypeError, ValueError):
                    continue
        self.model_combo.blockSignals(False)
        self._on_model_combo_changed()

    def _on_model_combo_changed(self) -> None:
        path_str = self.model_combo.currentData()
        if not path_str:
            self.unibo_adapter_panel.setVisible(False)
            self.model_details.setText("--")
            self.model_status.setText(f"未在 {self.models_root} 找到完整模型包")
            self.load_model_button.setEnabled(False)
            return
        song_path = song_key_path(path_str)
        if song_path is not None:
            self.unibo_adapter_panel.setVisible(False)
            info = self._song_models.get(str(path_str))
            if info is None:
                try:
                    runtime = SongLocalRuntime(song_path)
                    info = SongModelInfo(song_path, runtime.manifest["model_id"],
                                         float(runtime.manifest["validation_trial_accuracy"]),
                                         float(runtime.manifest["validation_trial_macro_f1"]))
                except Exception as exc:
                    self.model_details.setText("--")
                    self.model_status.setText(f"Song 模型包无效：{exc}")
                    self.load_model_button.setEnabled(False)
                    return
            self.model_details.setText(
                f"Song 真实 8 通道 / 250 Hz / 200 ms / 四分类 · S03 试次 F1 {info.validation_macro_f1:.3f}\n"
                "S01/S02 训练；因果滤波；同一人同一天的探索性模型，连续实时准确率未验证")
            self.model_status.setText(f"已选择 Song 实验模型（点击「加载模型」）")
            self.load_model_button.setEnabled(True)
            return
        unibo_path = unibo_key_path(path_str)
        if unibo_path is not None:
            self.unibo_adapter_panel.setVisible(True)
            info = self._unibo_models.get(str(path_str))
            if info is None or not unibo_path.is_file():
                self.model_details.setText("--")
                self.model_status.setText(f"UniBo 模型文件不存在：{unibo_path}")
                self.load_model_button.setEnabled(False)
                return
            posture_text = "需要 E6b 姿态上下文" if info.posture_required else "不依赖姿态上下文"
            self.unibo_posture.setEnabled(info.posture_required)
            self.model_details.setText(
                f"UniBo {info.experiment} · {info.fold} · 原生 4 通道 / 200 Hz / 4 类别 · {posture_text}\n"
                f"现场输入：8 通道 / 250 Hz；默认 ECU/EDC/FCR/FCU = CH1/CH3/CH5/CH7\n"
                f"权重：{unibo_path}")
            is_loaded = self.bundle is not None and self.bundle.artifact.resolve() == unibo_path.resolve()
            self.model_status.setText(
                f"当前运行中模型：UniBo {info.experiment} 实验适配"
                if is_loaded else
                f"已选择 UniBo {info.experiment}（确认通道映射后点击「加载模型」）")
            self.load_model_button.setEnabled(True)
            return
        self.unibo_adapter_panel.setVisible(False)
        path = Path(path_str)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            self.model_details.setText("--")
            self.model_status.setText(f"选择的目录缺少 manifest.json：{path}")
            self.load_model_button.setEnabled(False)
            return
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_data = json.load(handle)
            artifact_meta = manifest_data.get("artifact", {})
            network = manifest_data.get("network", {})
            signal = manifest_data.get("signal", {})
            training = manifest_data.get("training", {})
            source_ckpt = manifest_data.get("source_checkpoint", {})
            epoch_info = f"epoch {source_ckpt.get('epoch', '--')}" if "epoch" in source_ckpt else "自定义权重"
            val_acc = training.get("val_accuracy")
            acc_str = f" · 验证准确率 {float(val_acc):.1%}" if val_acc is not None else ""
            val_fnr = training.get("val_mean_fnr")
            fnr_str = f" · 验证 FNR {float(val_fnr):.1%}" if val_fnr is not None else ""
            test_cler = training.get("test_cler")
            cler_str = f" · 测试 CLER {float(test_cler):.4f}" if test_cler is not None else ""
            test_fnr = training.get("test_mean_fnr")
            test_fnr_str = f" · 测试 FNR {float(test_fnr):.1%}" if test_fnr is not None else ""
            trained_at = training.get("trained_at", "训练时间未知")
            algorithm_id = manifest_data.get("algorithm_id", META_CONV_LSTM)
            algorithm_name = algorithm_display_name(algorithm_id, short=True)
            channels = signal.get("input_channels", network.get("input_channels", 8))
            rate = signal.get("sample_rate_hz", network.get("sample_rate_hz", 2000))
            outputs = network.get("output_channels", 9)
            checkpoint_path = source_ckpt.get("path", artifact_meta.get("filename", ""))
            digest = str(source_ckpt.get("sha256", artifact_meta.get("sha256", "")))[:12]
            digest_str = f" · SHA-256 {digest}…" if digest else ""
            self.model_details.setText(
                f"[{algorithm_name}] {trained_at} · {channels} 通道 / {rate} Hz / {outputs} 类别 · "
                f"{epoch_info}{acc_str}{fnr_str}{cler_str}{test_fnr_str}\n"
                f"源文件：{checkpoint_path}{digest_str}")
            model_id = str(manifest_data.get("model_id", path.name))
            is_loaded = (
                self.bundle is not None
                and self.bundle.root.resolve() == path.resolve()
            )
            if is_loaded:
                self.model_status.setText(f"当前运行中模型：[{algorithm_name}] {model_id}")
            else:
                self.model_status.setText(f"已选择模型：{model_id}（点击「加载模型」即可切换）")
            self.load_model_button.setEnabled(True)
        except Exception as exc:
            self.model_details.setText(f"读取模型配置失败：{exc}")
            self.load_model_button.setEnabled(False)

    def browse_external_model(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择模型包目录 (包含 manifest.json)", str(Path.cwd()))
        if not selected:
            return
        manifest_path = Path(selected) / "manifest.json"
        song_manifest = Path(selected) / "song_manifest.json"
        if not manifest_path.is_file() and not song_manifest.is_file():
            self.model_status.setText(f"选择的目录缺少模型清单：{selected}")
            return
        if song_manifest.is_file():
            self.refresh_models()
            song_key = f"song::{Path(selected).resolve()}"
            index = self.model_combo.findData(song_key)
            if index < 0:
                self.model_combo.addItem(f"[Song 8ch 外部] {Path(selected).name}", song_key)
                index = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(index)
            self._on_model_combo_changed()
            return
        label = f"[外部] {Path(selected).name}"
        index = self.model_combo.findData(selected)
        if index < 0:
            self.model_combo.addItem(label, selected)
            index = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(index)
        self._on_model_combo_changed()

    def refresh_remote_models(self) -> None:
        if self.remote_model_worker is not None:
            return
        self.remote_model_status.setText("正在读取服务器训练记录……")
        self._set_remote_status("dataStatusRunning")
        self._start_remote_model_worker(RemoteModelWorker(
            "list", self.remote_host.text(), self.remote_repository.text(),
            self.models_root, algorithm_id=str(self.remote_algorithm.currentData())))

    def _remote_algorithm_changed(self) -> None:
        spec = get_algorithm(str(self.remote_algorithm.currentData() or META_CONV_LSTM))
        self.remote_repository.setText(spec.remote_repository)
        self.remote_model_combo.clear()
        self.download_remote_button.setEnabled(False)
        self.remote_model_status.setText(f"已选择 {spec.display_name}，请刷新服务器模型")

    def download_selected_remote_model(self) -> None:
        if self.remote_model_worker is not None:
            return
        model = self.remote_model_combo.currentData()
        if not isinstance(model, RemoteModelInfo):
            self.remote_model_status.setText("请先刷新并选择一个服务器模型")
            self._set_remote_status("dataStatusBad")
            return
        self.remote_model_status.setText(
            f"正在导出并下载 {model.model_id}，请勿关闭程序……")
        self._set_remote_status("dataStatusRunning")
        self._start_remote_model_worker(RemoteModelWorker(
            "download", self.remote_host.text(), self.remote_repository.text(),
            self.models_root, model, model.algorithm_id))

    def load_selected_model(self) -> None:
        path = self.model_combo.currentData()
        if not path:
            return
        if not self._stop_realtime_worker():
            return
        self.model_status.setText("正在校验并加载模型……")
        self.load_model_button.setEnabled(False)
        song_path = song_key_path(path)
        unibo_path = unibo_key_path(path)
        if song_path is not None:
            worker = SongRealtimeWorker(song_path, parent=self)
        elif unibo_path is not None:
            channel_map = tuple(
                int(combo.currentData()) for combo in self.unibo_channel_combos)
            if len(set(channel_map)) != 4:
                self.model_status.setText("ECU/EDC/FCR/FCU 必须选择四个不同的设备通道")
                self.load_model_button.setEnabled(True)
                return
            worker = UniBoRealtimeWorker(
                unibo_path, channel_map,
                posture=int(self.unibo_posture.currentData()), parent=self)
        else:
            worker = RealtimeInferenceWorker(Path(path), self)
        self.worker = worker
        worker.model_loaded.connect(self._model_loaded)
        worker.status_changed.connect(self._live_status_changed)
        worker.failed.connect(self._worker_failed)
        worker.calibration_progress.connect(self._calibration_progress)
        worker.calibration_finished.connect(self._calibration_finished)
        worker.prediction_ready.connect(self._prediction_ready)
        worker.start()

    def set_connected(self, connected: bool) -> None:
        self._connected = bool(connected)
        if not connected:
            self.stop_diagnostic()
            self._finish_calibration_timing("device_disconnected")
        self._update_live_buttons()
        if not connected and self.worker is not None:
            self.worker.pause_recognition()

    def ingest_emg(self, raw: np.ndarray, indices: np.ndarray,
                   received_ns: np.ndarray | None = None) -> None:
        if len(indices):
            first_captured = self._diagnostic is not None and self._last_live_sample_index is None
            self._last_live_sample_index = int(indices[-1])
            if first_captured:
                self._update_live_buttons()
        if self._diagnostic is not None:
            try:
                self._diagnostic.record_emg(raw, indices, received_ns)
            except (OSError, ValueError, RuntimeError) as exc:
                self._diagnostic_failed(exc)
        if self.worker is not None and self.worker.isRunning():
            self.worker.submit_emg(raw, indices)

    def ingest_imu(self, gyro: np.ndarray, accel: np.ndarray, received_ns: np.ndarray) -> None:
        if self._diagnostic is not None:
            try:
                self._diagnostic.record_imu(gyro, accel, received_ns)
            except (OSError, ValueError, RuntimeError) as exc:
                self._diagnostic_failed(exc)

    def notify_packet_loss(self, lost: int, previous_sequence: int, sequence: int) -> None:
        if self._diagnostic is not None:
            self._diagnostic.record_packet_loss(lost)
        if isinstance(self.worker, SongRealtimeWorker) and self.worker.isRunning():
            self.worker.notify_packet_loss(lost)

    def start_diagnostic(self) -> None:
        if (self._diagnostic is not None or self.bundle is None or not self._connected or
                self.worker is None or not self.worker.isRunning()):
            return
        self._last_live_sample_index = None
        try:
            self._diagnostic = LiveDiagnosticRecorder(
                self.models_root.parent / "data" / "live_diagnostics",
                model_id=self.bundle.model_id, model_sha256=self.bundle.sha256,
                labels=self.bundle.labels, sample_rate_hz=self.bundle.sample_rate,
                threshold=self.threshold.value(), hand=str(self.hand_combo.currentData()))
        except (OSError, ValueError) as exc:
            self.diagnostic_status.setText(f"诊断记录无法启动：{exc}")
            return
        self.diagnostic_status.setText(f"正在诊断记录：{self._diagnostic.directory}")
        self._update_live_buttons()

    def stop_diagnostic(self) -> None:
        recorder = self._diagnostic
        if recorder is None:
            return
        self._diagnostic = None
        try:
            path = recorder.close()
        except (OSError, RuntimeError, ValueError) as exc:
            self.diagnostic_status.setText(f"诊断记录关闭异常，检查 {recorder.directory}：{exc}")
        else:
            try:
                summary = analyze_live_diagnostic(path)
            except (OSError, RuntimeError, ValueError) as exc:
                self.diagnostic_status.setText(f"诊断记录已保存到 {path}；自动分析失败：{exc}")
            else:
                neutral_share = summary["peak_label_fractions"].get("neutral")
                neutral_text = (f"；峰值静息 {neutral_share:.1%}" if neutral_share is not None else "")
                profile = summary["raw_signal_profile"]
                flat = max(profile["flat_one_second_windows_per_channel"])
                near_limit = max(profile["near_adc_limit_fraction_per_channel"] or [0])
                quality_text = (f"；平线整秒窗口最多 {flat} 个"
                                f"；近 ADC 上限占比最高 {near_limit:.1%}"
                                f"；设备报告丢包 {summary['reported_lost_packets']}")
                event_count = summary["event_intervals_with_predictions"]
                event_text = (f"；标记动作曾显示正确 {summary['event_intervals_ever_display_match']}/{event_count}"
                              if event_count else "")
                self.diagnostic_status.setText(
                    f"诊断记录与分析已保存：{path}；{summary['prediction_frames']} 帧预测"
                    f"{neutral_text}；{summary['manual_intervals']} 段人工标记{event_text}{quality_text}")
        self._update_live_buttons()

    def _diagnostic_failed(self, exc: Exception) -> None:
        self.stop_diagnostic()
        self.diagnostic_status.setText(f"诊断记录提前停止：{exc}")

    def _mark_diagnostic_action(self, event: str) -> None:
        if self._diagnostic is None:
            return
        try:
            self._diagnostic.record_annotation(
                str(self.annotation_action.currentData()), event, self._last_live_sample_index)
        except (OSError, ValueError, RuntimeError) as exc:
            self._diagnostic_failed(exc)

    def start_calibration(self) -> None:
        if self.worker is None or self.bundle is None or not self._connected:
            self.live_status.setText("请先连接设备并加载模型")
            return
        if self._calibration_timing is not None:
            self.live_status.setText("校准正在进行，请等待完成")
            return
        if self.bundle.metadata.get("song_real8_local"):
            self.live_status.setText("Song 模型当前使用零校准；可直接开始识别")
            return
        seconds = self.calibration_seconds.value()
        sample_rate = self.bundle.sample_rate
        self.calibration_progress.setRange(0, seconds * sample_rate)
        self.calibration_progress.setValue(0)
        self._show_calibration_guidance(0)
        self._calibration_timing = (
            time.monotonic(), seconds, str(self.bundle.model_id), str(self.bundle.sha256))
        self.worker.begin_calibration(seconds)

    def start_recognition(self) -> None:
        if self.worker is not None:
            self._finish_calibration_timing("recognition_started_before_completion")
            self.worker.begin_recognition()

    def pause_recognition(self) -> None:
        if self.worker is not None:
            self._finish_calibration_timing("paused_before_completion")
            self.worker.pause_recognition()

    def browse_replay(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择 Meta 对齐文件", str(Path.cwd()), "HDF5 (*.hdf5 *.h5)")
        if filename:
            self.replay_path.setText(filename)

    def run_replay(self) -> None:
        if self.bundle is None or self.replay_worker is not None:
            self.replay_result.setText("请先加载模型，或等待当前回放结束")
            return
        if self.bundle.metadata.get("experimental_adapter"):
            self.replay_result.setText("UniBo 适配器请使用实时 8 通道输入；该回放格式不兼容")
            return
        path = Path(self.replay_path.text().strip())
        if not path.is_file():
            self.replay_result.setText("请选择存在的 session_meta_aligned.hdf5")
            return
        worker = OfflineReplayWorker(
            self.bundle.root, path, threshold=0.35, parent=self)
        self.replay_worker = worker
        worker.progress.connect(self._replay_progress)
        worker.succeeded.connect(self._replay_succeeded)
        worker.failed.connect(self._replay_failed)
        worker.finished.connect(self._replay_finished)
        self.run_replay_button.setEnabled(False)
        self.replay_result.setText("正在进行本地离线推理……")
        worker.start()

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        self.stop_diagnostic()
        ok = True
        if self.remote_model_worker is not None:
            self.remote_model_worker.cancel()
            ok = self.remote_model_worker.wait(timeout_ms) and ok
            self.remote_model_worker = None
        if self.replay_worker is not None:
            self.replay_worker.stop()
            ok = self.replay_worker.wait(timeout_ms) and ok
            self.replay_worker = None
        worker = self.worker
        if worker is not None:
            worker.stop()
            ok = worker.wait(timeout_ms) and ok
            self.worker = None
        return ok

    def _start_remote_model_worker(self, worker: RemoteModelWorker) -> None:
        self.remote_model_worker = worker
        worker.succeeded.connect(self._remote_model_succeeded)
        worker.failed.connect(self._remote_model_failed)
        worker.finished.connect(self._remote_model_finished)
        self._set_remote_busy(True)
        worker.start()

    def _remote_model_succeeded(self, result: object) -> None:
        worker = self.remote_model_worker
        action = worker.action if worker is not None else ""
        if action == "list":
            models = result if isinstance(result, list) else []
            self.remote_model_combo.clear()
            for model in models:
                if not isinstance(model, RemoteModelInfo):
                    continue
                validation = model.val_accuracy if model.val_accuracy is not None else model.val_event_macro_f1
                val = (f"val FNR {model.val_mean_fnr:.1%}" if model.val_mean_fnr is not None else
                       (f"val {validation:.1%}" if validation is not None else "val --"))
                cler = (f"event-F1 {model.test_event_macro_f1:.3f}"
                        if model.test_event_macro_f1 is not None else
                        (f"CLER {model.test_cler:.3f}" if model.test_cler is not None else "test --"))
                size = model.size_bytes / (1024 ** 2)
                self.remote_model_combo.addItem(
                    f"{model.created_at} · epoch {model.epoch} · {val} · {cler} · {size:.0f} MB",
                    model,
                )
            self.download_remote_button.setEnabled(bool(models))
            self.remote_model_status.setText(
                f"服务器共有 {len(models)} 个可下载的最佳模型")
            self._set_remote_status("dataStatusGood" if models else "dataStatusIdle")
            return
        path = Path(str(result)).resolve()
        self.refresh_models()
        index = self.model_combo.findData(str(path))
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.remote_model_status.setText(f"模型已下载并校验：{path}")
        self._set_remote_status("dataStatusGood")

    def _remote_model_failed(self, message: str) -> None:
        self.remote_model_status.setText(f"服务器模型操作失败：{message}")
        self._set_remote_status("dataStatusBad")

    def _remote_model_finished(self) -> None:
        worker = self.remote_model_worker
        self.remote_model_worker = None
        self._set_remote_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _set_remote_busy(self, busy: bool) -> None:
        for widget in (
            self.remote_algorithm, self.remote_host, self.remote_repository, self.refresh_remote_button,
            self.remote_model_combo,
        ):
            widget.setEnabled(not busy)
        self.download_remote_button.setEnabled(
            not busy and self.remote_model_combo.count() > 0)

    def _set_remote_status(self, object_name: str) -> None:
        self.remote_model_status.setObjectName(object_name)
        self.remote_model_status.style().unpolish(self.remote_model_status)
        self.remote_model_status.style().polish(self.remote_model_status)

    def _model_loaded(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.annotation_action.clear()
        for label in bundle.labels:
            self.annotation_action.addItem(bundle.display_names.get(label, label), label)
        online_threshold = float(bundle.preprocessing.get(
            "online_event_threshold", 0.50))
        self.threshold.blockSignals(True)
        self.threshold.setValue(online_threshold)
        self.threshold.blockSignals(False)
        training = bundle.metadata.get("training", {})
        source_ckpt = bundle.metadata.get("source_checkpoint", {})
        epoch_info = f"epoch {source_ckpt.get('epoch', '--')}" if "epoch" in source_ckpt else "自定义权重"
        val_acc = training.get("val_accuracy")
        acc_str = f" · 验证准确率 {float(val_acc):.1%}" if val_acc is not None else ""
        val_fnr = training.get("val_mean_fnr")
        fnr_str = f" · 验证 FNR {float(val_fnr):.1%}" if val_fnr is not None else ""
        test_cler = training.get("test_cler")
        cler_str = f" · 测试 CLER {float(test_cler):.4f}" if test_cler is not None else ""
        test_fnr = training.get("test_mean_fnr")
        test_fnr_str = f" · 测试 FNR {float(test_fnr):.1%}" if test_fnr is not None else ""
        trained_at = training.get("trained_at", "训练时间未知")
        checkpoint_path = source_ckpt.get("path", bundle.artifact.name)
        digest = str(source_ckpt.get("sha256", bundle.sha256))[:12]
        algorithm_name = algorithm_display_name(bundle.algorithm_id, short=True)
        self.model_status.setText(
            f"模型加载成功：[{algorithm_name}] {bundle.model_id}；切换模型后必须重新校准")
        self.model_details.setText(
            f"{trained_at} · {bundle.input_channels} 通道 / {bundle.sample_rate} Hz / "
            f"{bundle.output_channels} 类别 · {epoch_info}{acc_str}{fnr_str}{cler_str}{test_fnr_str}\n"
            f"源文件：{checkpoint_path} · SHA-256 {digest}…")
        self._build_probability_rows(bundle)
        self._update_live_buttons()
        experimental = bool(bundle.metadata.get("experimental_adapter"))
        self.calibration_seconds.setValue(8 if experimental else 24)
        self.run_replay_button.setEnabled(not experimental)
        if experimental:
            if bundle.metadata.get("song_real8_local"):
                self.model_status.setText("Song 8 通道实验模型已校验；连接设备后可直接开始识别")
                self.model_details.setText(
                    f"250 Hz / 8 通道 / 200 ms 因果滤波四分类 · S03 验证 ACC {float(val_acc):.1%}\n"
                    "同人单日探索性模型；只验证过提示动作稳定区间，连续实时性能与延迟未验证")
                self.replay_result.setText("Song 原始 HDF5 回放请使用 benchmarks/song_real8_study.py 的因果模式")
                self.calibration_instruction.setText("零校准模型")
                self.calibration_detail.setText("此模型当前不使用现场校准；连接设备后直接开始识别")
                self._update_live_buttons()
                self.load_model_button.setEnabled(True)
                return
            mapping = bundle.metadata.get("channel_mapping", {})
            mapping_text = ", ".join(f"{key}={value}" for key, value in mapping.items())
            self.model_status.setText(
                f"模型加载成功：[{algorithm_name}] {bundle.model_id}；请先静息校准")
            self.model_details.setText(
                f"现场 8 通道 / 250 Hz → UniBo 4 通道 / 200 Hz · 四分类 · {mapping_text}\n"
                f"姿态：{bundle.metadata.get('posture_name')} · 源文件：{checkpoint_path} · "
                f"SHA-256 {digest}… · 实验域适配")
            self.replay_result.setText("UniBo 适配模型使用 200 ms 原始窗口；请通过上方实时识别验证")
        self.load_model_button.setEnabled(True)


    def _build_probability_rows(self, bundle: ModelBundle) -> None:
        while self.probability_grid.count():
            item = self.probability_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._probability_bars.clear()
        self._probability_values.clear()
        is_left = (self.hand_combo.currentData() == "left")
        for row, name in enumerate(bundle.labels):
            display_name = bundle.display_names.get(name, name)
            if name == "thumb_in":
                display_name = "拇指向内 (→)" if is_left else "拇指向内 (←)"
            elif name == "thumb_out":
                display_name = "拇指向外 (←)" if is_left else "拇指向外 (→)"
            label = QLabel(display_name)
            label.setMinimumWidth(130)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            value = QLabel("0.000")
            value.setMinimumWidth(52)
            self.probability_grid.addWidget(label, row, 0)
            self.probability_grid.addWidget(bar, row, 1)
            self.probability_grid.addWidget(value, row, 2)
            self.probability_grid.setColumnStretch(1, 1)
            self._probability_bars[name] = bar
            self._probability_values[name] = value

    def _on_hand_changed(self) -> None:
        hand = str(self.hand_combo.currentData() or "right")
        self.gesture_icon.set_hand(hand)
        self.calibration_icon.set_hand(hand)
        if self.bundle is not None:
            self._build_probability_rows(self.bundle)

    def _reset_current_gesture(self) -> None:
        self.gesture_icon.set_action("idle")
        self.current_gesture.setText("等待识别")
        self.current_gesture_status.setText("手部自然放松")
        self.current_gesture_status.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #475569; padding: 4px 10px; background-color: #f1f5f9; border-radius: 6px;")

    def _prediction_ready(self, frame: PredictionFrame) -> None:
        if self._diagnostic is not None:
            try:
                self._diagnostic.record_prediction(frame, self.threshold.value())
            except (OSError, ValueError, RuntimeError) as exc:
                self._diagnostic_failed(exc)
        for name, probability in zip(frame.labels, frame.probabilities):
            if name in self._probability_bars:
                self._probability_bars[name].setValue(round(float(probability) * 1000))
                self._probability_values[name].setText(f"{float(probability):.3f}")
        peak_index = int(np.argmax(frame.probabilities))
        peak = float(frame.probabilities[peak_index])
        # Only the new music label set may drive the music engine.  This keeps
        # legacy thumb models from accidentally controlling a direction layer.
        music_labels = {
            "forward": 1, "backward": 2, "left": 3, "right": 4,
            "up": 5, "down": 6, "index_pinch": 7,
        }
        required = set(music_labels)
        if required.issubset(set(frame.labels)):
            peak_name = frame.labels[peak_index]
            gesture = music_labels.get(peak_name, 0) if peak >= self.threshold.value() else 0
            self.music_gesture_ready.emit(gesture, peak if gesture else 1.0)
        else:
            self.music_gesture_ready.emit(0, 1.0)
        is_song = self.bundle is not None and self.bundle.metadata.get("song_real8_local")
        if is_song:
            self.current_probability.setText(
                f"最高概率 {peak:.3f} · 计算 {frame.inference_ms:.0f} ms · 200 ms 窗口 · 数据龄未测")
            self._gesture_reset_timer.stop()
            if frame.active_label in SONG_DISPLAY:
                label = frame.active_label
                confidence = float(frame.probabilities[frame.labels.index(label)])
                self.current_gesture.setText(SONG_DISPLAY[label])
                self.current_gesture_status.setText(f"当前识别 · 置信度 {confidence:.1%}")
                self.current_gesture_status.setStyleSheet(
                    "font-size: 14px; font-weight: 600; color: #1d4ed8; padding: 4px 10px; background-color: #dbeafe; border-radius: 6px;")
                self.gesture_icon.set_action("rest" if label == "neutral" else label)
            else:
                self._reset_current_gesture()
        else:
            self.current_probability.setText(
                f"最高概率 {peak:.3f} · 推理 {frame.inference_ms:.0f} ms"
                f" · 输出数据龄 {frame.output_age_ms:.0f} ms"
                f"（固定延迟 {frame.fixed_lag_ms:.0f} ms）")
        is_left = (self.hand_combo.currentData() == "left")
        for event in frame.events:
            display_name = event.display_name
            if event.name == "thumb_in":
                display_name = "拇指向内 (向右)" if is_left else "拇指向内 (向左)"
            elif event.name == "thumb_out":
                display_name = "拇指向外 (向左)" if is_left else "拇指向外 (向右)"

            if event.hold_valid is True:
                gesture_text = display_name
                status_text = "保持完成 ✓"
                badge_style = "color: #047857; background-color: #d1fae5;"
            elif event.hold_valid is False and not event.synthetic:
                gesture_text = display_name
                status_text = "过早释放 ✗"
                badge_style = "color: #b91c1c; background-color: #fee2e2;"
            else:
                gesture_text = display_name
                status_text = f"置信度 {event.probability:.1%}"
                badge_style = "color: #1d4ed8; background-color: #dbeafe;"
            if not is_song:
                self.current_gesture.setText(gesture_text)
                self.current_gesture_status.setText(status_text)
                self.current_gesture_status.setStyleSheet(
                    f"font-size: 14px; font-weight: 600; padding: 4px 10px; border-radius: 6px; {badge_style}")
                self.gesture_icon.set_action("rest" if event.name == "neutral" else event.name)
                self._gesture_reset_timer.start(2000)
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            hold = (f"  hold={event.hold_duration_seconds:.3f}s"
                    if event.hold_duration_seconds is not None else "")
            source = "  synthetic" if event.synthetic else ""
            self.event_log.appendPlainText(
                f"{stamp}  {display_name}  p={event.probability:.3f}"
                f"  sample={event.sample_index}{hold}{source}")

    def _calibration_progress(self, current: int, total: int) -> None:
        self.calibration_progress.setRange(0, max(1, total))
        self.calibration_progress.setValue(current)
        if total > 0:
            if self.bundle is not None and self.bundle.metadata.get("experimental_adapter"):
                self.calibration_instruction.setText("保持自然静息")
                self.calibration_detail.setText(
                    "手掌放松且暂时不要动作；正在估计四路肌肉映射的幅值增益")
                self.calibration_icon.set_action("rest")
                return
            guidance_index = min(
                len(CALIBRATION_GUIDANCE) - 1,
                int((current / total) * len(CALIBRATION_GUIDANCE)),
            )
            self._show_calibration_guidance(guidance_index)

    def _finish_calibration_timing(self, outcome: str, *, scale: float | None = None) -> tuple[float | None, bool]:
        active = self._calibration_timing
        self._calibration_timing = None
        if active is None:
            return None, False
        started, requested_seconds, model_id, model_sha256 = active
        elapsed = max(0.0, time.monotonic() - started)
        record = {
            "timestamp_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "model_id": model_id,
            "model_sha256": model_sha256,
            "requested_signal_seconds": requested_seconds,
            "click_to_outcome_seconds": elapsed,
            "outcome": outcome,
            "scale": float(scale) if scale is not None else None,
            "scope": "UI click to worker outcome on this computer; excludes electrode placement and setup",
        }
        try:
            self.models_root.mkdir(parents=True, exist_ok=True)
            with (self.models_root / "calibration_timings.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return elapsed, False
        return elapsed, True

    def _calibration_finished(self, scale: float) -> None:
        if self._calibration_timing is None:
            return  # A queued completion from an interrupted or unloaded worker.
        elapsed, saved = self._finish_calibration_timing("completed", scale=scale)
        self.calibration_progress.setValue(self.calibration_progress.maximum())
        self.calibration_instruction.setText("校准完成")
        if self.bundle is not None and self.bundle.metadata.get("experimental_adapter"):
            self.calibration_detail.setText(
                f"UniBo 四路几何平均转换增益 {scale:.6g}，现在可以点击“开始实时识别”")
        else:
            self.calibration_detail.setText(
                f"本次佩戴缩放系数 {scale:.3f}，现在可以点击“开始实时识别”")
        if elapsed is not None:
            suffix = f"；点击至完成 {elapsed:.1f} 秒"
            if not saved:
                suffix += "（本地计时记录未保存）"
            self.calibration_detail.setText(self.calibration_detail.text() + suffix)
        self.calibration_icon.set_action("idle")
        self.start_button.setEnabled(self._connected)

    def _show_calibration_guidance(self, index: int) -> None:
        if self.bundle is not None and self.bundle.metadata.get("experimental_adapter"):
            self.calibration_instruction.setText("保持自然静息")
            self.calibration_detail.setText("校准期间保持手掌自然放松，不要执行捏合、握拳或张手")
            self.calibration_icon.set_action("rest")
            return
        item = CALIBRATION_GUIDANCE[int(index)]
        title, detail = item[0], item[1]
        action = item[2] if len(item) > 2 else "idle"
        is_left = (self.hand_combo.currentData() == "left")
        if action == "thumb_in":
            title = "拇指向内"
            detail = "用拇指向内（向右）滑动 2 次，然后自然放松" if is_left else "用拇指向内（向左）滑动 2 次，然后自然放松"
        elif action == "thumb_out":
            title = "拇指向外"
            detail = "用拇指向外（向左）滑动 2 次，然后自然放松" if is_left else "用拇指向外（向右）滑动 2 次，然后自然放松"
        self.calibration_instruction.setText(title)
        self.calibration_detail.setText(
            f"{detail} · 当前步骤 {int(index) + 1}/{len(CALIBRATION_GUIDANCE)}")
        self.calibration_icon.set_action(action)

    def _threshold_changed(self, value: float) -> None:
        if self.worker is not None:
            self.worker.set_threshold(value)

    def _live_status_changed(self, message: str) -> None:
        self.live_status.setText(message)

    def _worker_failed(self, message: str) -> None:
        self._finish_calibration_timing("worker_failed")
        self.model_status.setText(f"模型或实时推理失败：{message}")
        self.live_status.setText(message)
        self.bundle = None
        self._stop_realtime_worker()
        self.load_model_button.setEnabled(self.model_combo.count() > 0)

    def _stop_realtime_worker(self) -> bool:
        self._finish_calibration_timing("model_unloaded")
        self.stop_diagnostic()
        worker = self.worker
        if worker is None:
            return True
        worker.stop()
        if not worker.wait(10000):
            self.model_status.setText("旧模型仍在退出，暂不能切换")
            return False
        worker.deleteLater()
        self.worker = None
        self.bundle = None
        self.event_log.clear()
        self.calibration_progress.setValue(0)
        return True

    def _update_live_buttons(self) -> None:
        enabled = self.bundle is not None and self._connected
        self.start_diagnostic_button.setEnabled(enabled and self._diagnostic is None)
        self.stop_diagnostic_button.setEnabled(self._diagnostic is not None)
        can_annotate = self._diagnostic is not None and self._last_live_sample_index is not None
        self.mark_action_start_button.setEnabled(can_annotate)
        self.mark_action_end_button.setEnabled(can_annotate)
        is_song = enabled and bool(self.bundle.metadata.get("song_real8_local"))
        self.calibrate_button.setEnabled(enabled and not is_song)
        self.pause_button.setEnabled(enabled)
        if is_song:
            self.start_button.setEnabled(True)
        elif not enabled:
            self.start_button.setEnabled(False)

    def _replay_progress(self, current: int, total: int) -> None:
        self.replay_progress.setRange(0, max(1, total))
        self.replay_progress.setValue(current)

    def _replay_succeeded(self, result: dict) -> None:
        assert self.bundle is not None
        peaks = result["peak_probabilities"]
        counts = result["event_counts"]
        strongest = sorted(peaks, key=peaks.get, reverse=True)[:3]
        peak_text = "，".join(
            f"{self.bundle.display_names.get(name, name)} {peaks[name]:.3f}"
            for name in strongest)
        total_events = sum(counts.values())
        evaluation = result.get("evaluation", {})
        summary = (
            f"回放完成：{result['duration_seconds']:.1f} 秒，当前阈值 {result['threshold']:.2f}，"
            f"状态机后检测 {total_events} 个事件；最高概率：{peak_text}")
        detail_lines: list[str] = []
        if "cler" in evaluation:
            detail_lines.append(f"CLER {float(evaluation['cler']):.3f}")
        if "mean_fnr" in evaluation:
            totals = evaluation.get("ground_truth_counts", {})
            correct = evaluation.get("correct_counts", {})
            truth_total = sum(int(value) for value in totals.values())
            correct_total = sum(int(value) for value in correct.values())
            detail_lines.append(
                f"Mean FNR {float(evaluation['mean_fnr']):.3f}；"
                f"真值 {truth_total}，正确匹配 {correct_total}")

            per_class = evaluation.get("per_class_fnr", {})
            class_parts = []
            for name in self.bundle.labels:
                total = int(totals.get(name, 0))
                if not total:
                    continue
                display = self.bundle.display_names.get(name, name)
                class_parts.append(
                    f"{display} {int(correct.get(name, 0))}/{total} "
                    f"(FNR {float(per_class.get(name, 0.0)):.2f})")
            if class_parts:
                detail_lines.append("分类别：" + "；".join(class_parts))

            recommended = evaluation.get("recommended_threshold")
            class_thresholds = evaluation.get("recommended_class_thresholds", {})
            if recommended is not None:
                detail_lines.append(f"推荐统一阈值：{float(recommended):.2f}")
            if class_thresholds:
                detail_lines.append("推荐分类别阈值：" + "；".join(
                    f"{self.bundle.display_names.get(name, name)} "
                    f"{float(class_thresholds[name]):.2f}"
                    for name in self.bundle.labels if name in class_thresholds))

            sweep = evaluation.get("threshold_sweep", [])
            if sweep:
                detail_lines.append("阈值扫描（阈值/F1/FNR/误报每分钟）：" + "；".join(
                    f"{float(row['threshold']):.2f}/"
                    f"{float(row['macro_f1']):.2f}/"
                    f"{float(row['mean_fnr']):.2f}/"
                    f"{float(row['false_positives_per_minute']):.1f}"
                    for row in sweep))
        if not evaluation:
            detail_lines.append("文件中没有可用 prompts，未计算 CLER/FNR")
        self.replay_result.setText("\n".join([summary, *detail_lines]))
        self.replay_progress.setValue(self.replay_progress.maximum())

    def _replay_failed(self, message: str) -> None:
        self.replay_result.setText(f"离线回放失败：{message}")

    def _replay_finished(self) -> None:
        worker = self.replay_worker
        self.replay_worker = None
        self.run_replay_button.setEnabled(self.bundle is not None)
        if worker is not None:
            worker.deleteLater()

    @staticmethod
    def _make_card(title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(11)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        description_label = QLabel(description)
        description_label.setObjectName("cardDescription")
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return frame, layout
