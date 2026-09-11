from __future__ import annotations

import json
import subprocess
import threading
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
from emgforce.inference.model_bundle import ModelBundle, discover_model_bundles
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

    def __init__(self, models_root: Path) -> None:
        super().__init__()
        self.models_root = Path(models_root)
        self.worker: RealtimeInferenceWorker | None = None
        self.replay_worker: OfflineReplayWorker | None = None
        self.remote_model_worker: RemoteModelWorker | None = None
        self.bundle: ModelBundle | None = None
        self._connected = False
        self._probability_bars: dict[str, QProgressBar] = {}
        self._probability_values: dict[str, QLabel] = {}
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
            "本地识别模型", "Conv-LSTM 与 MPF+TDS 模型可在同一上位机中安全切换；每次只加载一个后端")
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
            "实时识别", "固定延迟零相位预处理；校准只对本次电极佩戴有效")
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
            "九路事件概率", "论文在线阈值 0.50；50 ms 防抖，并对食指/中指执行 press-release 状态机")
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
        if previous:
            index = self.model_combo.findData(previous)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._on_model_combo_changed()

    def _on_model_combo_changed(self) -> None:
        path_str = self.model_combo.currentData()
        if not path_str:
            self.model_details.setText("--")
            self.model_status.setText(f"未在 {self.models_root} 找到完整模型包")
            self.load_model_button.setEnabled(False)
            return
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
        if not manifest_path.is_file():
            self.model_status.setText(f"选择的目录缺少 manifest.json：{selected}")
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
        self._update_live_buttons()
        if not connected and self.worker is not None:
            self.worker.pause_recognition()

    def ingest_emg(self, raw: np.ndarray, indices: np.ndarray) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.submit_emg(raw, indices)

    def start_calibration(self) -> None:
        if self.worker is None or not self._connected:
            self.live_status.setText("请先连接设备并加载模型")
            return
        seconds = self.calibration_seconds.value()
        self.calibration_progress.setRange(0, seconds * 2000)
        self.calibration_progress.setValue(0)
        self._show_calibration_guidance(0)
        self.worker.begin_calibration(seconds)

    def start_recognition(self) -> None:
        if self.worker is not None:
            self.worker.begin_recognition()

    def pause_recognition(self) -> None:
        if self.worker is not None:
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
        self.run_replay_button.setEnabled(True)
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
            self.current_gesture.setText(gesture_text)
            self.current_gesture_status.setText(status_text)
            self.current_gesture_status.setStyleSheet(
                f"font-size: 14px; font-weight: 600; padding: 4px 10px; border-radius: 6px; {badge_style}")
            self.gesture_icon.set_action(event.name)
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
            guidance_index = min(
                len(CALIBRATION_GUIDANCE) - 1,
                int((current / total) * len(CALIBRATION_GUIDANCE)),
            )
            self._show_calibration_guidance(guidance_index)

    def _calibration_finished(self, scale: float) -> None:
        self.calibration_progress.setValue(self.calibration_progress.maximum())
        self.calibration_instruction.setText("校准完成")
        self.calibration_detail.setText(
            f"本次佩戴缩放系数 {scale:.3f}，现在可以点击“开始实时识别”")
        self.calibration_icon.set_action("idle")
        self.start_button.setEnabled(self._connected)

    def _show_calibration_guidance(self, index: int) -> None:
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
        self.model_status.setText(f"模型或实时推理失败：{message}")
        self.live_status.setText(message)
        self.bundle = None
        self._stop_realtime_worker()
        self.load_model_button.setEnabled(self.model_combo.count() > 0)

    def _stop_realtime_worker(self) -> bool:
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
        self.calibrate_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        if not enabled:
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
