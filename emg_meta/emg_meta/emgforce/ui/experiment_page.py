from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from emgforce.experiment.models import ParticipantInfo, SessionInfo
from emgforce.experiment.prompt_engine import PromptState
from emgforce.experiment.protocol_loader import ProtocolLoader
from emgforce.quality.monitor import SignalQualityMonitor

from .prompt_window import ParticipantPromptWindow


class ExperimentPage(QWidget):
    start_requested = Signal(object, object, object)
    stop_requested = Signal()
    pause_requested = Signal(); resume_requested = Signal(); skip_requested = Signal(); repeat_requested = Signal()
    bad_requested = Signal(str, str); manual_mark_requested = Signal(str)
    new_donning_requested = Signal(); new_stage_requested = Signal(str)

    def __init__(self, protocol_dir: Path) -> None:
        super().__init__(); self.setObjectName("experimentPage")
        self.loader = ProtocolLoader(protocol_dir); self.protocols = {}
        self._was_running = False

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setObjectName("experimentScroll")
        scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(); content.setObjectName("experimentContent")
        page = QVBoxLayout(content); page.setContentsMargins(22, 18, 22, 24); page.setSpacing(16)

        workspace = QHBoxLayout(); workspace.setSpacing(16)
        left_container = QWidget(); left_container.setObjectName("experimentLeftColumn")
        left_container.setMinimumWidth(500); left_container.setMaximumWidth(570)
        left_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        left_column = QVBoxLayout(left_container)
        left_column.setContentsMargins(0, 0, 0, 0); left_column.setSpacing(14)
        participant_box, participant = self._make_card("参与者信息", "建立参与者身份和惯用手信息", grid=True)
        self.participant_id = QLineEdit(); self.dominant = QComboBox()
        for text, value in (("未填写", ""), ("右手", "right"),
                            ("左手", "left"), ("双利手", "ambidextrous")):
            self.dominant.addItem(text, value)
        self.participant_id.setPlaceholderText("例如：P001")
        self._add_field(participant, 0, "参与者编号 *", self.participant_id)
        self._add_field(participant, 1, "惯用手", self.dominant)

        session_box, session = self._make_card("实验设置", "确定场次、数据集、流程和当前实验条件", grid=True)
        self.session_id = QLineEdit("S01"); self.experiment_name = QLineEdit("手势数据集_v1")
        self.protocol = QComboBox(); self.stage_name = QLineEdit("默认")
        self.dataset_split = QComboBox()
        for text, value in (("自动（S01–S08 训练，S09–S10 验证，S11–S12 测试）", "auto"),
                            ("训练集 train", "train"), ("验证集 val", "val"),
                            ("测试集 test", "test")):
            self.dataset_split.addItem(text, value)
        self.session_id.setPlaceholderText("例如：S01")
        self.experiment_name.setPlaceholderText("例如：手势数据集_v1")
        self.stage_name.setPlaceholderText("例如：默认、掌心向上")
        for row, (field_title, widget) in enumerate((("场次编号 *", self.session_id),
                ("实验名称 *", self.experiment_name), ("实验协议", self.protocol),
                ("数据集划分", self.dataset_split),
                ("当前阶段", self.stage_name))):
            self._add_field(session, row, field_title, widget)
        left_column.addWidget(participant_box); left_column.addWidget(session_box)
        self.reload_protocols()

        check_box, check = self._make_card("信号质量检查", "分析最近 5 秒肌电数据，仅用于提示电极接触质量")
        self.signal_result = QLabel("尚未检查"); self.signal_result.setObjectName("signalResult")
        self.signal_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_grid_widget = QWidget(); quality_grid = QGridLayout(self.quality_grid_widget)
        quality_grid.setContentsMargins(0, 0, 0, 0); quality_grid.setHorizontalSpacing(8); quality_grid.setVerticalSpacing(8)
        self.quality_cards: list[tuple[QFrame, QLabel, QLabel]] = []
        for channel in range(8):
            quality_card = QFrame(); quality_card.setObjectName("qualityWaiting")
            quality_card.setFixedHeight(78)
            quality_layout = QVBoxLayout(quality_card)
            quality_layout.setContentsMargins(8, 5, 8, 5); quality_layout.setSpacing(1)
            channel_label = QLabel(f"通道 {channel + 1}"); channel_label.setObjectName("qualityChannel")
            status_label = QLabel("等待检查"); status_label.setObjectName("qualityStatus")
            metrics_label = QLabel("RMS --\n峰峰值 --"); metrics_label.setObjectName("qualityMetrics")
            quality_layout.addWidget(channel_label); quality_layout.addWidget(status_label); quality_layout.addWidget(metrics_label)
            quality_grid.addWidget(quality_card, channel // 4, channel % 4)
            quality_grid.setColumnStretch(channel % 4, 1)
            self.quality_cards.append((quality_card, status_label, metrics_label))
        self.quality_grid_widget.hide()
        self.signal_check = QPushButton("执行信号检查"); self.continue_anyway = QPushButton("忽略警告并继续")
        self.signal_check.setObjectName("primary")
        result_area = QVBoxLayout(); result_area.addWidget(self.signal_result); result_area.addWidget(self.quality_grid_widget)
        action_area = QHBoxLayout(); action_area.setSpacing(8)
        self.signal_check.setMinimumHeight(38); self.continue_anyway.setMinimumHeight(38)
        action_area.addWidget(self.signal_check); action_area.addWidget(self.continue_anyway); action_area.addStretch()
        check.addLayout(result_area); check.addLayout(action_area)
        left_column.addWidget(check_box); left_column.addStretch()

        panel, panel_layout = self._make_card("离散网格导航任务", "使用离散手势沿节点序列导航，并在彩色节点完成激活动作")
        self.prompt_panel = ParticipantPromptWindow(self)
        panel_layout.addWidget(self.prompt_panel)

        primary_actions = QHBoxLayout(); primary_actions.setSpacing(8)
        self.start = QPushButton("开始实验"); self.start.setObjectName("record")
        self.pause = QPushButton("暂停"); self.resume = QPushButton("继续"); self.skip = QPushButton("跳过试次"); self.repeat = QPushButton("重复试次")
        self.bad = QPushButton("标记异常"); self.manual = QPushButton("手动标记"); self.stop = QPushButton("停止实验"); self.stop.setObjectName("danger")
        action_buttons = (self.start, self.pause, self.resume, self.skip, self.repeat, self.bad, self.manual, self.stop)
        for button in action_buttons:
            button.setMinimumHeight(40); button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            primary_actions.addWidget(button, 1)
        primary_actions.setAlignment(Qt.AlignmentFlag.AlignLeft)
        panel_layout.addLayout(primary_actions)

        divider = QFrame(); divider.setObjectName("divider"); divider.setFrameShape(QFrame.Shape.HLine)
        panel_layout.addWidget(divider)
        conditions = QHBoxLayout(); conditions.setSpacing(10)
        condition_label = QLabel("实验条件"); condition_label.setObjectName("muted")
        self.new_donning = QPushButton("结束当前佩戴 / 开始新佩戴"); self.new_stage = QPushButton("结束当前阶段 / 开始新阶段")
        conditions.addWidget(condition_label); conditions.addWidget(self.new_donning); conditions.addWidget(self.new_stage); conditions.addStretch()
        panel_layout.addLayout(conditions)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        workspace.addWidget(left_container, 0, Qt.AlignmentFlag.AlignTop)
        workspace.addWidget(panel, 1, Qt.AlignmentFlag.AlignTop)
        page.addLayout(workspace); page.addStretch()

        scroll.setWidget(content); root.addWidget(scroll); self.set_running(False)
        self.start.clicked.connect(self._emit_start); self.stop.clicked.connect(self.stop_requested)
        self.pause.clicked.connect(self.pause_requested); self.resume.clicked.connect(self.resume_requested)
        self.skip.clicked.connect(self.skip_requested); self.repeat.clicked.connect(self.repeat_requested)
        self.bad.clicked.connect(self._mark_bad); self.manual.clicked.connect(lambda: self.manual_mark_requested.emit("操作员手动标记"))
        self.new_donning.clicked.connect(self.new_donning_requested); self.new_stage.clicked.connect(lambda: self.new_stage_requested.emit(self.stage_name.text().strip()))

    @staticmethod
    def _make_card(title: str, description: str, horizontal: bool = False,
                   grid: bool = False) -> tuple[QFrame, QVBoxLayout | QHBoxLayout | QGridLayout]:
        frame = QFrame(); frame.setObjectName("card")
        outer = QVBoxLayout(frame); outer.setContentsMargins(18, 16, 18, 16); outer.setSpacing(12)
        title_label = QLabel(title); title_label.setObjectName("cardTitle")
        description_label = QLabel(description); description_label.setObjectName("cardDescription")
        outer.addWidget(title_label); outer.addWidget(description_label)
        body = QHBoxLayout() if horizontal else (QGridLayout() if grid else QVBoxLayout())
        body.setSpacing(10)
        outer.addLayout(body)
        return frame, body

    @staticmethod
    def _add_field(layout: QGridLayout, row: int, title: str, widget: QWidget) -> None:
        label = QLabel(title); label.setObjectName("fieldLabel")
        widget.setMinimumHeight(38)
        widget.setFixedWidth(300)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignLeft)
        layout.setColumnStretch(1, 1)

    def reload_protocols(self) -> None:
        try: self.protocols = self.loader.discover()
        except Exception as exc: QMessageBox.warning(self, "实验协议错误", str(exc)); self.protocols = {}
        self.protocol.clear(); self.protocol.addItems(self.protocols)
        default_protocol = "meta_discrete_7_v1"
        if default_protocol in self.protocols:
            self.protocol.setCurrentText(default_protocol)

    def run_signal_check(self, samples: np.ndarray) -> bool:
        if len(samples) < 100:
            self.quality_grid_widget.hide(); self.signal_result.show()
            self.signal_result.setText("无数据：请先连接设备并等待至少 0.05 秒"); return False
        result = SignalQualityMonitor().calculate(samples)
        passed = all(item.status == "GOOD" for item in result)
        status_names = {"GOOD": "良好", "WARNING": "警告", "NO DATA": "无数据"}
        card_names = {"GOOD": "qualityGood", "WARNING": "qualityWarning", "NO DATA": "qualityBad"}
        self.signal_result.hide(); self.quality_grid_widget.show()
        for (quality_card, status_label, metrics_label), quality in zip(self.quality_cards, result):
            quality_card.setObjectName(card_names.get(quality.status, "qualityBad"))
            status_label.setText(status_names.get(quality.status, quality.status))
            metrics_label.setText(f"RMS {quality.rms:,.0f}\n峰峰值 {quality.peak_to_peak:,.0f}")
            quality_card.style().unpolish(quality_card); quality_card.style().polish(quality_card)
        return passed

    def _emit_start(self) -> None:
        try:
            participant = ParticipantInfo(
                self.participant_id.text().strip(), str(self.dominant.currentData() or "")
            )
            info = SessionInfo(
                self.session_id.text().strip(), self.experiment_name.text().strip(),
                self.protocol.currentText(),
                stage_name=self.stage_name.text().strip() or "default",
                dataset_split=str(self.dataset_split.currentData() or "auto"),
            )
            protocol = self.loader.load(self.protocols[self.protocol.currentText()])
            participant.validate(); info.validate()
        except Exception as exc: QMessageBox.warning(self, "无法开始实验", str(exc)); return
        self.start_requested.emit(participant, info, protocol)

    def _mark_bad(self) -> None:
        reason = "operator_reject"
        self.bad_requested.emit(reason, "操作员标记为异常")

    def set_running(self, running: bool) -> None:
        if self._was_running and not running:
            if self.prompt_panel.current_state != PromptState.FINISHED:
                self.prompt_panel.reset_task()
        self._was_running = running
        for button in (self.pause, self.resume, self.skip, self.repeat, self.bad, self.manual, self.stop, self.new_donning, self.new_stage): button.setEnabled(running)

        self.start.setEnabled(not running)
        for widget in (self.participant_id, self.session_id, self.experiment_name,
                       self.protocol, self.dataset_split):
            widget.setEnabled(not running)

    def update_state(self, state: PromptState, label: str, trial: int, total: int) -> None:
        self.prompt_panel.update_prompt(state, label, trial, total)

    def update_phase_duration(self, state: PromptState, seconds: float) -> None:
        self.prompt_panel.update_phase_duration(state, seconds)

    def update_identity(self, participant: str, session: str, protocol: str, donning: int, stage: int) -> None:
        # Identity remains in the HDF5 metadata; the operator panel intentionally
        # shows only the participant prompt and compact trial progress.
        return None

    @staticmethod
    def _label_name(label: str) -> str:
        return {
            "click": "点击", "left_swipe": "向左滑动", "right_swipe": "向右滑动",
            "rest": "静息", "thumb_up": "拇指向上", "thumb_down": "拇指向下",
            "index_pinch": "食指捏合", "middle_pinch": "中指捏合",
            "fist": "握拳", "open_hand": "张开手掌",
            "thumb_tap": "拇指轻点", "thumb_swipe_left": "拇指向左滑",
            "thumb_swipe_right": "拇指向右滑", "thumb_swipe_up": "拇指向上滑",
            "thumb_swipe_down": "拇指向下滑", "index_hold": "食指捏合并保持",
            "middle_hold": "中指捏合并保持",
            "null_finger_snap": "弹指（非目标动作）",
            "null_finger_flick": "快速甩动手指（非目标动作）",
            "null_typing": "自然连续打字",
        }.get(label, label)
