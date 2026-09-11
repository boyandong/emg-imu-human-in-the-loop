from __future__ import annotations

from pathlib import Path
import re
import json
import hashlib

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

from .prompt_window import ACTION_NAMES, ParticipantPromptWindow


SESSIONS_PER_PARTICIPANT = 4


class ExperimentPage(QWidget):
    start_requested = Signal(object, object, object)
    stop_requested = Signal()
    pause_requested = Signal(); resume_requested = Signal(); skip_requested = Signal(); repeat_requested = Signal()
    bad_requested = Signal(str, str); manual_mark_requested = Signal(str)
    new_donning_requested = Signal(); new_stage_requested = Signal(str)

    def __init__(self, protocol_dir: Path, data_root: Path | None = None) -> None:
        super().__init__(); self.setObjectName("experimentPage")
        self.loader = ProtocolLoader(protocol_dir); self.protocols = {}
        self.data_root = Path(data_root) if data_root is not None else Path(protocol_dir).parent / "data"
        self._was_running = False
        self._quality_approved = False
        self._quality_report: dict = {}

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
        self.participant_id.editingFinished.connect(self.refresh_session_id)
        self._add_field(participant, 0, "参与者编号 *", self.participant_id)
        self._add_field(participant, 1, "惯用手", self.dominant)

        session_box, session = self._make_card("实验设置", "确定场次、数据集、流程和当前实验条件", grid=True)
        self.session_id = QLineEdit("S01"); self.session_id.setReadOnly(True)
        self.experiment_name = QLineEdit("肌律二十八组合正式数据集_v2")
        self.protocol = QComboBox(); self.stage_name = QLineEdit("默认")
        self.dataset_split = QComboBox()
        self.dataset_split.addItem(
            "自动固定（S01–S02 train，S03 validation，S04 final test）", "auto")
        self.session_id.setPlaceholderText("例如：S01")
        self.experiment_name.setPlaceholderText("例如：手势数据集_v1")
        self.stage_name.setPlaceholderText("例如：默认、掌心向上")
        for row, (field_title, widget) in enumerate((("场次编号 *", self.session_id),
                ("实验名称 *", self.experiment_name), ("实验协议", self.protocol),
                ("数据集划分", self.dataset_split),
                ("当前阶段", self.stage_name))):
            self._add_field(session, row, field_title, widget)
        self.session_plan_status = QLabel("输入受试者编号后自动安排场次 · 每人正式采集 4 轮")
        self.session_plan_status.setObjectName("muted")
        session.addWidget(self.session_plan_status, 5, 0, 1, 2)
        self.donning_notes = QLineEdit()
        self.donning_notes.setPlaceholderText("佩戴编号、局部参考照片文件名或其他说明")
        self.tested_arm = QComboBox(); self.tested_arm.addItem("右臂", "right"); self.tested_arm.addItem("左臂", "left")
        self.channel1_orientation = QLineEdit(); self.channel1_orientation.setPlaceholderText("例如：通道1朝拇指侧/标记线朝上")
        self.anatomical_marker = QLineEdit(); self.anatomical_marker.setPlaceholderText("例如：腕横纹上方 6 cm")
        self.strap_setting = QLineEdit(); self.strap_setting.setPlaceholderText("例如：刻度3，松紧 2/5")
        self.physical_condition = QLineEdit(); self.physical_condition.setPlaceholderText("近期上肢用力、湿皮肤、疲劳；没有则填“无”")
        self.anatomical_distance_mm = QLineEdit(); self.anatomical_distance_mm.setPlaceholderText("例如：60")
        self.strap_scale = QLineEdit(); self.strap_scale.setPlaceholderText("例如：刻度 3")
        self.strap_tightness = QComboBox()
        for score in range(6): self.strap_tightness.addItem(f"{score} / 5", score)
        self.skin_condition = QLineEdit(); self.skin_condition.setPlaceholderText("例如：干燥、无破损")
        self.fatigue_before = QComboBox()
        for score in range(11): self.fatigue_before.addItem(f"{score} / 10", score)
        self.reference_photo = QLineEdit(); self.reference_photo.setPlaceholderText("统一角度参考照片的完整路径")
        self.operator_id = QLineEdit(); self.operator_id.setPlaceholderText("例如：OP01")
        for row, (title, widget) in enumerate((
                ("测试手臂 *", self.tested_arm), ("通道1方向 *", self.channel1_orientation),
                ("解剖高度/起点 *", self.anatomical_marker), ("绑带刻度与松紧 *", self.strap_setting),
                ("身体与皮肤状态 *", self.physical_condition), ("佩戴备注", self.donning_notes)), start=6):
            self._add_field(session, row, title, widget)
        for row, (title, widget) in enumerate((
                ("解剖距离 mm *", self.anatomical_distance_mm),
                ("绑带刻度 *", self.strap_scale), ("绑带松紧 *", self.strap_tightness),
                ("皮肤状态 *", self.skin_condition), ("采集前疲劳 *", self.fatigue_before),
                ("参考照片路径 *", self.reference_photo), ("操作员编号 *", self.operator_id)), start=12):
            self._add_field(session, row, title, widget)
        left_column.addWidget(participant_box); left_column.addWidget(session_box)
        self.reload_protocols()

        check_box, check = self._make_card("采集前质量门控", "先静息采集 8 秒；检查饱和、断线、50 Hz 干扰和相邻通道异常")
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
        self.signal_check = QPushButton("执行/重新执行信号检查"); self.continue_anyway = QPushButton("填写原因并忽略警告")
        self.signal_check.setObjectName("primary")
        result_area = QVBoxLayout(); result_area.addWidget(self.signal_result); result_area.addWidget(self.quality_grid_widget)
        action_area = QHBoxLayout(); action_area.setSpacing(8)
        self.signal_check.setMinimumHeight(38); self.continue_anyway.setMinimumHeight(38)
        action_area.addWidget(self.signal_check); action_area.addWidget(self.continue_anyway); action_area.addStretch()
        check.addLayout(result_area); check.addLayout(action_area)
        left_column.addWidget(check_box); left_column.addStretch()

        panel, panel_layout = self._make_card(
            "音乐控制动作采集",
            "每轮 144 次：28 种手臂×手部组合随机出现；保留 ±200 ms 起始偏移")
        self.prompt_panel = ParticipantPromptWindow(self)
        panel_layout.addWidget(self.prompt_panel)

        primary_actions = QHBoxLayout(); primary_actions.setSpacing(8)
        self.start = QPushButton("开始实验"); self.start.setObjectName("record")
        self.pause = QPushButton("暂停"); self.resume = QPushButton("继续"); self.skip = QPushButton("废弃试次"); self.repeat = QPushButton("废弃并立即补采")
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
        formal = {name: path for name, path in self.protocols.items()
                  if self.loader.load(path).formal_collection}
        if formal:
            self.protocols = formal
            self.protocol.clear(); self.protocol.addItems(self.protocols)
        default_protocol = "jilv_music_28_v2"
        if default_protocol in self.protocols:
            self.protocol.setCurrentText(default_protocol)

    def run_signal_check(self, samples: np.ndarray) -> bool:
        if len(samples) < 8 * 250:
            self.quality_grid_widget.hide(); self.signal_result.show()
            self.signal_result.setText("数据不足：请保持自然放松并连续等待至少 8 秒"); return False
        monitor = SignalQualityMonitor()
        result = monitor.calculate(samples)
        self._quality_report = monitor.report(samples)
        passed = all(item.status == "GOOD" for item in result)
        passed = passed and bool(self._quality_report.get("passed"))
        self._quality_approved = passed
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
        if not self._quality_approved:
            QMessageBox.warning(self, "质量门控未通过",
                                "请先连接设备，静息至少 8 秒并执行信号检查；如确认设备状态可用，可点击“忽略警告并继续”。")
            return
        try:
            participant_id = self.participant_id.text().strip()
            protocol = self.loader.load(self.protocols[self.protocol.currentText()])
            required = (self.channel1_orientation.text().strip(),
                        self.anatomical_marker.text().strip(), self.strap_setting.text().strip(),
                        self.physical_condition.text().strip(),
                        self.anatomical_distance_mm.text().strip(), self.strap_scale.text().strip(),
                        self.skin_condition.text().strip(), self.reference_photo.text().strip(),
                        self.operator_id.text().strip())
            if not all(required):
                raise ValueError("请完整填写结构化佩戴、皮肤、疲劳、参考照片和操作员信息")
            participant = ParticipantInfo(participant_id, str(self.dominant.currentData() or ""))
            participant.validate()
            if protocol.formal_collection and not participant.dominant_hand:
                raise ValueError("正式采集必须填写惯用手")
            session_id = self._available_session_id(participant_id)
            if session_id != self.session_id.text().strip():
                self.session_id.setText(session_id)
            photo_path = Path(self.reference_photo.text().strip())
            if float(self.anatomical_distance_mm.text()) <= 0:
                raise ValueError("解剖距离必须为正数（mm）")
            if protocol.formal_collection and not photo_path.is_file():
                raise ValueError("正式采集必须选择存在的参考照片文件")
            photo_hash = ""
            if photo_path.is_file():
                photo_hash = hashlib.sha256(photo_path.read_bytes()).hexdigest()
            model_frozen = session_id != "S04"
            if session_id == "S04":
                answer = QMessageBox.question(
                    self, "S04 正式测试锁定",
                    "S04 是 Final test。确认分类模型、阈值和校准算法已经冻结，且采集后不会根据结果调参吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    raise ValueError("未确认算法冻结，不能创建 S04")
                model_frozen = True
            info = SessionInfo(
                session_id, self.experiment_name.text().strip(),
                self.protocol.currentText(),
                stage_name=self.stage_name.text().strip() or "default",
                dataset_split=str(self.dataset_split.currentData() or "auto"),
                quality_report_json=json.dumps(self._quality_report, ensure_ascii=False),
                donning_notes=self.donning_notes.text().strip(),
                tested_arm=str(self.tested_arm.currentData()),
                channel1_orientation=self.channel1_orientation.text().strip(),
                anatomical_marker=self.anatomical_marker.text().strip(),
                strap_setting=self.strap_setting.text().strip(),
                stabilization_sec=60,
                physical_condition=self.physical_condition.text().strip(),
                recorded_arm=str(self.tested_arm.currentData()), donning_code="D01",
                anatomical_distance_mm=float(self.anatomical_distance_mm.text()),
                strap_scale=self.strap_scale.text().strip(),
                strap_tightness=int(self.strap_tightness.currentData()),
                skin_condition=self.skin_condition.text().strip(),
                fatigue_before=int(self.fatigue_before.currentData()),
                reference_photo_name=photo_path.name,
                reference_photo_sha256=photo_hash,
                operator_id=self.operator_id.text().strip(),
                quality_override_reason=str(self._quality_report.get("operator_override_reason", "")),
                model_frozen_confirmed=model_frozen,
            )
            info.validate()
        except Exception as exc: QMessageBox.warning(self, "无法开始实验", str(exc)); return
        self.start_requested.emit(participant, info, protocol)

    def _used_session_numbers(self, participant_id: str) -> set[int]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", participant_id):
            return set()
        participant_dir = self.data_root / participant_id
        used: set[int] = set()
        if participant_dir.exists():
            for path in participant_dir.iterdir():
                if not path.is_dir():
                    continue
                match = re.search(r"_S(\d+)$", path.name, re.IGNORECASE)
                if match:
                    used.add(int(match.group(1)))
        return used

    def _available_session_id(self, participant_id: str) -> str:
        """Find the first unused formal S01..S04 across all collection dates."""
        used = self._used_session_numbers(participant_id)
        for number in range(1, SESSIONS_PER_PARTICIPANT + 1):
            if number not in used:
                return f"S{number:02d}"
        raise ValueError("该受试者已经完成 4 / 4 个正式 Session，不能再创建新场次")

    def refresh_session_id(self) -> None:
        participant_id = self.participant_id.text().strip()
        if not participant_id:
            self.session_id.setText("S01")
            self.session_plan_status.setText("输入受试者编号后自动安排场次 · 每人正式采集 4 轮")
            self.start.setEnabled(not self._was_running)
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", participant_id):
            self.session_plan_status.setText("受试者编号只能包含字母、数字、点、下划线和连字符")
            self.start.setEnabled(False)
            return
        used = self._used_session_numbers(participant_id)
        try:
            session_id = self._available_session_id(participant_id)
        except ValueError:
            self.session_id.setText("已完成")
            self.session_plan_status.setText("该受试者已完成 4 / 4 个正式 Session")
            self.start.setEnabled(False)
            return
        self.session_id.setText(session_id)
        completed = len(used & set(range(1, SESSIONS_PER_PARTICIPANT + 1)))
        self.session_plan_status.setText(
            f"已完成 {completed} / 4 轮 · 下一轮 {session_id}"
        )
        self.start.setEnabled(not self._was_running)

    def _mark_bad(self) -> None:
        reason = "operator_reject"
        self.bad_requested.emit(reason, "操作员标记为异常")

    def set_running(self, running: bool) -> None:
        if self._was_running and not running:
            if self.prompt_panel.current_state != PromptState.FINISHED:
                self.prompt_panel.reset_task()
            self._quality_approved = False
            self._quality_report = {}
            self.signal_result.show(); self.signal_result.setText("新 Session 必须重新执行 8 秒质量检查")
            self.quality_grid_widget.hide()
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
        return ACTION_NAMES.get(label, {
            "click": "点击", "left_swipe": "向左滑动", "right_swipe": "向右滑动",
            "rest": "静息", "thumb_up": "拇指向上", "thumb_down": "拇指向下",
            "index_pinch": "食指捏合", "middle_pinch": "中指捏合",
            "fist": "舒适力度稳定握拳", "open_hand": "主动伸展张手（不是自然放松）",
            "thumb_tap": "拇指轻点", "thumb_swipe_left": "拇指向左滑",
            "thumb_swipe_right": "拇指向右滑", "thumb_swipe_up": "拇指向上滑",
            "thumb_swipe_down": "拇指向下滑", "index_hold": "食指捏合并保持",
            "middle_hold": "中指捏合并保持",
            "null_finger_snap": "弹指（非目标动作）",
            "null_finger_flick": "快速甩动手指（非目标动作）",
            "null_typing": "自然连续打字",
        }.get(label, label))
