from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from emgforce.processing.alignment_review import AlignmentReviewStore
from emgforce.processing.meta_corpus import (
    CORPUS_FILENAME, infer_data_root, validate_training_export,
)
from emgforce.processing.meta_alignment import export_meta_aligned
from emgforce.storage.hdf5_reader import Hdf5SessionReader


EXCLUSION_REASON_NAMES = {
    "search_boundary_hit": "命中搜索边界",
    "low_confidence": "置信度过低",
    "low_template_correlation": "模板相关性过低",
    "ambiguous_template_peak": "主峰与次峰难以区分",
    "paired_event_failed": "同 Trial 配对事件不合格",
    "invalid_trial": "原始 Trial 无效",
}


class AlignmentWorker(QThread):
    """Run the CPU-heavy alignment pipeline without freezing the Qt UI."""

    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, source: Path) -> None:
        super().__init__()
        self.source = Path(source)

    def run(self) -> None:
        try:
            target = export_meta_aligned(self.source)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(str(target))


class DataCheckPage(QWidget):
    """One-click alignment and read-only automatic label-quality report."""

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self.setObjectName("dataCheckPage")
        self.data_root = Path(data_root)
        self.reader: Hdf5SessionReader | None = None
        self.review_store: AlignmentReviewStore | None = None
        self._alignment_worker: AlignmentWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("dataCheckScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("dataCheckContent")
        page = QVBoxLayout(content)
        page.setContentsMargins(22, 18, 22, 24)
        page.setSpacing(16)

        action_card, action_layout = self._make_card(
            "原始数据与自动处理",
            "一次选择即可完成标签对齐、训练滤波和 Meta corpus 登记；原始数据不会被修改")
        row = QHBoxLayout()
        row.setSpacing(10)
        self.path = QLineEdit()
        self.path.setPlaceholderText("请选择原始实验文件 session.h5")
        self.path.setReadOnly(True)
        self.path.setMinimumHeight(40)
        self.align_button = QPushButton("选择原始文件并自动对齐")
        self.align_button.setObjectName("primary")
        self.align_button.setMinimumHeight(40)
        self.align_button.setMinimumWidth(220)
        self.align_button.clicked.connect(self.browse)
        row.addWidget(self.path, 1)
        row.addWidget(self.align_button)
        action_layout.addLayout(row)
        self.alignment_progress = QProgressBar()
        self.alignment_progress.setRange(0, 0)
        self.alignment_progress.setTextVisible(False)
        self.alignment_progress.setFixedHeight(8)
        self.alignment_progress.hide()
        self.alignment_status = QLabel("选择原始 session.h5 后将自动完成对齐和严格质检")
        self.alignment_status.setWordWrap(True)
        self._set_status_style(self.alignment_status, "dataStatusIdle")
        action_layout.addWidget(self.alignment_progress)
        action_layout.addWidget(self.alignment_status)
        self.training_status = QLabel("训练就绪状态：等待生成 Meta 导出")
        self.training_status.setWordWrap(True)
        self._set_status_style(self.training_status, "dataStatusIdle")
        action_layout.addWidget(self.training_status)
        page.addWidget(action_card)

        summary_card, summary_layout = self._make_card(
            "实验数据摘要", "核对参与者、场次、采样数量和原始文件完整性")
        self.summary = QLabel("尚未打开实验文件")
        self.summary.setObjectName("statusText")
        self.summary.setWordWrap(True)
        self.validation = QLabel("等待选择原始文件")
        self.validation.setWordWrap(True)
        self._set_status_style(self.validation, "validationIdle")
        summary_layout.addWidget(self.summary)
        summary_layout.addWidget(self.validation)
        page.addWidget(summary_card)

        qc_card, qc_layout = self._make_card(
            "全自动标签质检", "系统按整 Trial 自动保留或排除，不需要逐事件人工审核")
        self.qc_card_title = qc_card.findChild(QLabel, "cardTitle")
        self.qc_availability = QLabel(
            "打开 session.h5 后将自动读取 session_meta_aligned.hdf5")
        self.qc_availability.setObjectName("cardDescription")
        self.qc_overview = QLabel("尚未加载自动质检结果")
        self.qc_overview.setWordWrap(True)
        self.qc_overview.setObjectName("progressText")

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        self.qc_total_value = self._add_metric(
            metrics, 0, "目标事件", "--", "qcMetricNeutral")
        self.qc_kept_value = self._add_metric(
            metrics, 1, "保留事件", "--", "qcMetricGood")
        self.qc_excluded_value = self._add_metric(
            metrics, 2, "排除事件", "--", "qcMetricBad")
        self.qc_trials_value = self._add_metric(
            metrics, 3, "保留 / 总 Trial", "--", "qcMetricAccent")

        result_panel = QFrame()
        result_panel.setObjectName("statusPanel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(14, 12, 14, 12)
        result_layout.setSpacing(7)
        self.qc_reasons = QLabel("")
        self.qc_reasons.setWordWrap(True)
        self.qc_reasons.setObjectName("statusText")
        self.qc_trials = QLabel("")
        self.qc_trials.setWordWrap(True)
        self.qc_trials.setObjectName("statusText")
        result_layout.addWidget(self.qc_reasons)
        result_layout.addWidget(self.qc_trials)

        self.qc_thresholds = QLabel("")
        self.qc_thresholds.setWordWrap(True)
        self.qc_thresholds.setObjectName("muted")
        self.qc_policy = QLabel(
            "处理规则：可靠事件自动进入训练标签；不确定事件自动排除。"
            "按下/松开动作按整个 Trial 成对处理，任一事件不合格则整个 Trial 排除。"
            "原始 EMG、UI cue、对齐事件及排除原因始终保留。")
        self.qc_policy.setWordWrap(True)
        self.qc_policy.setObjectName("muted")
        qc_layout.addWidget(self.qc_availability)
        qc_layout.addWidget(self.qc_overview)
        qc_layout.addLayout(metrics)
        qc_layout.addWidget(result_panel)
        qc_layout.addWidget(self.qc_thresholds)
        qc_layout.addWidget(self.qc_policy)
        page.addWidget(qc_card)
        page.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)

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

    @staticmethod
    def _add_metric(layout: QGridLayout, column: int, title: str, value: str,
                    object_name: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName(object_name)
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 10, 14, 10)
        box.setSpacing(2)
        label = QLabel(title)
        label.setObjectName("qcMetricLabel")
        value_label = QLabel(value)
        value_label.setObjectName("qcMetricValue")
        box.addWidget(label)
        box.addWidget(value_label)
        layout.addWidget(frame, 0, column)
        layout.setColumnStretch(column, 1)
        return value_label

    @staticmethod
    def _set_status_style(label: QLabel, object_name: str) -> None:
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)

    def browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择原始实验数据", str(self.data_root),
            "原始 HDF5 文件 (session.h5);;HDF5 文件 (*.h5)")
        if selected:
            self.start_automatic_alignment(Path(selected))

    def start_automatic_alignment(self, source: Path) -> None:
        """Validate one raw session, then create and display its aligned export."""
        source = Path(source)
        if source.name != "session.h5":
            QMessageBox.warning(
                self, "请选择原始文件",
                "自动对齐只接受采集生成的原始 session.h5，不能选择对齐后的文件。")
            return
        if self._alignment_worker is not None and self._alignment_worker.isRunning():
            return
        if not self._load_raw_summary(source):
            return
        target = source.with_name("session_meta_aligned.hdf5")
        if target.exists():
            self.alignment_status.setText("对齐文件已存在，已直接加载自动质检结果（未覆盖原文件）")
            self._set_status_style(self.alignment_status, "dataStatusGood")
            self._open_automatic_qc(source)
            return
        self.align_button.setEnabled(False)
        self.alignment_progress.show()
        self.alignment_status.setText("正在执行模板对齐与严格自动质检，请稍候……")
        self._set_status_style(self.alignment_status, "dataStatusRunning")
        worker = AlignmentWorker(source)
        self._alignment_worker = worker
        worker.succeeded.connect(self._alignment_succeeded)
        worker.failed.connect(self._alignment_failed)
        worker.finished.connect(self._alignment_worker_finished)
        worker.start()

    def _alignment_succeeded(self, target: str) -> None:
        aligned_path = Path(target)
        self.alignment_status.setText(f"自动对齐与质检完成：{aligned_path.name}")
        self._set_status_style(self.alignment_status, "dataStatusGood")
        self._open_automatic_qc(aligned_path.with_name("session.h5"))

    def _alignment_failed(self, message: str) -> None:
        self.alignment_status.setText(f"自动对齐失败：{message}")
        self._set_status_style(self.alignment_status, "dataStatusBad")
        QMessageBox.critical(self, "自动对齐失败", message)

    def _alignment_worker_finished(self) -> None:
        self.alignment_progress.hide()
        self.align_button.setEnabled(True)
        worker = self._alignment_worker
        self._alignment_worker = None
        if worker is not None:
            worker.deleteLater()

    def _load_raw_summary(self, path: Path) -> bool:
        """Load raw-session metadata without requiring an aligned sibling."""
        try:
            reader = Hdf5SessionReader(path)
            summary = reader.summary()
            problems = reader.validate()
        except Exception as exc:
            QMessageBox.critical(
                self, "无法打开 HDF5",
                f"无法按本实验格式读取：{path}\n\n{exc}\n\n请选择原始 session.h5。")
            return False
        self.reader = reader
        self.path.setText(str(path))
        self.summary.setText("  |  ".join((
            f"参与者：{summary['participant']}", f"实验场次：{summary['session']}",
            f"时长：{summary['duration']:.2f} 秒", f"肌电样本：{summary['emg_samples']:,}",
            f"惯性样本：{summary['imu_samples']:,}",
            f"试次：{summary['trial_count']}（有效 {summary['valid_trials']} / "
            f"异常 {summary['bad_trials']}）",
            f"丢包事件：{summary['packet_loss_events']}",
        )))
        self.validation.setText("原始文件检查通过" if not problems else "；".join(problems))
        self._set_status_style(
            self.validation, "validationGood" if not problems else "validationBad")
        return True

    def open_session(self, path: Path) -> None:
        path = Path(path)
        if path.name in {"session_meta_aligned.hdf5", "session_meta_aligned_v2.hdf5"}:
            raw_path = path.with_name("session.h5")
            if not raw_path.exists():
                QMessageBox.critical(
                    self, "缺少原始实验文件",
                    f"已选择对齐文件：{path.name}\n"
                    "但同一目录中没有 session.h5，无法生成自动质检报告。")
                return
            path = raw_path
        if not self._load_raw_summary(path):
            return
        self._open_automatic_qc(path)

    def _reset_qc_metrics(self) -> None:
        for label in (self.qc_total_value, self.qc_kept_value,
                      self.qc_excluded_value, self.qc_trials_value):
            label.setText("--")

    def _open_automatic_qc(self, raw_path: Path) -> None:
        aligned_path = raw_path.with_name("session_meta_aligned.hdf5")
        if not aligned_path.exists():
            aligned_path = raw_path.with_name("session_meta_aligned_v2.hdf5")
        if not aligned_path.exists():
            self.review_store = None
            self.qc_availability.setText(
                "未找到 session_meta_aligned.hdf5 或旧版V2对齐文件；请先生成对齐文件")
            self.qc_overview.setText("尚无自动质检结果")
            self.qc_reasons.clear()
            self.qc_trials.clear()
            self.qc_thresholds.clear()
            self._reset_qc_metrics()
            self.training_status.setText("训练就绪状态：尚无 Meta 导出")
            self._set_status_style(self.training_status, "dataStatusIdle")
            return
        try:
            self.review_store = AlignmentReviewStore(raw_path, aligned_path)
            result = self.review_store.automatic_summary()
        except Exception as exc:
            self.review_store = None
            self.qc_availability.setText(f"无法加载自动质检结果：{exc}")
            self._reset_qc_metrics()
            return

        self.qc_availability.setText(f"自动质检文件：{aligned_path}")
        readiness = validate_training_export(aligned_path)
        data_root = infer_data_root(aligned_path)
        corpus = data_root / CORPUS_FILENAME if data_root is not None else None
        if readiness.ready:
            corpus_text = str(corpus) if corpus is not None and corpus.exists() else "未登记"
            self.training_status.setText(
                f"训练就绪：8通道 / {readiness.sample_rate:g} Hz / "
                f"{readiness.preprocessing_version} / prompts {readiness.prompt_count}；"
                f"Corpus：{corpus_text}")
            self._set_status_style(self.training_status, "dataStatusGood")
        else:
            self.training_status.setText(
                "训练未就绪：" + "；".join(readiness.problems))
            self._set_status_style(self.training_status, "dataStatusBad")
        self.qc_overview.setText("自动处理完成，以下标签可直接用于后续训练流程")
        self.qc_total_value.setText(str(result["total_events"]))
        self.qc_kept_value.setText(str(result["included_events"]))
        self.qc_excluded_value.setText(str(result["excluded_events"]))
        self.qc_trials_value.setText(
            f"{len(result['included_trials'])} / {result['total_trials']}")
        reasons = result["reason_counts"]
        if reasons:
            reason_text = "；".join(
                f"{EXCLUSION_REASON_NAMES.get(reason, reason)}：{count} 个事件"
                for reason, count in reasons.items())
            self.qc_reasons.setText(
                f"自动排除原因：{reason_text}（同一事件可能同时命中多项）")
        else:
            self.qc_reasons.setText("自动排除原因：无，全部通过")
        excluded_trials = result["excluded_trials"]
        self.qc_trials.setText(
            "自动排除的 Trial："
            + ("、".join(map(str, excluded_trials)) if excluded_trials else "无"))
        thresholds = result["thresholds"]
        self.qc_thresholds.setText(
            f"自动门槛：置信度 ≥ {thresholds['confidence']:.2f}；"
            f"模板相关性 ≥ {thresholds['correlation']:.2f}；"
            f"主峰与次峰差值 ≥ {thresholds['margin']:.2f}；且不得命中搜索边界。"
            f"算法：{result['algorithm']}")
