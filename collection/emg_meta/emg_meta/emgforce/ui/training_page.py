from __future__ import annotations

import subprocess
import threading

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from emgforce.algorithms import ALGORITHMS, META_CONV_LSTM, get_algorithm
from emgforce.transfer.dataset_upload import CommandResult
from emgforce.transfer.remote_training import (
    RemoteTrainingSettings, RemoteTrainingStatus, fetch_remote_training,
    start_remote_training, stop_remote_training, test_training_connection,
)


class RemoteTrainingWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action: str, settings: RemoteTrainingSettings) -> None:
        super().__init__()
        self.action = action
        self.settings = settings
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
            if self.action == "test":
                result = test_training_connection(self.settings, runner=self._run_command)
            elif self.action == "start":
                result = start_remote_training(self.settings, runner=self._run_command)
            elif self.action == "stop":
                result = stop_remote_training(self.settings, runner=self._run_command)
            elif self.action == "poll":
                result = fetch_remote_training(self.settings, runner=self._run_command)
            else:
                raise ValueError(f"未知远端训练操作：{self.action}")
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
            raise RuntimeError(f"远端命令执行超时：{args[0]}")
        finally:
            with self._process_lock:
                self._process = None
        if self._cancelled.is_set():
            raise InterruptedError
        return CommandResult(process.returncode, stdout, stderr)


class TrainingPage(QWidget):
    """Launch persistent remote training in tmux and stream its pane output."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("trainingPage")
        self.worker: RemoteTrainingWorker | None = None
        self.log_timer = QTimer(self)
        self.log_timer.setInterval(2000)
        self.log_timer.timeout.connect(self.refresh_status)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("trainingScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(22, 18, 22, 24)
        page.setSpacing(16)

        settings_card, settings_layout = self._make_card(
            "远端训练设置",
            "通过 SSH 在服务器 tmux 会话中训练；关闭上位机不会中断远端任务")
        fields = QGridLayout()
        fields.setHorizontalSpacing(18)
        fields.setVerticalSpacing(10)
        defaults = RemoteTrainingSettings()
        self.algorithm = QComboBox()
        for spec in ALGORITHMS.values():
            self.algorithm.addItem(spec.display_name, spec.algorithm_id)
        self.algorithm.setCurrentIndex(self.algorithm.findData(defaults.algorithm_id))
        self.host = QLineEdit(defaults.host)
        self.repository = QLineEdit(defaults.repository)
        self.data_location = QLineEdit(defaults.data_location)
        self.conda_init = QLineEdit(defaults.conda_init)
        self.conda_env = QLineEdit(defaults.conda_env)
        self.python = QLineEdit(defaults.python)
        self.tmux_session = QLineEdit(defaults.tmux_session)
        self.gpu_ids = QLineEdit(defaults.gpu_ids)
        self.max_epochs = QSpinBox()
        self.max_epochs.setRange(1, 10000)
        self.max_epochs.setValue(defaults.max_epochs)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(defaults.batch_size)
        entries = (
            ("训练算法", self.algorithm), ("SSH 主机", self.host),
            ("训练仓库", self.repository),
            ("远端数据目录", self.data_location), ("Python", self.python),
            ("Conda 初始化脚本", self.conda_init), ("Conda 环境", self.conda_env),
            ("tmux 会话", self.tmux_session), ("GPU 编号", self.gpu_ids),
            ("训练轮数", self.max_epochs), ("Batch size", self.batch_size),
        )
        for index, (title, widget) in enumerate(entries):
            row, pair = divmod(index, 2)
            label = QLabel(title)
            label.setObjectName("fieldLabel")
            widget.setMinimumHeight(38)
            fields.addWidget(label, row, pair * 2)
            fields.addWidget(widget, row, pair * 2 + 1)
        fields.setColumnStretch(1, 1)
        fields.setColumnStretch(3, 1)
        settings_layout.addLayout(fields)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.test_button = QPushButton("测试训练环境")
        self.start_button = QPushButton("在 tmux 中开始训练")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("终止远端训练")
        self.stop_button.setObjectName("danger")
        for button in (self.test_button, self.start_button, self.stop_button):
            button.setMinimumHeight(42)
        self.test_button.clicked.connect(self.test_connection)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.algorithm.currentIndexChanged.connect(self._algorithm_changed)
        actions.addWidget(self.test_button)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        settings_layout.addLayout(actions)
        self.status_label = QLabel("尚未连接远端训练环境")
        self.status_label.setWordWrap(True)
        self._set_status("dataStatusIdle")
        settings_layout.addWidget(self.status_label)
        page.addWidget(settings_card)

        log_card, log_layout = self._make_card(
            "实时训练日志", "日志直接读取 tmux pane，重新打开上位机后仍可继续查看")
        log_actions = QHBoxLayout()
        self.refresh_button = QPushButton("立即刷新")
        self.auto_refresh = QCheckBox("每 2 秒自动刷新")
        self.auto_refresh.setChecked(True)
        self.clear_button = QPushButton("清空本地显示")
        self.refresh_button.clicked.connect(self.refresh_status)
        self.auto_refresh.toggled.connect(self._auto_refresh_changed)
        log_actions.addWidget(self.refresh_button)
        log_actions.addWidget(self.auto_refresh)
        log_actions.addWidget(self.clear_button)
        log_actions.addStretch()
        log_layout.addLayout(log_actions)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_output.setMinimumHeight(360)
        self.log_output.setPlaceholderText("测试连接或点击“立即刷新”后显示 tmux 日志")
        self.clear_button.clicked.connect(self.log_output.clear)
        log_layout.addWidget(self.log_output, 1)
        page.addWidget(log_card, 1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self._algorithm_changed()

    def settings(self) -> RemoteTrainingSettings:
        return RemoteTrainingSettings(
            self.host.text(), self.repository.text(), self.data_location.text(),
            self.conda_init.text(), self.conda_env.text(), self.python.text(),
            self.tmux_session.text(), self.gpu_ids.text(),
            self.max_epochs.value(), self.batch_size.value(),
            str(self.algorithm.currentData() or META_CONV_LSTM),
        )

    def _algorithm_changed(self) -> None:
        spec = get_algorithm(str(self.algorithm.currentData() or META_CONV_LSTM))
        self.repository.setText(spec.remote_repository)
        self.tmux_session.setText(spec.tmux_session)
        self.conda_env.setText(spec.conda_env)
        self.python.setText(spec.python)
        self.max_epochs.setValue(300 if spec.algorithm_id == "personal_mpf_tds_v1" else 30)
        self.status_label.setText(f"已选择 {spec.display_name}；同一时间只允许一个 GPU 训练任务")
        self._set_status("dataStatusIdle")

    def test_connection(self) -> None:
        self._run("test", "正在检查 SSH、tmux、Python、PyTorch 和远端数据……")

    def start_training(self) -> None:
        self._run("start", "正在创建远端 tmux 训练任务……")

    def stop_training(self) -> None:
        if self.worker is not None:
            return
        answer = QMessageBox.question(
            self, "终止远端训练",
            f"确定终止 tmux 会话 {self.tmux_session.text().strip()}？训练进程会被结束。")
        if answer == QMessageBox.StandardButton.Yes:
            self._run("stop", "正在终止远端训练……")

    def refresh_status(self) -> None:
        if self.worker is None:
            self._run("poll", "", show_busy=False)

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        self.log_timer.stop()
        worker = self.worker
        if worker is None:
            return True
        worker.cancel()
        return worker.wait(timeout_ms)

    def _run(self, action: str, status: str, *, show_busy: bool = True) -> None:
        if self.worker is not None:
            return
        if status:
            self.status_label.setText(status)
            self._set_status("dataStatusRunning")
        worker = RemoteTrainingWorker(action, self.settings())
        self.worker = worker
        worker.succeeded.connect(self._operation_succeeded)
        worker.failed.connect(self._operation_failed)
        worker.finished.connect(self._worker_finished)
        if show_busy:
            self._set_busy(True)
        worker.start()

    def _operation_succeeded(self, result: object) -> None:
        worker = self.worker
        action = worker.action if worker is not None else ""
        if isinstance(result, RemoteTrainingStatus):
            self._show_remote_status(result)
            return
        detail = str(result).strip()
        if action == "test":
            self.status_label.setText("远端训练环境检查通过" + (f"：{detail}" if detail else ""))
            self._set_status("dataStatusGood")
        elif action == "start":
            self.status_label.setText(detail)
            self._set_status("dataStatusRunning")
        else:
            self.status_label.setText(detail)
            self._set_status("dataStatusIdle")
        if action in {"test", "start"} and self.auto_refresh.isChecked():
            self.log_timer.start()
        if action in {"start", "stop"}:
            QTimer.singleShot(300, self.refresh_status)

    def _operation_failed(self, message: str) -> None:
        self.status_label.setText(message)
        self._set_status("dataStatusBad")

    def _worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _show_remote_status(self, status: RemoteTrainingStatus) -> None:
        if not status.exists:
            self.status_label.setText(
                f"tmux 会话 {self.tmux_session.text().strip()} 不存在，尚未开始训练或已被终止")
            self._set_status("dataStatusIdle")
            return
        scroll = self.log_output.verticalScrollBar()
        at_bottom = scroll.value() >= scroll.maximum() - 2
        if status.output != self.log_output.toPlainText():
            self.log_output.setPlainText(status.output)
            if at_bottom:
                self.log_output.verticalScrollBar().setValue(
                    self.log_output.verticalScrollBar().maximum())
        if status.running:
            self.status_label.setText(
                f"远端训练正在运行 · tmux {self.tmux_session.text().strip()} · {status.pane_command}")
            self._set_status("dataStatusRunning")
        else:
            self.status_label.setText(
                f"tmux 会话仍在，训练进程当前未运行 · {status.pane_command or 'shell'}")
            self._set_status("dataStatusGood")

    def _auto_refresh_changed(self, checked: bool) -> None:
        if checked:
            self.log_timer.start()
            self.refresh_status()
        else:
            self.log_timer.stop()

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.algorithm, self.host, self.repository, self.data_location, self.conda_init,
            self.conda_env, self.python,
            self.tmux_session, self.gpu_ids, self.max_epochs, self.batch_size,
            self.test_button, self.start_button, self.stop_button,
        ):
            widget.setEnabled(not busy)

    def _set_status(self, object_name: str) -> None:
        self.status_label.setObjectName(object_name)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

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
