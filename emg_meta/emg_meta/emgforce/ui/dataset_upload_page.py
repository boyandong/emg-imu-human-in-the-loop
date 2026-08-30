from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from emgforce.processing.meta_corpus import rebuild_corpus
from emgforce.transfer.dataset_upload import (
    CommandResult, UploadPlan, build_upload_plan, test_ssh_connection, upload_plan,
)


DEFAULT_SSH_HOST = "my-gpu-server"
DEFAULT_REMOTE_ROOT = "/home/qxy/qxy/emg_data/emgforce_dataset"


class DatasetUploadWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, mode: str, host: str, remote_root: str,
                 plan: UploadPlan | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.host = host
        self.remote_root = remote_root
        self.plan = plan
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
            if self.mode == "test":
                response = test_ssh_connection(
                    self.host, self.remote_root, runner=self._run_command)
                state = "远端目录已存在" if "REMOTE_DIR_EXISTS" in response else "远端目录将在上传时创建"
                self.succeeded.emit(f"SSH 连接成功；{state}")
                return
            if self.plan is None:
                raise RuntimeError("缺少上传计划")
            uploaded, skipped = upload_plan(
                self.plan, self.host, self.remote_root,
                runner=self._run_command,
                progress=lambda current, total, message: self.progress.emit(
                    current, total, message),
                cancelled=self._cancelled.is_set,
            )
            self.succeeded.emit(
                f"上传完成：新上传/更新 {uploaded} 个文件，跳过相同文件 {skipped} 个；"
                f"远端数据目录：{self.remote_root}")
        except InterruptedError:
            self.failed.emit("上传已取消")
        except Exception as exc:
            if self._cancelled.is_set():
                self.failed.emit("上传已取消")
            else:
                self.failed.emit(str(exc))

    def _run_command(self, args: list[str], timeout: float) -> CommandResult:
        if self._cancelled.is_set():
            raise InterruptedError("上传已取消")
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
            raise RuntimeError(f"命令执行超时：{args[0]}")
        finally:
            with self._process_lock:
                self._process = None
        if self._cancelled.is_set():
            raise InterruptedError("上传已取消")
        return CommandResult(process.returncode, stdout, stderr)


class DatasetUploadPage(QWidget):
    """Validate and upload the corpus-defined Meta training package over SSH."""

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self.setObjectName("datasetUploadPage")
        self.data_root = Path(data_root)
        self.plan: UploadPlan | None = None
        self.worker: DatasetUploadWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("datasetUploadScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("datasetUploadContent")
        page = QVBoxLayout(content)
        page.setContentsMargins(22, 18, 22, 24)
        page.setSpacing(16)

        dataset_card, dataset_layout = self._make_card(
            "本地训练数据集",
            "自动读取 corpus，只上传登记且通过 meta_8ch_v1 就绪检查的对齐文件")
        path_row = QHBoxLayout()
        self.local_path = QLineEdit(str(self.data_root.resolve()))
        self.local_path.setReadOnly(True)
        self.local_path.setMinimumHeight(40)
        self.refresh_button = QPushButton("重新扫描数据集")
        self.refresh_button.setMinimumHeight(40)
        self.refresh_button.clicked.connect(self.refresh_plan)
        self.regenerate_button = QPushButton("重新生成 CSV")
        self.regenerate_button.setMinimumHeight(40)
        self.regenerate_button.clicked.connect(self.regenerate_corpus)
        path_row.addWidget(self.local_path, 1)
        path_row.addWidget(self.refresh_button)
        path_row.addWidget(self.regenerate_button)
        dataset_layout.addLayout(path_row)

        metrics = QGridLayout()
        metrics.setSpacing(10)
        self.sessions_value = self._metric(metrics, 0, "Session", "--")
        self.split_value = self._metric(metrics, 1, "Train / Val / Test", "--")
        self.prompts_value = self._metric(metrics, 2, "有效 Prompts", "--")
        self.size_value = self._metric(metrics, 3, "上传大小", "--")
        dataset_layout.addLayout(metrics)
        self.file_list = QPlainTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setMaximumHeight(150)
        self.file_list.setPlaceholderText("等待扫描 corpus")
        dataset_layout.addWidget(self.file_list)
        self.local_status = QLabel("正在扫描本地训练数据集……")
        self.local_status.setWordWrap(True)
        self._set_status(self.local_status, "dataStatusRunning")
        dataset_layout.addWidget(self.local_status)
        page.addWidget(dataset_card)

        server_card, server_layout = self._make_card(
            "SSH 服务器",
            "使用系统 SSH 密钥连接；上传到临时文件并通过 SHA-256 后再原子替换")
        fields = QGridLayout()
        fields.setHorizontalSpacing(12)
        fields.setVerticalSpacing(10)
        self.host = QLineEdit(DEFAULT_SSH_HOST)
        self.remote_root = QLineEdit(DEFAULT_REMOTE_ROOT)
        for row, (title, widget) in enumerate((
                ("SSH 主机", self.host), ("远端数据集目录", self.remote_root))):
            label = QLabel(title)
            label.setObjectName("fieldLabel")
            widget.setMinimumHeight(40)
            fields.addWidget(label, row, 0)
            fields.addWidget(widget, row, 1)
        fields.setColumnStretch(1, 1)
        server_layout.addLayout(fields)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.test_button = QPushButton("测试 SSH 连接")
        self.upload_button = QPushButton("一键上传训练数据集")
        self.upload_button.setObjectName("primary")
        self.cancel_button = QPushButton("取消上传")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setEnabled(False)
        for button in (self.test_button, self.upload_button, self.cancel_button):
            button.setMinimumHeight(42)
        self.test_button.clicked.connect(self.test_connection)
        self.upload_button.clicked.connect(self.start_upload)
        self.cancel_button.clicked.connect(self.cancel_upload)
        actions.addWidget(self.test_button)
        actions.addWidget(self.upload_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        server_layout.addLayout(actions)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(10)
        self.remote_status = QLabel("尚未连接服务器")
        self.remote_status.setWordWrap(True)
        self._set_status(self.remote_status, "dataStatusIdle")
        self.training_hint = QLabel(
            "上传完成后，Meta 训练的 data_location 应设置为上面的远端数据集目录；"
            "当前 Conv-LSTM 训练配置已经固定为 8 通道。")
        self.training_hint.setWordWrap(True)
        self.training_hint.setObjectName("muted")
        server_layout.addWidget(self.progress_bar)
        server_layout.addWidget(self.remote_status)
        server_layout.addWidget(self.training_hint)
        page.addWidget(server_card)
        page.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)
        QTimer.singleShot(0, self.refresh_plan)

    def refresh_plan(self) -> bool:
        try:
            self.plan = build_upload_plan(self.data_root, refresh_manifest=True)
        except Exception as exc:
            self.plan = None
            self.local_status.setText(f"数据集未就绪：{exc}")
            self._set_status(self.local_status, "dataStatusBad")
            self.file_list.clear()
            self._clear_metrics()
            self.upload_button.setEnabled(False)
            return False
        plan = self.plan
        self.sessions_value.setText(str(plan.sessions))
        self.split_value.setText(
            f"{plan.train_sessions} / {plan.val_sessions} / {plan.test_sessions}")
        self.prompts_value.setText(str(plan.prompts))
        self.size_value.setText(_format_bytes(plan.total_bytes))
        self.file_list.setPlainText("\n".join(
            f"{item.relative_path}  ·  {_format_bytes(item.size_bytes)}"
            for item in plan.items))
        warning = "；".join(plan.warnings)
        self.local_status.setText(
            f"训练数据集检查通过，共 {len(plan.items)} 个文件。"
            + (f" 注意：{warning}" if warning else ""))
        self._set_status(
            self.local_status, "dataStatusRunning" if warning else "dataStatusGood")
        self.upload_button.setEnabled(self.worker is None)
        return True

    def regenerate_corpus(self) -> bool:
        if self.worker is not None:
            return False
        self.local_status.setText("正在扫描对齐文件并重新生成 CSV……")
        self._set_status(self.local_status, "dataStatusRunning")
        try:
            result = rebuild_corpus(self.data_root)
        except Exception as exc:
            self.local_status.setText(f"CSV 重新生成失败：{exc}")
            self._set_status(self.local_status, "dataStatusBad")
            return False

        ready = self.refresh_plan()
        skipped = f"；跳过 {len(result.skipped)} 个未通过检查的文件" if result.skipped else ""
        message = (
            f"CSV 已直接替换，共登记 {result.sessions} 个 Session；已自动划分 "
            f"Train / Val / Test = {result.train_sessions} / {result.val_sessions} / "
            f"{result.test_sessions}{skipped}。")
        if ready:
            self.local_status.setText(message)
            self._set_status(
                self.local_status,
                "dataStatusRunning" if result.skipped else "dataStatusGood")
        else:
            self.local_status.setText(f"{message} 但上传数据集尚未就绪。")
        return True

    def test_connection(self) -> None:
        if self.worker is not None:
            return
        self._start_worker(DatasetUploadWorker(
            "test", self.host.text(), self.remote_root.text()))
        self.remote_status.setText("正在测试 SSH 连接……")
        self._set_status(self.remote_status, "dataStatusRunning")

    def start_upload(self) -> None:
        if self.worker is not None or not self.refresh_plan():
            return
        assert self.plan is not None
        self.progress_bar.setRange(0, len(self.plan.items))
        self.progress_bar.setValue(0)
        self.remote_status.setText("正在建立连接并上传……")
        self._set_status(self.remote_status, "dataStatusRunning")
        self._start_worker(DatasetUploadWorker(
            "upload", self.host.text(), self.remote_root.text(), self.plan))

    def cancel_upload(self) -> None:
        if self.worker is not None:
            self.remote_status.setText("正在取消当前操作……")
            self.worker.cancel()

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        worker = self.worker
        if worker is None:
            return True
        worker.cancel()
        return worker.wait(timeout_ms)

    def _start_worker(self, worker: DatasetUploadWorker) -> None:
        self.worker = worker
        worker.progress.connect(self._upload_progress)
        worker.succeeded.connect(self._operation_succeeded)
        worker.failed.connect(self._operation_failed)
        worker.finished.connect(self._worker_finished)
        self._set_busy(True)
        worker.start()

    def _upload_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        self.remote_status.setText(message)

    def _operation_succeeded(self, message: str) -> None:
        if self.worker is not None and self.worker.mode == "upload":
            self.progress_bar.setValue(self.progress_bar.maximum())
        self.remote_status.setText(message)
        self._set_status(self.remote_status, "dataStatusGood")

    def _operation_failed(self, message: str) -> None:
        self.remote_status.setText(message)
        self._set_status(
            self.remote_status,
            "dataStatusIdle" if message == "上传已取消" else "dataStatusBad")

    def _worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        self._set_busy(False)
        if worker is not None:
            worker.deleteLater()

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.host, self.remote_root, self.refresh_button,
                       self.regenerate_button,
                       self.test_button, self.upload_button):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _clear_metrics(self) -> None:
        for label in (self.sessions_value, self.split_value,
                      self.prompts_value, self.size_value):
            label.setText("--")

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
    def _metric(layout: QGridLayout, column: int, title: str, value: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName("qcMetricNeutral")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 10, 14, 10)
        box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("qcMetricLabel")
        value_label = QLabel(value)
        value_label.setObjectName("qcMetricValue")
        box.addWidget(title_label)
        box.addWidget(value_label)
        layout.addWidget(frame, 0, column)
        layout.setColumnStretch(column, 1)
        return value_label

    @staticmethod
    def _set_status(label: QLabel, object_name: str) -> None:
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"
