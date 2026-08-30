from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from emgforce.config import BAUDRATE, SOFTWARE_NAME
from emgforce.controller import AcquisitionController
from emgforce.experiment.session import ExperimentSession
from emgforce.styles import LIGHT_STYLESHEET
from emgforce.worker import AcquisitionConfig, AcquisitionWorker

from .data_check_page import DataCheckPage
from .dataset_upload_page import DatasetUploadPage
from .device_page import DevicePage
from .experiment_page import ExperimentPage
from .realtime_inference_page import RealtimeInferencePage
from .training_page import TrainingPage


LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__(); self.project_root = Path(project_root)
        self.setWindowTitle(f"{SOFTWARE_NAME} · 表面肌电科研数据采集")
        self.resize(1500, 950); self.setMinimumSize(1180, 760)
        self.setStyleSheet(LIGHT_STYLESHEET)
        self.worker: AcquisitionWorker | None = None; self.connected_port = ""
        self.acquisition = AcquisitionController(self)
        self.session = ExperimentSession(self.acquisition, self.project_root / "data", self)
        self.main_tabs = QTabWidget()
        self.device_page = DevicePage(); self.experiment_page = ExperimentPage(self.project_root / "protocols"); self.data_check_page = DataCheckPage(self.project_root / "data"); self.dataset_upload_page = DatasetUploadPage(self.project_root / "data"); self.training_page = TrainingPage(); self.realtime_inference_page = RealtimeInferencePage(self.project_root / "models")
        self.prompt_window = self.experiment_page.prompt_panel
        self.main_tabs.addTab(self.device_page, "设备监测"); self.main_tabs.addTab(self.experiment_page, "实验采集"); self.main_tabs.addTab(self.data_check_page, "数据检查"); self.main_tabs.addTab(self.dataset_upload_page, "数据上传"); self.main_tabs.addTab(self.training_page, "训练模型"); self.main_tabs.addTab(self.realtime_inference_page, "实时识别")
        root = QWidget(); root.setObjectName("root")
        root_layout = QVBoxLayout(root); root_layout.setContentsMargins(0, 0, 0, 0); root_layout.setSpacing(0)
        root_layout.addWidget(self.device_page.header); root_layout.addWidget(self.main_tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("● 未连接 | 2000 Hz | 8 通道 | 丢包 0.0%")
        self._wire()
        LOGGER.info("app start")

    def _wire(self) -> None:
        self.device_page.connect_requested.connect(self.connect_device); self.device_page.disconnect_requested.connect(self.disconnect_device)
        self.acquisition.emg_display_ready.connect(self.device_page.append_emg); self.acquisition.emg_display_ready.connect(self.realtime_inference_page.ingest_emg); self.acquisition.imu_display_ready.connect(self.device_page.update_imu)
        self.acquisition.statistics_ready.connect(self.device_page.update_stats); self.acquisition.packet_loss.connect(self.session.packet_loss)
        self.acquisition.recording_error.connect(self._recorder_error)
        page = self.experiment_page
        page.signal_check.clicked.connect(lambda: page.run_signal_check(self.device_page.recent_raw(5)))
        page.continue_anyway.clicked.connect(lambda: page.signal_result.setText("已由操作人员确认：忽略警告并继续"))
        page.start_requested.connect(self.start_session); page.stop_requested.connect(self.stop_session)
        page.pause_requested.connect(self.session.prompt.pause); page.resume_requested.connect(self.session.prompt.resume)
        page.skip_requested.connect(self.session.prompt.skip); page.repeat_requested.connect(self.session.prompt.repeat)
        page.bad_requested.connect(self._mark_bad); page.manual_mark_requested.connect(self.session.manual_mark)
        page.new_donning_requested.connect(self._new_donning); page.new_stage_requested.connect(self._new_stage)
        self.session.prompt.state_changed.connect(page.update_state)
        self.session.prompt.phase_scheduled.connect(page.update_phase_duration)
        self.session.prompt.finished.connect(self._protocol_finished)
        self.session.session_stopped.connect(self._session_stopped); self.session.recorder_error.connect(self._recorder_error)

    def connect_device(self, port: str, simulated: bool) -> None:
        if self.worker and self.worker.isRunning(): return
        if not simulated and not port:
            QMessageBox.warning(self, "没有串口", "请选择串口设备"); return
        worker = AcquisitionWorker(AcquisitionConfig(port, BAUDRATE, simulated), self)
        worker.packets_ready.connect(self.acquisition.ingest_packets); worker.stats_ready.connect(self.acquisition.update_statistics)
        worker.connected.connect(self._connected); worker.disconnected.connect(self._disconnected); worker.error.connect(self._worker_error)
        self.worker = worker; worker.start(); self.statusBar().showMessage("正在连接设备…")

    def disconnect_device(self) -> None:
        if self.worker: self.worker.stop()

    def _connected(self, name: str) -> None:
        reconnect = bool(self.connected_port); self.connected_port = name; self.device_page.set_connected(name, True)
        self.realtime_inference_page.set_connected(True)
        self.statusBar().showMessage(f"● 已连接 | {name} | 2000 Hz | 8 通道")
        LOGGER.info("serial connect: %s", name)
        if reconnect or self.session.active: self.session.device_reconnected()

    def _disconnected(self, message: str) -> None:
        was_connected = bool(self.connected_port); self.connected_port = ""; self.device_page.set_connected("", False)
        self.realtime_inference_page.set_connected(False)
        self.statusBar().showMessage(f"● 已断开 | {message}"); LOGGER.info("serial disconnect")
        if was_connected: self.session.device_disconnected()

    def _worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "串口错误", message); LOGGER.error("serial error: %s", message)

    def start_session(self, participant, info, protocol) -> None:
        if not self.connected_port:
            QMessageBox.warning(self, "设备未连接", "请先连接串口或模拟设备"); return
        info.serial_port = self.connected_port
        try: paths = self.session.start(participant, info, protocol)
        except Exception as exc: QMessageBox.critical(self, "无法开始实验", str(exc)); return
        self.experiment_page.prompt_panel.configure_task(
            self.session.prompt.sequence, protocol.labels, protocol
        )
        self.experiment_page.set_running(True); self.experiment_page.update_identity(participant.participant_id, info.session_id, protocol.name, 1, info.stage_id)
        self.device_page.set_recording(True)
        self.statusBar().showMessage(f"● 正在记录 | {paths.hdf5}")

    def stop_session(self) -> None:
        try: self.session.stop()
        except Exception as exc: QMessageBox.critical(self, "停止实验失败", str(exc))

    def _protocol_finished(self) -> None:
        """Safely close recording as soon as the final trial is complete."""
        if not self.session.active:
            return
        from PySide6.QtCore import QCoreApplication
        self.statusBar().showMessage("🎉 本次 Session 数据采集已全部完成，正在安全保存数据…")
        QCoreApplication.processEvents()
        self.stop_session()


    def _session_stopped(self, path: Path) -> None:
        self.experiment_page.set_running(False)
        aligned = self.session.last_aligned_path
        message = (f"🎉 采集完成！原始文件：{path} | Meta 对齐文件：{aligned}"
                   if aligned else f"🎉 采集完成！实验文件：{path}")
        self.statusBar().showMessage(message)
        self.device_page.set_recording(False)
        if path:
            self.data_check_page.open_session(Path(path))
            self.dataset_upload_page.refresh_plan()


    def _mark_bad(self, reason: str, note: str) -> None:
        try: self.session.mark_bad(reason, note)
        except Exception as exc: QMessageBox.warning(self, "无法标记", str(exc))

    def _new_donning(self) -> None:
        try:
            self.session.new_donning(); self.experiment_page.update_identity(self.session.participant.participant_id, self.session.info.session_id, self.session.protocol.name, self.session.donning_id, self.session.stage_id)
        except Exception as exc: QMessageBox.warning(self, "佩戴轮次", str(exc))

    def _new_stage(self, name: str) -> None:
        try:
            self.session.new_stage(name); self.experiment_page.update_identity(self.session.participant.participant_id, self.session.info.session_id, self.session.protocol.name, self.session.donning_id, self.session.stage_id)
        except Exception as exc: QMessageBox.warning(self, "实验阶段", str(exc))

    def _recorder_error(self, message: str) -> None:
        QMessageBox.critical(self, "HDF5 记录错误", message); LOGGER.error("HDF5 exception: %s", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            if self.session.active: self.session.stop()
            if not self.dataset_upload_page.shutdown():
                raise TimeoutError("数据上传线程未能停止")
            if not self.training_page.shutdown():
                raise TimeoutError("远端训练连接线程未能停止")
            if not self.realtime_inference_page.shutdown():
                raise TimeoutError("实时推理线程未能停止")
            worker = self.worker
            if worker is not None:
                if worker.isRunning():
                    worker.stop()
                if not worker.wait(3000):
                    raise TimeoutError("串口采集线程未能停止")
                self.worker = None
        except Exception as exc:
            QMessageBox.critical(self, "无法安全退出", str(exc)); event.ignore(); return
        LOGGER.info("app close"); event.accept()
