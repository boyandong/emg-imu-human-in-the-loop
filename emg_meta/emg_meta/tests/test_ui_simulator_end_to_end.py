from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from PySide6.QtCore import QEventLoop, QPoint, QRect, QTimer
from PySide6.QtWidgets import QApplication

from emgforce.experiment.models import ParticipantInfo, ProtocolConfig, SessionInfo
from emgforce.ui import MainWindow


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _wait(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def test_serial_port_label_keeps_device_name_and_connection_value() -> None:
    info = SimpleNamespace(
        device="COM8", description="USB-SERIAL CH340 (COM8)",
        manufacturer="wch.cn", product="USB Serial", serial_number="ABC123",
        vid=0x1A86, pid=0x7523, hwid="USB VID:PID=1A86:7523",
    )
    from emgforce.ui.device_page import DevicePage
    assert DevicePage._port_label(info) == "COM8  —  USB-SERIAL CH340"
    tooltip = DevicePage._port_tooltip(info)
    assert "制造商：wch.cn" in tooltip
    assert "VID:PID：1A86:7523" in tooltip


def test_window_closes_without_ever_connecting_a_device(tmp_path, monkeypatch) -> None:
    """Closing a fresh window must not call wait() on a missing worker."""
    app = QApplication.instance() or QApplication([])
    (tmp_path / "protocols").mkdir()
    window = MainWindow(tmp_path)
    window.show()
    app.processEvents()

    critical_messages: list[str] = []
    monkeypatch.setattr(
        "emgforce.ui.main_window.QMessageBox.critical",
        lambda _parent, _title, message: critical_messages.append(str(message)),
    )

    assert window.worker is None
    assert window.close() is True
    app.processEvents()
    assert not window.isVisible()
    assert critical_messages == []


def test_six_page_ui_and_simulator_session_round_trip(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir()
    shutil.copyfile(
        Path(__file__).parents[1] / "protocols" / "meta_discrete_7.json",
        protocol_dir / "meta_discrete_7.json",
    )
    window = MainWindow(tmp_path)
    window.show(); app.processEvents()
    assert window.main_tabs.count() == 6
    assert [window.main_tabs.tabText(i) for i in range(6)] == [
        "设备监测", "实验采集", "数据检查", "数据上传", "训练模型", "实时识别"]
    assert window.experiment_page.start.text() == "开始实验"
    assert window.experiment_page.signal_check.text() == "执行信号检查"
    assert window.experiment_page.dominant.currentText() == "未填写"
    assert window.experiment_page.protocol.currentText() == "meta_discrete_7_short_v2"
    assert window.data_check_page.path.placeholderText() == "请选择原始实验文件 session.h5"
    assert window.data_check_page.align_button.text() == "选择原始文件并自动对齐"
    assert window.dataset_upload_page.host.text() == "my-gpu-server"
    assert window.dataset_upload_page.upload_button.text() == "一键上传训练数据集"
    assert window.training_page.tmux_session.text() == "emgforce_train_conv_lstm"
    assert window.training_page.algorithm.count() == 2
    assert window.realtime_inference_page.remote_algorithm.count() == 2
    assert window.training_page.start_button.text() == "在 tmux 中开始训练"
    assert window.realtime_inference_page.download_remote_button.text() == "下载所选模型"
    inference_page = window.realtime_inference_page
    assert inference_page.calibration_instruction.text() == "等待开始校准"
    inference_page._calibration_progress(0, 16_000)
    assert inference_page.calibration_instruction.text() == "自然静息"
    inference_page._calibration_progress(14_000, 16_000)
    assert inference_page.calibration_instruction.text() == "中指保持"
    assert window.prompt_window.windowTitle() == "参与者动作提示"
    action_buttons = (
        window.experiment_page.start, window.experiment_page.pause,
        window.experiment_page.resume, window.experiment_page.skip,
        window.experiment_page.repeat, window.experiment_page.bad,
        window.experiment_page.manual, window.experiment_page.stop,
    )
    assert len({button.y() for button in action_buttons}) == 1
    device = window.device_page
    assert len(device.emg_cards) == 8
    assert device.render_timer.interval() == 50
    assert device.emg_cards[0].plot.getAxis("left").isVisible() is False
    assert device.emg_cards[0].plot.getAxis("bottom").isVisible() is False
    assert device.emg_cards[-1].plot.getAxis("bottom").isVisible() is True
    assert device.emg_cards[0].plot.getViewBox().state["mouseEnabled"] == [False, False]
    assert len(device.gyro_card.curves) == 3
    assert len(device.accel_card.curves) == 3
    review_page = window.data_check_page
    assert review_page.qc_card_title.text() == "全自动标签质检"
    assert not hasattr(review_page, "review_raw_plot")
    for width, height in ((1500, 950), (1920, 1080), (2560, 1440)):
        window.resize(width, height); app.processEvents()
        for spin in (device.window_spin, device.emg_range_spin):
            parent = spin.parentWidget()
            rect = QRect(spin.mapTo(parent, QPoint(0, 0)), spin.size())
            assert parent.contentsRect().contains(rect)
    window.connect_device("__SIM__", True)
    _wait(120)
    assert window.connected_port == "模拟设备"
    assert len(device.imu_time) > 0
    assert all(len(values) > 0 for values in device.gyro_values)
    assert all(len(values) > 0 for values in device.accel_values)
    labels = [
        "thumb_tap", "thumb_swipe_left", "thumb_swipe_right",
        "thumb_swipe_up", "thumb_swipe_down", "index_hold", "middle_hold",
    ]
    protocol = ProtocolConfig(
        "meta_tiny", labels, 1, countdown_sec=.01, prompt_duration_sec=.02,
        rest_min_sec=.01, rest_max_sec=.01, randomize=False,
        hold_min_sec=.02, hold_max_sec=.02, release_display_sec=.01,
        timed_null_actions=["null_finger_snap"], timed_null_repetitions=1,
        timed_null_prompt_duration_sec=.01,
        continuous_null_blocks=[{
            "name": "null_typing", "instruction": "请自然打字", "duration_sec": .03,
        }],
    )
    window.start_session(ParticipantInfo("P_TEST"),
                         SessionInfo("S_TEST", "e2e", "meta_tiny"), protocol)
    _wait(600)
    assert window.session.active is False
    assert window.experiment_page.stop.isEnabled() is False
    _wait(100)
    h5_path = next((tmp_path / "data" / "P_TEST").rglob("session.h5"))
    with h5py.File(h5_path, "r") as h5:
        for removed in ("wrist_circumference", "age", "gender", "operator",
                        "notes", "session_notes"):
            assert removed not in h5["meta"].attrs
        assert h5["streams/emg/raw"].shape[1] == 8
        assert len(h5["streams/emg/raw"]) > 0
        assert len(h5["streams/imu/gyro"]) > 0
        assert len(h5["trials"]) == 9
        assert len(h5["cue_events"]) == 9
        assert all(not value.decode().startswith("null_")
                   for value in h5["cue_events"]["name"])
        assert np.all(h5["cue_events"]["scheduled_monotonic_ns"] > 0)
        assert np.all(h5["cue_events"]["emitted_monotonic_ns"] > 0)
        assert "prompts" not in h5
        kinds = [value.decode() for value in h5["events"]["event_type"]]
        assert "PROMPT_START" in kinds
        assert "PROMPT_END" in kinds
        assert kinds[-1] == "SESSION_END"
    aligned_path = h5_path.with_name("session_meta_aligned.hdf5")
    assert aligned_path.exists()
    with h5py.File(aligned_path, "r") as h5:
        assert h5["data"].attrs["task"] == "discrete_gestures"
        assert h5["data"].attrs["preprocessing_version"] == "meta_8ch_v1"
        assert h5["data"].attrs["highpass_hz"] == 40.0
        assert h5["data"].attrs["source_emg_units"] == "device_raw_counts"
        assert h5["data"].attrs["emg_units"] == "normalized_device_counts"
        assert len(h5["alignment_events"]) == 9
        assert h5["meta"].attrs["training_label_shift_applied"] == 0
        assert h5["alignment_events"].attrs["algorithm"] == \
            "session_rerp_template_beam_v3"
        np.testing.assert_allclose(
            h5["meta"].attrs["training_target_pulse_window_sec"], [0.08, 0.12])
    assert pd.read_hdf(aligned_path, "stages")["name"].tolist() == [
        "default", "null_timed_snap_flick", "null_typing",
    ]
    assert all(not name.startswith("null_")
               for name in pd.read_hdf(aligned_path, "prompts")["name"])
    assert not hasattr(review_page, "tabs")
    assert review_page.review_store is not None
    assert "自动处理完成" in review_page.qc_overview.text()
    assert "训练就绪" in review_page.training_status.text()
    assert review_page.qc_total_value.text() == "9"
    corpus_path = tmp_path / "data" / "discrete_gestures_corpus.csv"
    corpus = pd.read_csv(corpus_path)
    assert len(corpus) == 1
    assert corpus.iloc[0]["split"] == "train"
    assert corpus.iloc[0]["dataset"].endswith("/session_meta_aligned.hdf5")
    assert (tmp_path / "data" / "training_manifest.json").exists()
    # Selecting the aligned export directly must transparently resolve the raw
    # sibling instead of raising an HDF5 "component not found" dialog.
    review_page.open_session(aligned_path); app.processEvents()
    assert Path(review_page.path.text()) == h5_path
    assert review_page.review_store is not None
    assert review_page.review_store.aligned_path == aligned_path
    assert not hasattr(review_page, "adjust_p1")
    manual_dir = tmp_path / "manual_alignment"
    manual_dir.mkdir()
    manual_source = manual_dir / "session.h5"
    shutil.copyfile(h5_path, manual_source)
    review_page.start_automatic_alignment(manual_source)
    for _ in range(100):
        if review_page._alignment_worker is None:
            break
        _wait(20)
    assert review_page._alignment_worker is None
    assert (manual_dir / "session_meta_aligned.hdf5").exists()
    assert "自动对齐与质检完成" in review_page.alignment_status.text()
    assert "自动处理完成" in review_page.qc_overview.text()
    window.disconnect_device(); _wait(100)
    window.close(); app.processEvents()
