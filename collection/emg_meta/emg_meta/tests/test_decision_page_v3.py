"""Real application entry, Qt controls and persistent classifier subprocess."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QFileDialog
from emgforce.ui import MainWindow
from emgforce.ui.main_window_v2 import MainWindowV2
from emgforce.ui.realtime_inference_page_v3 import RealtimeInferencePageV3


def test_default_application_guided_decision_profiles_and_quality(tmp_path, monkeypatch):
    python = Path('D:/miniconda/python.exe')
    if not python.is_file():
        pytest.skip('Classifier Python unavailable')
    app = QApplication.instance() or QApplication([])
    assert MainWindow is MainWindowV2
    window = MainWindow(tmp_path, song_realtime=True)
    page = window.realtime_inference_page
    assert isinstance(page, RealtimeInferencePageV3)
    assert window.main_tabs.currentWidget() is page
    p = page.personal_session_panel
    p.python.setText(str(python))
    p.user.setText('fixture')
    p.session.setText('long_recording')
    errors = []

    def wait(predicate):
        deadline = time.monotonic()+15
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), p.status.text()

    def guided(kind):
        (p.enroll_button if kind == 'personal' else p.session_button).click()
        wait(lambda: p.state.get('capture') is not None)
        assert not p.decision.isEnabled() and not p.quality.isEnabled()
        for i in range(4):
            p.trial_button.click()
            wait(lambda: p.state['capture']['collecting'])
            raw = np.random.default_rng(i+33).integers(-1000, 1000, size=(500, 8))
            # Exercise the replacement page's acquisition signal connections.
            window.acquisition.emg_inference_ready.emit(raw, np.arange(i*500, (i+1)*500), None)
            wait(lambda: p.state['capture']['completed'] == i+1)
        assert p.save_button.isEnabled()

    try:
        page.set_connected(True)
        p.load_button.click()
        wait(lambda: p.state is not None)
        p.worker.failed.connect(errors.append)
        assert p.decision.currentData() == 'baseline'
        assert not p.state['use_anchor'] and not p.state['use_session_routing']
        guided('personal')
        personal = tmp_path/'decision_personal.zip'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(personal), ''))
        p.save_button.click()
        wait(lambda: p.state['personal_trials'] == 4)
        audit = json.loads(Path(str(personal)+'.capture.json').read_text(encoding='utf8'))
        assert audit['contract_id'] == p.state['decision_policy_id']
        assert p.personal.text() == str(personal)
        p.decision.setCurrentIndex(2)
        wait(lambda: p.state['anchor_enabled'])
        assert not p.state['session_routing_enabled']
        assert p.shutdown()
        p.session.setText('current_recording')
        p.load_button.click()
        wait(lambda: p.state is not None and p.state['personal_trials'] == 4 and p.state['use_anchor'])
        p.worker.failed.connect(errors.append)
        guided('session')
        session = tmp_path/'decision_session.zip'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(session), ''))
        p.save_button.click()
        wait(lambda: p.state['session_trials'] == 4)
        assert p.state['anchor_enabled'] and p.state['session_routing_enabled']
        p.recognize_button.click()
        wait(lambda: p.state['mode'] == 'recognizing')
        good = np.random.default_rng(123).integers(-1000, 1000, size=(100, 8))
        window.acquisition.emg_inference_ready.emit(good, np.arange(2000, 2100), None)
        wait(lambda: 'probabilities' in p.state)
        assert p.state['output_sample_indices'] == list(range(2049, 2100, 10))
        p.decision.setCurrentIndex(1)
        wait(lambda: not p.state['use_anchor'] and p.state['use_session_routing'])
        assert p.state['mode'] == 'idle' and p.prediction.text() == '等待识别'
        p.decision.setCurrentIndex(2)
        wait(lambda: p.state['use_anchor'])
        p.quality.setCurrentIndex(1)
        wait(lambda: p.state['quality_mode'] == 'structural')
        p.recognize_button.click()
        wait(lambda: p.state['mode'] == 'recognizing')
        bad = good.copy()
        bad[:, 2] = 0
        window.acquisition.emg_inference_ready.emit(bad, np.arange(2200, 2300), None)
        wait(lambda: 'quality_rejected' in p.state)
        assert p.state['quality_rejected'][-1]
        assert p.prediction.text() == '信号质量不足 · 未知'
        window.acquisition.packet_loss.emit(1, 2300, 0)
        wait(lambda: p.state['mode'] == 'idle')
        assert p.prediction.text() == '等待识别'
        identity = p.state['session_profile_id']
        p.session_profile.setText(str(personal))
        p.apply_button.click()
        wait(lambda: p.status.text().startswith('操作失败'))
        p.worker.send('info')
        wait(lambda: not p.status.text().startswith('操作失败'))
        assert p.state['session_profile_id'] == identity
        assert len(errors) == 1 and 'identity' in errors[0]
        page.set_connected(False)
        wait(lambda: p.state['mode'] == 'idle')
        assert not p.recognize_button.isEnabled()
    finally:
        assert page.shutdown()
        window.close()
        app.processEvents()
