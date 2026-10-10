"""Actual Qt controls and persistent classifier subprocess; no physical port."""
import os
from pathlib import Path
import time
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QFileDialog
from emgforce.ui.realtime_inference_page import RealtimeInferencePage


def test_page_guided_personal_session_save_reload_replay_and_disconnect(tmp_path,monkeypatch):
    python=Path('D:/miniconda/python.exe')
    if not python.is_file():pytest.skip('Classifier Python not configured')
    app=QApplication.instance() or QApplication([])
    page=RealtimeInferencePage(tmp_path/'models');p=page.personal_session_panel
    p.python.setText(str(python));p.user.setText('fixture');p.session.setText('recording1')
    def wait(predicate):
        deadline=time.monotonic()+10
        while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert predicate(),p.status.text()
    def guided(kind):
        button=p.enroll_button if kind=='personal' else p.session_button
        button.click();wait(lambda:p.state and p.state.get('capture') is not None)
        assert not p.recognize_button.isEnabled()
        for i in range(4):
            assert p.trial_button.isEnabled()
            p.trial_button.click();wait(lambda:p.state['capture']['collecting'])
            start=i*500
            page.ingest_emg(np.random.default_rng(i+7).integers(-1000,1000,size=(500,8)),np.arange(start,start+500))
            wait(lambda:p.state['capture']['completed']==i+1)
        assert p.save_button.isEnabled()
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        assert page.worker is None and page.bundle is None
        assert not p.user.isEnabled() and p.recognize_button.isEnabled()
        guided('personal')
        personal=tmp_path/'personal.zip'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(personal),''))
        p.save_button.click();wait(lambda:p.state['personal_trials']==4)
        assert p.personal.text()==str(personal) and personal.is_file()
        assert p.shutdown();p.session.setText('recording2');p.load_button.click()
        wait(lambda:p.state is not None and p.state['personal_trials']==4)
        guided('session')
        current=tmp_path/'current.zip'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(current),''))
        p.save_button.click();wait(lambda:p.state['session_trials']==4)
        assert p.session_profile.text()==str(current)
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        page.ingest_emg(np.random.default_rng(18).integers(-1000,1000,size=(100,8)),np.arange(1000,1100))
        wait(lambda:'probabilities' in p.state)
        assert p.state['output_sample_indices']==[1049,1059,1069,1079,1089,1099]
        assert len(p.state['weights'])==6 and '等待识别' not in p.prediction.text()
        page.set_connected(False);wait(lambda:p.state['mode']=='idle')
        assert p.prediction.text()=='等待识别' and not p.recognize_button.isEnabled()
        windows=tmp_path/'held.npz'
        np.savez(windows,emg=np.random.default_rng(33).normal(size=(4,50,8)).astype(np.float32),
            sample_rate_hz=np.array(250.),trial_ids=np.array(['held1','held2','held3','held4']),window_offsets=np.zeros(4,int))
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(windows),''))
        p.replay_button.click();wait(lambda:'prediction' in p.state)
        assert len(p.state['prediction']['trial_ids'])==4
        # Bad session file cannot discard the active, validated profile pair.
        identity=p.state['session_profile_id'];p.session_profile.setText(str(personal));p.apply_button.click()
        wait(lambda:p.status.text().startswith('操作失败'))
        p.worker.send('info');wait(lambda:not p.status.text().startswith('操作失败'))
        assert p.state['session_profile_id']==identity
        # Returning to an ordinary model stops the classifier process first.
        p.session_profile.setText(str(current))
        key=next(i for i in range(page.model_combo.count()) if str(page.model_combo.itemData(i)).startswith('song::'))
        page.model_combo.setCurrentIndex(key);page.load_selected_model()
        wait(lambda:page.bundle is not None)
        assert p.worker is None
    finally:
        assert page.shutdown();page.close();app.processEvents()


def test_no_identity_or_disconnected_capture_is_rejected(tmp_path):
    app=QApplication.instance() or QApplication([])
    page=RealtimeInferencePage(tmp_path/'models');p=page.personal_session_panel
    try:
        p.load_button.click();assert p.worker is None and '请输入' in p.status.text()
        assert not p.enroll_button.isEnabled() and not p.recognize_button.isEnabled()
    finally:page.shutdown();page.close();app.processEvents()
