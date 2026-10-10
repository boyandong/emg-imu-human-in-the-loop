"""Actual Qt/acquisition-signal/subprocess full-action registration and decisions."""
import json
import os
from pathlib import Path
import time
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QFileDialog
from emgforce.ui.main_window_v4 import MainWindowV4
from emgforce.ui.realtime_inference_page_v5 import RealtimeInferencePageV5


def signal(label,n=500):
    rng=np.random.default_rng(sum(map(ord,label)))
    if label=='neutral':return rng.normal(0,2,(n,8))
    a=np.roll(np.arange(1,9),2*['fist','index_pinch','open_hand'].index(label))*100
    return rng.normal(size=(n,8))*a*np.sin(np.linspace(0,np.pi,n))[:,None]**2+rng.normal(0,2,(n,8))


def test_actual_application_full_action_profiles_manual_auto_and_gap(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);window=MainWindowV4(tmp_path,song_realtime=True)
    page=window.realtime_inference_page;assert isinstance(page,RealtimeInferencePageV5)
    p=page.personal_session_panel;p.model_choice.setCurrentIndex(2);p.python.setText('D:/miniconda/python.exe')
    p.user.setText('fixture');p.session.setText('long');errors=[]
    def wait(condition):
        until=time.monotonic()+20
        while not condition() and time.monotonic()<until:app.processEvents();time.sleep(.01)
        assert condition(),p.status.text()
    def emit(raw,start):window.acquisition.emg_inference_ready.emit(raw,np.arange(start,start+len(raw)),None)
    def guided(kind,path):
        (p.temporal_enroll_button if kind=='personal' else p.temporal_session_button).click()
        wait(lambda:p.state['temporal']['capture'] is not None)
        assert not p.model_choice.isEnabled() and not p.decision.isEnabled() and not p.recognize_button.isEnabled()
        for i in range(4):
            label=p.state['temporal']['capture']['next_label'];p.temporal_trial_button.click()
            wait(lambda:p.state['temporal']['capture']['collecting'])
            emit(signal(label),i*500)
            wait(lambda:p.state['temporal']['capture']['samples']==500)
            p.temporal_end_button.click();wait(lambda:p.state['temporal']['capture']['completed']==i+1)
        assert p.temporal_save_button.isEnabled()
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),''))
        p.temporal_save_button.click();wait(lambda:p.state.get('saved_temporal_profile_path')==str(path))
        audit=json.loads(Path(str(path)+'.capture.json').read_text())
        assert audit['raw_and_filtered_samples_saved'] and audit['settle_samples_discarded']==0
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        p.worker.failed.connect(errors.append)
        assert p.state['schema']=='song_personal_session_stream_v5' and p.state['temporal']['mode']=='off'
        pp=tmp_path/'full_personal.zip';guided('personal',pp);assert p.temporal_personal.text()==str(pp)
        assert p.shutdown();p.session.setText('current');p.load_button.click()
        wait(lambda:p.state is not None and p.state['temporal']['personal_profile_id'])
        p.worker.failed.connect(errors.append)
        sp=tmp_path/'full_current.zip';guided('session',sp);assert p.temporal_session.text()==str(sp)
        p.temporal_mode.setCurrentIndex(1);wait(lambda:p.state['temporal']['mode']=='manual')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        p.temporal_action_start.click();wait(lambda:p.state['temporal']['action_open'])
        emit(signal('fist',513),3000);wait(lambda:'probabilities' in p.state)
        p.temporal_action_end.click();wait(lambda:'temporal_events' in p.state)
        assert '手动标记边界' in p.temporal_prediction.text() and '动作结束后输出' in p.temporal_prediction.text()
        p.quality.setCurrentIndex(1);wait(lambda:p.state['quality_mode']=='structural')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        p.temporal_action_start.click();wait(lambda:p.state['temporal']['action_open'])
        bad=signal('fist');bad[:,2]=0;emit(bad,4000);wait(lambda:'quality_rejected' in p.state)
        p.temporal_action_end.click();wait(lambda:'temporal_events' in p.state)
        assert p.state['temporal_events'][0]['rejected'] and '信号质量不足' in p.temporal_prediction.text()
        p.temporal_mode.setCurrentIndex(2);wait(lambda:p.state['temporal']['mode']=='auto')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        raw=np.concatenate((signal('neutral',400),signal('open_hand'),signal('neutral',200)))
        emit(raw,5000);wait(lambda:'temporal_events' in p.state)
        assert p.state['temporal_events'][0]['boundary_kind']=='estimated' and '自动估计边界' in p.temporal_prediction.text()
        window.acquisition.packet_loss.emit(1,6000,0);wait(lambda:p.state['mode']=='idle')
        assert not p.state['temporal']['action_open'] and p.temporal_prediction.text()=='完整动作结果：等待动作结束'
        page.set_connected(False);assert not p.temporal_enroll_button.isEnabled()
        assert not errors,errors
        p.model_choice.setCurrentIndex(0);assert p.worker is None and not p.temporal_personal.text()
    finally:
        assert page.shutdown();window.close();app.processEvents()


def test_model_change_reverts_selection_when_worker_cannot_stop(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);window=MainWindowV4(tmp_path)
    p=window.realtime_inference_page.personal_session_panel
    try:
        monkeypatch.setattr(p,'shutdown',lambda *a:False)
        p.model_choice.setCurrentIndex(2);assert p.model_choice.currentData()=='six'
    finally:
        monkeypatch.undo();window.close();app.processEvents()
