"""Qt acquisition signals and actual classifier process share one registration."""
import os
from pathlib import Path
import time
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QFileDialog

import main_decision_v4
from emgforce.ui.main_window_v5 import MainWindowV5
from test_decision_page_v5 import signal


def test_actual_joint_page_registration_reload_manual_auto_quality_and_gap(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    window=main_decision_v4.create_window(tmp_path);window.resize(1450,950);window.show()
    page=window.realtime_inference_page;p=page.personal_session_panel
    assert isinstance(window,MainWindowV5) and window.main_tabs.currentWidget() is page
    assert p.model_choice.currentData()=='six'
    p.model_choice.setCurrentIndex(p.model_choice.findData('seven_joint'))
    p.python.setText('D:/miniconda/python.exe');p.user.setText('fixture');p.session.setText('long')
    errors=[]
    def wait(condition):
        deadline=time.monotonic()+20
        while not condition() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert condition(),p.status.text()
    def emit(raw,start):window.acquisition.emg_inference_ready.emit(raw,np.arange(start,start+len(raw)),None)
    def guided(kind,path):
        (p.temporal_enroll_button if kind=='personal' else p.temporal_session_button).click()
        wait(lambda:p.state['temporal']['capture'] is not None)
        assert not p.model_choice.isEnabled() and not p.decision.isEnabled() and not p.recognize_button.isEnabled()
        for i in range(4):
            label=p.state['temporal']['capture']['next_label'];p.temporal_trial_button.click()
            wait(lambda:p.state['temporal']['capture']['collecting'])
            emit(signal(label),i*500);wait(lambda:p.state['temporal']['capture']['samples']==500)
            p.temporal_end_button.click();wait(lambda:p.state['temporal']['capture']['completed']==i+1)
        assert p.progress.value()==4 and p.temporal_save_button.isEnabled()
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),''))
        p.temporal_save_button.click();wait(lambda:p.state.get('saved_joint_profile_path')==str(path))
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        p.worker.failed.connect(errors.append)
        assert p.state['schema']=='song_personal_session_stream_v6'
        assert '统一' in p.temporal_group.title() and p.personal.isHidden() and p.enroll_button.isHidden()
        pp=tmp_path/'p.zip';guided('personal',pp)
        assert p.state['personal_trials']==4 and p.state['temporal']['calibration_cost']['unique_native_calibration_trials']==4
        assert '长期 4 次 / 8.0 秒' in p.joint_cost.text() and '只计一次' in p.joint_cost.text()
        assert p.temporal_personal.text()==str(pp)
        assert p.shutdown();p.session.setText('current');p.load_button.click()
        wait(lambda:p.state is not None and p.state['temporal']['personal_profile_id'])
        p.worker.failed.connect(errors.append)
        sp=tmp_path/'s.zip';guided('session',sp)
        assert p.state['session_trials']==4 and '本会话 4 次 / 8.0 秒' in p.joint_cost.text()
        assert p.temporal_session.text()==str(sp)
        p.temporal_mode.setCurrentIndex(1);wait(lambda:p.state['temporal']['mode']=='manual')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        p.temporal_action_start.click();wait(lambda:p.state['temporal']['action_open'])
        emit(signal('fist',513),3000);wait(lambda:'probabilities' in p.state)
        p.temporal_action_end.click();wait(lambda:'temporal_events' in p.state)
        assert p.state['temporal_events'][0]['shared_registration'] and '手动标记边界' in p.temporal_prediction.text()
        p.quality.setCurrentIndex(1);wait(lambda:p.state['quality_mode']=='structural')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        p.temporal_action_start.click();wait(lambda:p.state['temporal']['action_open'])
        bad=signal('fist');bad[:,2]=0;emit(bad,4000);wait(lambda:'quality_rejected' in p.state)
        p.temporal_action_end.click();wait(lambda:'temporal_events' in p.state)
        assert p.state['temporal_events'][0]['rejected'] and '信号质量不足' in p.temporal_prediction.text()
        p.temporal_mode.setCurrentIndex(2);wait(lambda:p.state['temporal']['mode']=='auto')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        emit(np.concatenate((signal('neutral',400),signal('open_hand'),signal('neutral',200))),5000)
        wait(lambda:'temporal_events' in p.state)
        assert p.state['temporal_events'][0]['boundary_kind']=='estimated' and '自动估计边界' in p.temporal_prediction.text()
        window.acquisition.packet_loss.emit(1,6000,0);wait(lambda:p.state['mode']=='idle')
        assert not p.state['temporal']['action_open'] and p.temporal_prediction.text()=='完整动作结果：等待动作结束'
        # Cancelled registration and a disconnect must not replace either branch.
        identity=p.state['temporal']['personal_profile_id']
        p.temporal_enroll_button.click();wait(lambda:p.state['temporal']['capture'] is not None)
        page.set_connected(False);wait(lambda:p.state['temporal']['capture'] is None)
        assert p.state['temporal']['personal_profile_id']==identity and not p.temporal_enroll_button.isEnabled()
        page.set_connected(True)
        artifact=os.environ.get('EMGIMU_JOINT_GUI_ARTIFACT')
        if artifact:
            app.processEvents();path=Path(artifact);path.parent.mkdir(parents=True,exist_ok=True)
            assert p.grab().save(str(path))
        assert not errors,errors
        p.model_choice.setCurrentIndex(0)
        assert p.worker is None and not p.temporal_personal.text() and not p.enroll_button.isHidden()
    finally:
        assert page.shutdown();window.close();app.processEvents()


def test_shared_page_failed_profile_load_refreshes_idle_state_and_preserves_profiles(tmp_path):
    app=QApplication.instance() or QApplication([]);window=main_decision_v4.create_window(tmp_path)
    page=window.realtime_inference_page;p=page.personal_session_panel
    p.model_choice.setCurrentIndex(p.model_choice.findData('seven_joint'));p.python.setText('D:/miniconda/python.exe')
    p.user.setText('fixture');p.session.setText('current');errors=[]
    def wait(condition):
        deadline=time.monotonic()+15
        while not condition() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert condition(),p.status.text()
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        p.worker.failed.connect(errors.append)
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        p.temporal_personal.setText(str(tmp_path/'missing.zip'));p.temporal_apply_button.click()
        wait(lambda:bool(errors) and p.state['mode']=='idle')
        assert not p.worker._accept_emg and p.state['personal_profile_id'] is None
        assert '操作失败' in p.status.text() and p.temporal_prediction.text().startswith('完整动作结果：操作失败')
    finally:assert page.shutdown();window.close();app.processEvents()


def test_shared_launcher_and_failed_shutdown_preserve_model_choice(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);window=main_decision_v4.create_window(tmp_path)
    p=window.realtime_inference_page.personal_session_panel
    try:
        assert p.model_choice.findData('seven_joint')==3 and p.temporal_mode.currentData()=='off'
        installer=Path(main_decision_v4.__file__).with_name('install_decision_shortcut_v4.ps1').read_text(encoding='utf-8-sig')
        assert 'main_decision_v4.py' in installer
        with monkeypatch.context() as patch:
            patch.setattr(p,'shutdown',lambda *a:False);p.model_choice.setCurrentIndex(3)
            assert p.model_choice.currentData()=='six' and not p.enroll_button.isHidden()
    finally:window.close();app.processEvents()


def test_failed_shared_save_keeps_capture_and_allows_retry(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);window=main_decision_v4.create_window(tmp_path)
    page=window.realtime_inference_page;p=page.personal_session_panel
    p.model_choice.setCurrentIndex(3);p.python.setText('D:/miniconda/python.exe')
    p.user.setText('fixture');p.session.setText('long');errors=[]
    def wait(condition):
        deadline=time.monotonic()+15
        while not condition() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert condition(),p.status.text()
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        p.worker.failed.connect(errors.append)
        p.temporal_enroll_button.click();wait(lambda:p.state['temporal']['capture'] is not None)
        for i in range(4):
            label=p.state['temporal']['capture']['next_label'];p.temporal_trial_button.click()
            wait(lambda:p.state['temporal']['capture']['collecting'])
            window.acquisition.emg_inference_ready.emit(signal(label),np.arange(i*500,(i+1)*500),None)
            wait(lambda:p.state['temporal']['capture']['samples']==500)
            p.temporal_end_button.click();wait(lambda:p.state['temporal']['capture']['completed']==i+1)
        existing=tmp_path/'existing.zip';existing.write_bytes(b'existing user content')
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(existing),''))
        p.temporal_save_button.click();wait(lambda:bool(errors))
        assert existing.read_bytes()==b'existing user content'
        assert p.state['temporal']['capture']['completed']==4 and p.temporal_save_button.isEnabled()
        assert p.state['personal_profile_id'] is None and p.state['temporal']['personal_profile_id'] is None
        target=tmp_path/'retry.zip'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(target),''))
        p.temporal_save_button.click();wait(lambda:p.state.get('saved_joint_profile_path')==str(target))
        assert p.state['personal_trials']==4 and p.state['temporal']['capture'] is None
        assert p.temporal_personal.text()==str(target)
    finally:assert page.shutdown();window.close();app.processEvents()
