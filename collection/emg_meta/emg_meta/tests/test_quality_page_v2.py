"""Real Qt subprocess flow: opt-in quality, stale-label reset and raw replay."""
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication,QFileDialog
from emgforce.ui.realtime_inference_page_v2 import RealtimeInferencePageV2
from emgforce.inference.song_local import SongLocalRuntime


def test_current_app_quality_flow_and_raw_capture_replay(tmp_path,monkeypatch):
    python=Path('D:/miniconda/python.exe')
    if not python.is_file():pytest.skip('Classifier Python unavailable')
    from emgforce.ui import main_window
    assert main_window.RealtimeInferencePage is RealtimeInferencePageV2
    app=QApplication.instance() or QApplication([]);page=RealtimeInferencePageV2(tmp_path/'models');p=page.personal_session_panel
    p.python.setText(str(python));p.user.setText('fixture');p.session.setText('quality_recording')
    def wait(predicate):
        deadline=time.monotonic()+10
        while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert predicate(),p.status.text()
    try:
        page.set_connected(True);p.load_button.click();wait(lambda:p.state is not None)
        assert p.state['quality_mode']=='off'
        p.quality.setCurrentIndex(1);wait(lambda:p.state['quality_mode']=='structural')
        p.recognize_button.click();wait(lambda:p.state['mode']=='recognizing')
        raw=np.random.default_rng(25).integers(-1000,1000,size=(100,8));raw[:,2]=0
        page.ingest_emg(raw,np.arange(100));wait(lambda:'quality_rejected' in p.state)
        assert p.state['quality_rejected'][-1] and p.prediction.text()=='信号质量不足 · 未知'
        p.enroll_button.click();wait(lambda:p.state.get('capture') is not None)
        assert not p.quality.isEnabled()
        p.trial_button.click();wait(lambda:p.state['capture']['collecting'])
        bad=np.random.default_rng(9).integers(-1000,1000,size=(500,8));bad[:,2]=0
        page.ingest_emg(bad,np.arange(500));wait(lambda:p.state.get('calibration_rejected'))
        assert p.state['capture']['completed']==0 and '未计入' in p.guidance.text()
        good=np.random.default_rng(19).integers(-1000,1000,size=(500,8))
        for i in range(4):
            p.trial_button.click();wait(lambda:p.state['capture']['collecting'])
            page.ingest_emg(good,np.arange(500+i*500,1000+i*500));wait(lambda:p.state['capture']['completed']==i+1)
        path=tmp_path/'quality_personal.zip'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(path),''))
        p.save_button.click();wait(lambda:p.state['personal_trials']==4)
        assert Path(str(path)+'.raw_windows.npz').is_file()
        audit=json.loads(Path(str(path)+'.capture.json').read_text(encoding='utf8'))
        assert audit['quality_mode']=='structural' and audit['raw_pre_software_highpass']
        assert not audit['physical_validation_proven']
        raw_windows=np.random.default_rng(41).integers(-1000,1000,size=(4,50,8));raw_windows[:,:,2]=0
        asset=Path(__file__).resolve().parents[1]/'model_assets/song_f0_rest_frozen_v2'
        runtime=SongLocalRuntime(asset);filtered=[]
        for values in raw_windows:runtime.reset();filtered.append(runtime._filter_block(values).astype(np.float32))
        ids=np.array(['held0','held1','held2','held3']);offsets=np.zeros(4,int)
        fp=tmp_path/'filtered.npz';rp=tmp_path/'raw.npz'
        for output,values in ((fp,np.stack(filtered)),(rp,raw_windows)):
            np.savez(output,emg=values,sample_rate_hz=np.array(250.),trial_ids=ids,window_offsets=offsets)
        paths=iter((str(fp),str(rp)))
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(next(paths),''))
        p.replay_button.click();wait(lambda:'prediction' in p.state)
        assert p.state['prediction']['quality_decision']['rejected']==[True]*4
        assert '4 个质量拒识' in p.prediction.text()
        page.set_connected(False);wait(lambda:p.state['mode']=='idle')
        assert p.prediction.text()=='等待识别'
    finally:assert page.shutdown();page.close();app.processEvents()
