"""GUI source-frozen models: data-free loading and simulated worker lifecycle."""
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import pytest
from emgforce.inference.song_local import SongLocalRuntime,discover_song_models,SongOnlineDecision,LABELS

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'model_assets'


@pytest.mark.parametrize('arm',['pooled','rest'])
def test_frozen_bundle_policy_and_builtin_discovery(arm,tmp_path):
    folder=ASSETS/f'song_f0_{arm}_frozen_v2';r=SongLocalRuntime(folder)
    assert r.format_version==2 and r.hop_samples==10 and r.consecutive_frames==2 and r.online_event_threshold==0.
    assert r.make_bundle().metadata['song_frozen_source_v2']
    assert folder.resolve() in {v.directory for v in discover_song_models(tmp_path,include_builtin=True)}
    assert discover_song_models(tmp_path)==[]
    raw=np.random.default_rng(18).integers(-2000,2000,size=(70,8),dtype=np.int32)
    gap,frames=r.ingest(raw,np.arange(70));assert not gap and [v[0] for v in frames]==[49,59,69]
    assert all(q.dtype==np.float64 for _,q in frames)
    d=SongOnlineDecision(r.consecutive_frames);q=np.array([.9,.03,.03,.04])
    assert d.step(q,0)==(None,False) and d.step(q,0)==('fist',True)


def test_mismatched_frozen_policy_is_rejected(tmp_path):
    import hashlib
    folder=tmp_path/'bad';shutil.copytree(ASSETS/'song_f0_rest_frozen_v2',folder)
    artifact=folder/'song_f0_model.json';manifest=folder/'song_manifest.json'
    model=json.loads(artifact.read_text(encoding='utf8'));model['stream_policy']['consecutive_frames']=3
    artifact.write_text(json.dumps(model),encoding='utf8')
    m=json.loads(manifest.read_text(encoding='utf8'));m['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();manifest.write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError,match='确认规则'):SongLocalRuntime(folder)


def test_application_environment_native_stream_matches_frozen_reference():
    """Check the GUI environment's NumPy/SciPy without importing sklearn."""
    import hashlib
    h5py=pytest.importorskip('h5py')
    here=ROOT.parents[2]/'emgimu_classifier/benchmarks/song_real8'
    p=json.loads((here/'SONG_F0_STREAM_V1_PROTOCOL.json').read_text(encoding='utf8'))
    classes=json.loads((here/'SONG_F0_STREAM_V1_RESULTS.json').read_text(encoding='utf8'))['classes']
    paths={s:Path(p['source_folder'])/f'2026-09-18_{s}/session.h5' for s in p['target_sessions']}
    if not all(v.is_file() for v in paths.values()):pytest.skip('Native Song recordings unavailable')
    with np.load(here/'SONG_F0_STREAM_V1_EMISSIONS.npz',allow_pickle=False) as reference:
        for session,path in paths.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==p['hdf5_sha256'][session]
            with h5py.File(path) as f:raw=f['streams/emg/raw'][:]
            for name,arm in [('pooled','pooled_source_threshold'),('rest','rest_only_threshold')]:
                runtime=SongLocalRuntime(ASSETS/f'song_f0_{name}_frozen_v2');frames=[]
                for start in range(0,len(raw),4096):
                    gap,items=runtime.ingest(raw[start:start+4096],np.arange(start,min(start+4096,len(raw))))
                    assert not gap;frames.extend(items)
                np.testing.assert_array_equal([v[0] for v in frames],reference[session+'_ends'])
                cols=[LABELS.index(c) for c in classes]
                q=np.stack([v[1] for v in frames])
                np.testing.assert_allclose(q[:,cols],reference[session+'_'+arm+'_probabilities'],rtol=0,atol=1e-12)
                decision=SongOnlineDecision(2);confirmed=[]
                for probability in q:
                    label,_=decision.step(probability,0.)
                    confirmed.append(-1 if label is None else classes.index(label))
                np.testing.assert_array_equal(confirmed,reference[session+'_'+arm+'_confirmed'])


def test_page_load_start_pause_and_packet_loss_clear_stale_gesture(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from emgforce.ui.realtime_inference_page import RealtimeInferencePage
    app=QApplication.instance() or QApplication([])
    page=RealtimeInferencePage(tmp_path/'models')
    def wait(predicate):
        deadline=time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert predicate()
    try:
        page.refresh_models()
        folder=ASSETS/'song_f0_pooled_frozen_v2';index=page.model_combo.findData('song::'+str(folder.resolve()))
        assert index>=0;page.model_combo.setCurrentIndex(index);page.set_connected(True);page.load_selected_model()
        wait(lambda:page.bundle is not None)
        assert page.bundle.metadata['song_frozen_source_v2'] and page.threshold.value()==0.
        assert not page.threshold.isEnabled() and page.start_button.isEnabled() and not page.calibrate_button.isEnabled()
        assert '每 40 ms' in page.model_details.text()
        predictions=[];page.worker.prediction_ready.connect(predictions.append)
        page.start_recognition();raw=np.random.default_rng(7).integers(-2000,2000,size=(150,8),dtype=np.int32)
        page.ingest_emg(raw,np.arange(150));wait(lambda:bool(predictions))
        assert predictions[-1].output_sample_index==149
        page.current_gesture.setText('握拳');page.pause_recognition();wait(lambda:page.current_gesture.text()=='等待识别')
        page.start_recognition();page.ingest_emg(raw,np.arange(150,300));wait(lambda:len(predictions)>=2)
        page.current_gesture.setText('握拳');page.worker.notify_packet_loss(1);wait(lambda:page.current_gesture.text()=='等待识别')
        # A short post-loss segment cannot reuse the previous complete window.
        total=len(predictions);page.ingest_emg(raw[:49],np.arange(300,349))
        deadline=time.monotonic()+.15
        while time.monotonic()<deadline:app.processEvents();time.sleep(.01)
        assert len(predictions)==total
        page.ingest_emg(raw[49:60],np.arange(349,360));wait(lambda:len(predictions)>total)
        # An index gap has the same cold-window and display-reset contract.
        total=len(predictions);page.current_gesture.setText('握拳')
        page.ingest_emg(raw[:49],np.arange(400,449));wait(lambda:page.current_gesture.text()=='等待识别')
        assert len(predictions)==total
        page.ingest_emg(raw[49:60],np.arange(449,460));wait(lambda:len(predictions)>total)
        # Stop/disconnect keeps controls coherent; no real port is opened.
        page.set_connected(False);assert not page.start_button.isEnabled()
    finally:
        page._stop_realtime_worker();page.close();app.processEvents()
