"""Real separate-process lifecycle; portable prediction requires no source archive."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from emgimu.feature_bank.extended_window_cli_v1 import load_extended_workflow
from emgimu.feature_bank.core import FeatureBatch

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'benchmarks/song_real8'


def arguments():
    return ['--package',str(HERE/'song_extended_window_v1/source_bank.pkl'),
        '--policy',str(HERE/'song_extended_window_v1/policy.json'),
        '--acceptance',str(HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json'),
        '--gate-package',str(HERE/'song_raw_quality_v1/source_gate.pkl'),'--gate-results',str(HERE/'SONG_RAW_QUALITY_V1_RESULTS.json'),
        '--user','fixture','--preprocessing','song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1',
        '--channels',*[f'CH{i+1}' for i in range(8)]]


def test_cli_personal_session_predict_and_separate_raw_unknown(tmp_path):
    rng=np.random.default_rng(18);classes=('fist','index_pinch','neutral','open_hand')
    paths=[];arrays=[]
    for prefix in ('long','current','held'):
        x=rng.normal(size=(8,50,8)).astype(np.float32)*100
        ids=np.array([f'{prefix}:{c}' for c in classes for _ in range(2)]);offsets=np.tile([0,1],4)
        path=tmp_path/(prefix+'.npz');labels=tmp_path/(prefix+'.json')
        np.savez(path,emg=x,sample_rate_hz=np.array(250.),trial_ids=ids,window_offsets=offsets)
        labels.write_text(json.dumps({t:c for t,c in zip(ids,np.repeat(classes,2))}))
        paths.append((path,labels));arrays.append((x,ids,offsets))
    pp=tmp_path/'personal.zip';sp=tmp_path/'session.zip';output=tmp_path/'prediction.json'
    def run(action,options,success=True):
        result=subprocess.run([sys.executable,'-m','emgimu.feature_bank.extended_window_cli_v1',action,*arguments(),*options],cwd=ROOT,text=True,capture_output=True)
        assert (result.returncode==0)==success,result.stderr
        return result
    run('enroll',['--windows',str(paths[0][0]),'--labels',str(paths[0][1]),'--session','long','--output',str(pp)])
    run('session',['--windows',str(paths[1][0]),'--labels',str(paths[1][1]),'--session','current','--personal-profile',str(pp),'--output',str(sp)])
    options=['--windows',str(paths[2][0]),'--session','current','--personal-profile',str(pp),'--session-profile',str(sp),'--output',str(output)]
    run('predict',options+['--enable-anchor','--enable-session-routing']);saved=output.read_bytes();r=json.loads(saved)
    w=load_extended_workflow(HERE/'song_extended_window_v1/source_bank.pkl',
        HERE/'song_extended_window_v1/policy.json',HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json',HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    p=w.load_profile(pp,user_id='fixture');s=w.load_profile(sp,user_id='fixture',session_id='current',personal=p)
    x,ids,o=arrays[2];expected=w.predict(FeatureBatch(x,250.),ids,window_offsets=o,user_id='fixture',session_id='current',personal=p,session=s,
        observed_channel_ids=w.channels,preprocessing_id=w.preprocessing_id)
    np.testing.assert_array_equal(r['probabilities'],expected['probabilities']);assert r['anchor_enabled'] and r['session_routing_enabled'];assert np.asarray(r['spectral_context']['window_minus_long']).shape==(8,32)
    run('predict',options,False);assert output.read_bytes()==saved
    rejected=run('predict',options+['--labels',str(paths[2][1])],False);assert 'never accepts labels' in rejected.stderr
    raw=tmp_path/'raw.npz';rx=x.copy();rx[:,:,2]=0
    np.savez(raw,emg=rx,sample_rate_hz=np.array(250.),trial_ids=ids,window_offsets=o)
    options=options[:-1]+[str(tmp_path/'unknown.json'),'--raw-windows',str(raw),'--quality-mode','structural']
    run('predict',options+['--enable-anchor','--enable-session-routing']);unknown=json.loads((tmp_path/'unknown.json').read_text())
    assert unknown['predicted_labels']==['Unknown']*4 and unknown['quality_decision']['rejected']==[True]*4
