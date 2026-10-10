"""Actual subprocess lifecycle on native-sized eight-channel complete input."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from test_personal_temporal_bouts_v1 import batch


def test_temporal_cli_roundtrip_axes_coverage_and_no_overwrite(tmp_path):
    config=dict(source_bank_id='fixture',sample_rate_hz=250.,channel_ids=[f'CH{i+1}' for i in range(8)],
        preprocessing_id='fixture',class_names=['fist','index_pinch','neutral','open_hand'])
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config),encoding='utf8')
    common=['--config',str(cp),'--config-sha256',hashlib.sha256(cp.read_bytes()).hexdigest(),'--user','u']
    def run(action,args,ok=True):
        r=subprocess.run([sys.executable,'-m','emgimu.feature_bank.personal_temporal_cli_v1',action,*common,*args],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        assert (r.returncode==0)==ok,r.stderr
        return r
    paths=[]
    for name in ('long','current','held'):
        b,labels=batch(name,shots=1);path=tmp_path/(name+'.npz');lp=tmp_path/(name+'.json')
        np.savez(path,samples=np.concatenate(b.sequences),offsets=np.r_[0,np.cumsum([len(x) for x in b.sequences])],
            trial_ids=np.array(b.trial_ids),recording_ids=np.array(b.recording_ids),starts=np.array(b.starts),
            sample_rate_hz=np.array(b.sample_rate_hz),channel_ids=np.array(b.channel_ids),
            preprocessing_id=np.array(b.preprocessing_id),boundary_kind=np.array(b.boundary_kind))
        lp.write_text(json.dumps(labels),encoding='utf8');paths.append((path,lp))
    pp=tmp_path/'personal.zip';sp=tmp_path/'session.zip';out=tmp_path/'result.json'
    run('enroll',['--bouts',str(paths[0][0]),'--labels',str(paths[0][1]),'--session','long','--output',str(pp)])
    run('session',['--bouts',str(paths[1][0]),'--labels',str(paths[1][1]),'--session','current','--personal-profile',str(pp),'--output',str(sp)])
    options=['--bouts',str(paths[2][0]),'--session','current','--personal-profile',str(pp),'--session-profile',str(sp),'--output',str(out)]
    run('predict',options)
    r=json.loads(out.read_text(encoding='utf8'));saved=out.read_bytes()
    assert r['session_template_enabled'] and r['certified_full_coverage']
    assert len(r['arms']['DTW_blended'])==4 and not r['physical_validation_proven']
    run('predict',options,False);assert out.read_bytes()==saved
    rejected=run('predict',options+['--labels',str(paths[2][1])],False)
    assert 'never accepts labels' in rejected.stderr
