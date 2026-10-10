"""Separate-process joint profiles with strict native input and label isolation."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_joint_bout_workflow_v1 import registered, native_fixture, ROOT, SONG


def write_bouts(path,batch,**extra):
    values=dict(samples=np.concatenate(batch.sequences),offsets=np.r_[0,np.cumsum(list(map(len,batch.sequences)))],
        trial_ids=np.asarray(batch.trial_ids),recording_ids=np.asarray(batch.recording_ids),starts=np.asarray(batch.starts),
        sample_rate_hz=batch.sample_rate_hz,channel_ids=np.asarray(batch.channel_ids),
        preprocessing_id=batch.preprocessing_id,boundary_kind=batch.boundary_kind)
    np.savez(path,**values,**extra)


def invoke(args,cwd):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return subprocess.run([sys.executable,'-m','emgimu.feature_bank.joint_bout_cli_v1',*map(str,args)],
        cwd=cwd,env=env,capture_output=True,text=True,timeout=60)


def common(action,bouts,output,session='current'):
    return [action,'--source-root',SONG,'--bouts',bouts,'--user','fixture','--session',session,'--output',output]


def test_separate_process_complete_joint_lifecycle_and_native_input_saved(registered,tmp_path):
    w,p,s,q,qr=registered
    source=[SONG/'song_extended_window_v1/source_bank.pkl',SONG/'song_raw_quality_v1/source_gate.pkl']
    fingerprints=[hashlib.sha256(f.read_bytes()).hexdigest() for f in source]
    bp,rp,bs,rs,bq,rq=[tmp_path/(n+'.npz') for n in ('long','long_raw','current','current_raw','query','query_raw')]
    for path,batch in zip((bp,rp,bs,rs,bq,rq),(p.calibration,p.raw_calibration,s.calibration,s.raw_calibration,q,qr)):
        write_bouts(path,batch)
    yp,ys=tmp_path/'long_labels.json',tmp_path/'current_labels.json'
    yp.write_text(json.dumps(dict(p.calibration_labels)),encoding='utf8')
    ys.write_text(json.dumps(dict(s.calibration_labels)),encoding='utf8')
    pp,sp=tmp_path/'personal.zip',tmp_path/'session.zip'
    commands=[common('enroll',bp,pp,'long')+['--raw-bouts',rp,'--labels',yp],
        common('session',bs,sp)+['--raw-bouts',rs,'--labels',ys,'--personal-profile',pp]]
    for cmd in commands:
        r=invoke(cmd,tmp_path);assert r.returncode==0,r.stderr
    lp=w.load_profile(pp,user_id='fixture');ls=w.load_profile(sp,user_id='fixture',session_id='current',personal=lp)
    # Independently registering an equivalent NPZ need not reproduce opaque
    # frozen child-profile IDs (their pickle hashes retain reference identity).
    # The saved IDs must persist; native contents, costs and predictions agree.
    for actual,expected in ((lp,p),(ls,s)):
        assert actual.calibration.trial_ids==expected.calibration.trial_ids
        assert actual.calibration_cost==expected.calibration_cost
        for x,y in zip(actual.calibration.sequences,expected.calibration.sequences):np.testing.assert_array_equal(x,y)
    for mode in ('off','structural','soft'):
        output=tmp_path/(mode+'.json')
        cmd=common('predict',bq,output)+['--raw-bouts',rq,'--personal-profile',pp,'--session-profile',sp,'--quality-mode',mode]
        r=invoke(cmd,tmp_path);assert r.returncode==0,r.stderr
        result=json.loads(output.read_text(encoding='utf8'))
        expected=w.predict(q,personal=p,session=s,user_id='fixture',session_id='current',raw_batch=qr,quality_mode=mode)
        np.testing.assert_array_equal(result['probabilities'],expected['probabilities'])
        assert result['labels']==list(expected['labels']) and result['joint_calibration_trials']==12
        original=output.read_bytes();r=invoke(cmd,tmp_path)
        assert r.returncode==2 and output.read_bytes()==original
    assert fingerprints==[hashlib.sha256(f.read_bytes()).hexdigest() for f in source]


@pytest.mark.parametrize('problem',['predict_labels','predict_reservations','missing_parent','enroll_prediction_flags',
    'embedded_labels','reserved_record','raw_axes','invalid_labels','invalid_reservations'])
def test_cli_invalid_lifecycle_or_inputs_never_create_output(registered,tmp_path,problem):
    w,p,_,q,raw=registered
    b,y,r=native_fixture(w,'new',shots=1)
    bp,rp,yp,pp,output=[tmp_path/n for n in ('input.npz','raw.npz','labels.json','personal.zip','result')]
    w.save_profile(p,pp);write_bouts(bp,b);write_bouts(rp,r);yp.write_text(json.dumps(y),encoding='utf8')
    args=common('enroll',bp,output,'new')+['--labels',yp]
    if problem in ('predict_labels','predict_reservations'):
        args=common('predict',bp,output)+['--personal-profile',pp,
            '--labels' if problem=='predict_labels' else '--reserved-evaluation',yp]
    elif problem=='missing_parent':args=common('predict',bp,output)
    elif problem=='enroll_prediction_flags':args+=['--disable-anchor']
    elif problem=='embedded_labels':write_bouts(bp,b,labels=np.arange(4))
    elif problem=='reserved_record':
        path=tmp_path/'reserved.json';path.write_text(json.dumps(dict(trial_ids=[],recording_ids=[b.recording_ids[0]])),encoding='utf8')
        args+=['--reserved-evaluation',path]
    elif problem=='raw_axes':
        write_bouts(rp,replace(r,starts=(1,)*4));args+=['--raw-bouts',rp]
    elif problem=='invalid_labels':yp.write_text('[]',encoding='utf8')
    elif problem=='invalid_reservations':args+=['--reserved-evaluation',yp]
    result=invoke(args,tmp_path)
    assert result.returncode==2 and not output.exists(),result.stderr
