import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.frozen_emg_bank_cli_v1 import main
from emgimu.feature_bank.core import FeatureBatch

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'feature_bank/models/epn_emg_calibrated_bank_v1.pkl'
ACCEPTANCE=ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json'


def window_file(tmp_path,trials=1,rate=200.,extra=None):
    x=(np.arange(trials*4*40*8,dtype=np.float32)%11).reshape(trials*4,40,8)
    ids=np.repeat([f'cli_trial_{i}' for i in range(trials)],4)
    offsets=np.tile(np.arange(4),trials)
    p=tmp_path/'windows.npz'
    np.savez(p,emg=x,sample_rate_hz=np.array(rate),trial_ids=ids,window_offsets=offsets,**(extra or {}))
    return p,x,ids,offsets


def args(action,windows,output,user='cli_user'):
    return [action,'--package',str(PACKAGE),'--acceptance',str(ACCEPTANCE),'--windows',str(windows),'--user',user,'--output',str(output)]


def test_offline_prediction_and_profile_roundtrip_match_direct_api(tmp_path):
    windows,x,ids,offsets=window_file(tmp_path,6)
    labels=tmp_path/'labels.json';labels.write_text(json.dumps({f'cli_trial_{i}':i for i in range(6)}),encoding='utf8')
    profile=tmp_path/'profile.pkl'
    before=hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    assert main(args('calibrate',windows,profile)+['--labels',str(labels)])==0
    bank=pickle.loads(PACKAGE.read_bytes());state=pickle.loads(profile.read_bytes())
    expected=bank.calibrate_user(FeatureBatch(x,200.),ids,{f'cli_trial_{i}':i for i in range(6)},window_offsets=offsets,user_id='cli_user')
    assert state==expected
    windows,x,ids,offsets=window_file(tmp_path)
    ids=np.full(4,'independent_eval');np.savez(windows,emg=x,sample_rate_hz=200.,trial_ids=ids,window_offsets=offsets)
    for name,options,profile_state in [('zero',[],None),('calibrated',['--profile',str(profile)],state)]:
        output=tmp_path/(name+'.json')
        assert main(args('predict',windows,output)+options)==0
        r=json.loads(output.read_text(encoding='utf8'))
        direct=bank.predict(FeatureBatch(x,200.),ids,window_offsets=offsets,user_id='cli_user',user_state=profile_state)
        np.testing.assert_array_equal(r['probabilities'],direct['probabilities'])
        assert r['predicted_labels']==list(direct['predicted_labels']) and r['trial_ids']==['independent_eval']
    assert hashlib.sha256(PACKAGE.read_bytes()).hexdigest()==before
    with pytest.raises(SystemExit):main(args('predict',windows,tmp_path/'wrong_user.json','different_user')+['--profile',str(profile)])
    assert not (tmp_path/'wrong_user.json').exists()


@pytest.mark.parametrize('case',['rate','embedded_labels','predict_labels','reserved_overlap','missing_labels','checkpoint_hash','existing_output'])
def test_cli_rejects_incompatible_or_leaking_input_without_writing_output(tmp_path,case):
    windows,x,ids,offsets=window_file(tmp_path,6,rate=250. if case=='rate' else 200.,extra={'labels':np.arange(24)} if case=='embedded_labels' else None)
    output=tmp_path/'result.json';invocation=args('predict',windows,output)
    labels=tmp_path/'labels.json';labels.write_text(json.dumps({f'cli_trial_{i}':i for i in range(6)}),encoding='utf8')
    if case=='predict_labels':invocation+=['--labels',str(labels)]
    elif case=='missing_labels':invocation[0]='calibrate'
    elif case=='reserved_overlap':
        reserved=tmp_path/'reserved.json';reserved.write_text(json.dumps(['cli_trial_0']),encoding='utf8')
        invocation[0]='calibrate';invocation+=['--labels',str(labels),'--evaluation-trials',str(reserved)]
    elif case=='checkpoint_hash':
        d=json.loads(ACCEPTANCE.read_text(encoding='utf8'));d['package_sha256']='0'*64
        bad=tmp_path/'acceptance.json';bad.write_text(json.dumps(d),encoding='utf8');invocation[invocation.index('--acceptance')+1]=str(bad)
    elif case=='existing_output':output.write_bytes(b'keep existing result')
    with pytest.raises(SystemExit) as error:main(invocation)
    assert error.value.code==2
    if case=='existing_output':assert output.read_bytes()==b'keep existing result'
    else:assert not output.exists()


def test_module_entrypoint_runs_in_separate_process_without_source_archive(tmp_path):
    import os
    import subprocess
    import sys
    windows,_,_,_=window_file(tmp_path)
    output=tmp_path/'process.json'
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')
    result=subprocess.run([sys.executable,'-m','emgimu.feature_bank.frozen_emg_bank_cli_v1',*args('predict',windows,output)],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    r=json.loads(output.read_text(encoding='utf8'))
    assert r['trial_ids']==['cli_trial_0'] and len(r['probabilities'])==1
    assert r['channels']==8 and r['sample_rate_hz']==200.
    np.testing.assert_allclose(np.sum(r['probabilities'],axis=1),1.,atol=1e-12)
