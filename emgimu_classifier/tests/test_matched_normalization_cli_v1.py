"""Paired packages, separate labels and archive-free process inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from emgimu.feature_bank.matched_normalization_cli_v1 import main

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_RESULTS.json'
PREPROCESSING='song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1'


def inputs(folder,prefix):
    ids=np.array([prefix+'_'+c for c in ('fist','index_pinch','neutral','open_hand') for _ in range(2)])
    labels={t:t[len(prefix)+1:] for t in ids}
    path=folder/(prefix+'.npz');np.savez(path,emg=np.random.default_rng(len(prefix)).normal(size=(8,50,8)).astype(np.float32)*100,
        sample_rate_hz=np.array(250.),trial_ids=ids,window_offsets=np.tile([0,1],4))
    label_path=folder/(prefix+'.json');label_path.write_text(json.dumps(labels),encoding='utf8')
    return path,label_path


def arguments(action,mode,windows,output,session):
    package=ROOT/f'benchmarks/song_real8/song_matched_normalization_v1/{mode}_source_bank.pkl'
    return [action,'--mode',mode,'--package',str(package),'--results',str(RESULT),'--windows',str(windows),
        '--output',str(output),'--user','fixture','--session',session,'--preprocessing',PREPROCESSING,
        '--channels',*[f'CH{i+1}' for i in range(8)]]


@pytest.mark.parametrize('mode',['raw','normalized'])
def test_calibration_and_independent_process_profile_inference(mode,tmp_path):
    long,ly=inputs(tmp_path,'long');current,cy=inputs(tmp_path,'current');held,hy=inputs(tmp_path,'held')
    pp=tmp_path/'personal.zip';sp=tmp_path/'session.zip';output=tmp_path/'prediction.json'
    main(arguments('enroll',mode,long,pp,'long')+['--labels',str(ly)])
    main(arguments('session',mode,current,sp,'current')+['--personal-profile',str(pp),'--labels',str(cy)])
    argv=arguments('predict',mode,held,output,'current')+['--personal-profile',str(pp),'--session-profile',str(sp),'--providers','F0']
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    result=subprocess.run([sys.executable,'-m','emgimu.feature_bank.matched_normalization_cli_v1',*argv],
        cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    r=json.loads(output.read_text(encoding='utf8'))
    assert r['normalization_used_by_classifier']==(mode=='normalized') and r['weights']==[1.]
    assert len(r['trial_ids'])==4 and r['provider_names']==['F0']
    assert not r['quality_rejection_enabled'] and not r['physical_validation_proven']
    with pytest.raises(SystemExit):main(argv) # No overwrite.
    with pytest.raises(SystemExit):main(argv+['--labels',str(hy)])
    foreign=argv.copy();index=foreign.index('--mode')+1;foreign[index]='raw' if mode=='normalized' else 'normalized'
    with pytest.raises(SystemExit):main(foreign) # Source bytes/domain cannot be silently interchanged.


def test_normalized_model_requires_personal_statistics(tmp_path):
    held,_=inputs(tmp_path,'held');output=tmp_path/'prediction.json'
    with pytest.raises(SystemExit):main(arguments('predict','normalized',held,output,'current'))
    assert not output.exists()
    assert main(arguments('predict','raw',held,output,'current'))==0
    r=json.loads(output.read_text(encoding='utf8'));assert not r['normalization_used_by_classifier']
