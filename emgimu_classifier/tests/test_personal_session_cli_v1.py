"""Separate-process inference uses persisted files, with no recording archive."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from emgimu.feature_bank.personal_session_cli_v1 import main,load_workflow

ROOT=Path(__file__).resolve().parents[1]
ACCEPTANCE=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'
PACKAGE=ROOT/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl'


def inputs(folder,prefix):
    classes=['fist','index_pinch','neutral','open_hand']
    ids=np.array([f'{prefix}_{c}_{i}' for c in classes for i in range(2) for _ in range(2)])
    labels={t:c for c in classes for t in ids if f'_{c}_' in t}
    values=np.random.default_rng(9).normal(size=(16,50,8)).astype(np.float32)*100
    windows=folder/(prefix+'.npz');np.savez(windows,emg=values,sample_rate_hz=np.array(250.),trial_ids=ids,window_offsets=np.tile([0,1],8))
    label_file=folder/(prefix+'.json');label_file.write_text(json.dumps(labels),encoding='utf8')
    return windows,label_file


def arguments(action,windows,output,session):
    return [action,'--package',str(PACKAGE),'--acceptance',str(ACCEPTANCE),'--windows',str(windows),
        '--user','fixture','--session',session,'--output',str(output),'--preprocessing',
        'song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1','--channels',*[f'CH{i+1}' for i in range(8)]]


def test_enroll_new_session_and_separate_process_unlabeled_prediction(tmp_path):
    long,ly=inputs(tmp_path,'long');current,cy=inputs(tmp_path,'current');held,_=inputs(tmp_path,'held')
    personal=tmp_path/'personal.zip';session=tmp_path/'session.zip';output=tmp_path/'prediction.json'
    assert main(arguments('enroll',long,personal,'recording1')+['--labels',str(ly)])==0
    assert main(arguments('session',current,session,'recording2')+['--labels',str(cy),'--personal-profile',str(personal)])==0
    argv=arguments('predict',held,output,'recording2')+['--personal-profile',str(personal),'--session-profile',str(session),'--providers','F0']
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')
    result=subprocess.run([sys.executable,'-m','emgimu.feature_bank.personal_session_cli_v1',*argv],env=env,cwd=tmp_path,
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text(encoding='utf8'))
    assert report['provider_names']==['F0'] and report['weights']==[1.]
    assert report['session_id']=='recording2' and len(report['trial_ids'])==8 and len(report['window_trial_ids'])==16
    assert report['session_descriptor']['current_sessions']==['recording2']
    assert not report['normalization_used_by_classifier'] and not report['quality_rejection_enabled']
    np.testing.assert_allclose(np.array(report['probabilities']).sum(1),1.,atol=1e-12)
    with pytest.raises(SystemExit) as error:main(argv)
    assert error.value.code==2 # Existing output is preserved.
    for extra in [['--labels',str(cy)],['--evaluation-trials',str(cy)]]:
        with pytest.raises(SystemExit) as error:main(argv+extra)
        assert error.value.code==2
    wrong=argv.copy();wrong[wrong.index('--session')+1]='wrong_recording'
    with pytest.raises(SystemExit) as error:main(wrong)
    assert error.value.code==2


def test_no_profile_uses_source_population_without_inventing_context(tmp_path):
    windows,_=inputs(tmp_path,'held');output=tmp_path/'population.json'
    assert main(arguments('predict',windows,output,'new_recording'))==0
    report=json.loads(output.read_text(encoding='utf8'));workflow=load_workflow(PACKAGE,ACCEPTANCE)
    np.testing.assert_allclose(report['weights'],workflow.bank.policy_.population,rtol=0,atol=1e-12)
    assert report['personal_profile_id'] is None and report['session_profile_id'] is None
    assert report['quality_observations'] is None and report['anchor_coordinates']=={}
