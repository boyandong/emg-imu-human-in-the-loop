"""Recover source-only fitted states, then prove portable unlabeled native replay."""
import json
import argparse
import pickle
import platform
from pathlib import Path
import numpy as np
import sklearn
from emgimu.datasets.epn612 import load_epn612_windows, GESTURES
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from benchmarks.new_bank_v3.emg_calibrated_fusion_v1 import fit_source
from benchmarks.new_bank_v3.emg_window_bank_v1 import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACKAGE=ROOT/'feature_bank/models/epn_emg_calibrated_bank_v1.pkl'
OUT=ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json'


def run(verify_only=False):
    if PACKAGE.exists() and not verify_only:raise FileExistsError('Use --verify-only for an existing source package')
    protocol=HERE/'EMG_CALIBRATED_FUSION_V1_PROTOCOL.json';result=HERE/'EMG_CALIBRATED_FUSION_V1_RESULTS.json'
    p=json.loads(protocol.read_text(encoding='utf8'));r=json.loads(result.read_text(encoding='utf8'))
    if sha(protocol)!=r['protocol_sha256']:raise ValueError('Protocol changed')
    for path,digest in p['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen training implementation changed')
    if sha(Path(p['archive']))!=p['archive_sha256']:raise ValueError('Native archive changed')
    if verify_only:
        previous=json.loads(OUT.read_text(encoding='utf8'))
        if sha(PACKAGE)!=previous['package_sha256']:raise ValueError('Existing source checkpoint changed')
        packed=PACKAGE.read_bytes();bank=pickle.loads(packed);ids=bank.policy_.source_trials
        for name,(scaler,model) in bank.models_.items():
            for field,value in [('mean',scaler.mean_),('scale',scaler.scale_),('coef',model.coef_),('intercept',model.intercept_),('classes',model.classes_),('iterations',model.n_iter_)]:
                if not np.array_equal(value,r['source_models'][name][field]):raise ValueError('Existing source parameters differ')
        if list(ids)!=r['source_trial_ids']:raise ValueError('Existing source trial IDs differ')
        print('1/3 Verify existing checkpoint; no source or target refitting',flush=True)
    else:
        print('1/3 Recover only six final source fits; no source-OOF or target retraining',flush=True)
        source=load_epn612_windows(p['archive'],users=p['source_users'])
        fitted,models,parameters,ids=fit_source(source,p['source_model_seed'])
        if parameters!=r['source_models'] or ids.tolist()!=r['source_trial_ids']:
            raise ValueError('Recovered source parameters differ from frozen results')
        bank=FrozenEmgProviderBankV1(fitted,models,r['temperatures'],classes=tuple(range(6)),class_names=GESTURES,
            source_trial_ids=ids,source_policy_id=r['protocol_sha256'],sample_rate_hz=200.,window_samples=40,channels=8,
            population=(1/6,)*6,n0=12.,reliability_temperature=1.)
        packed=pickle.dumps(bank,protocol=pickle.HIGHEST_PROTOCOL)
    bank=pickle.loads(packed);before=pickle.dumps(bank);rows=[]
    print('2/3 Pure-EMG unlabeled inference and separate user calibration',flush=True)
    for block in r['blocks']:
        user=str(block['user']);data=load_epn612_windows(p['archive'],users=[block['user']])
        mask=np.isin(data.trials,block['evaluation_ids'])
        batch=FeatureBatch(data.batch.emg[mask],200.) # intentionally no IMU, posture or labels
        trials=data.trials[mask]
        ordinal=np.empty(len(data.trials),dtype=int);counts={}
        for i,trial in enumerate(data.trials):
            ordinal[i]=counts.get(trial,0);counts[trial]=ordinal[i]+1
        providers=bank.predict_providers(batch,trials,window_offsets=ordinal[mask],user_id=user)
        if list(providers['trial_ids'])!=block['evaluation_ids']:raise ValueError('Native evaluation ordering changed')
        provider_error=max(float(np.max(abs(providers['probabilities'][name]-np.array(q)))) for name,q in block['provider_probabilities'].items())
        if provider_error>1e-12:raise ValueError('Compiled provider predictions differ')
        for entry in block['calibrations']:
            if entry['shots']:
                selected=np.isin(data.trials,entry['ids'])
                state=bank.calibrate_user(FeatureBatch(data.batch.emg[selected],200.),data.trials[selected],
                    dict(zip(entry['ids'],entry['labels'])),window_offsets=ordinal[selected],user_id=user,forbidden_evaluation_trials=block['evaluation_ids'])
                weight_error=float(np.max(abs(np.array(state.fusion_state.weights)-np.array(entry['weights']))))
            else:state=None;weight_error=0.
            full=bank.predict(batch,trials,window_offsets=ordinal[mask],user_id=user,user_state=state)
            error=float(np.max(abs(full['probabilities']-np.array(entry['probabilities']['reliability_bank']))))
            if error>1e-12 or weight_error>1e-12:raise ValueError('Compiled calibration/fusion differs')
            omissions=[]
            for missing in bank.providers_:
                active=[name for name in bank.providers_ if name!=missing]
                output=bank.predict(batch,trials,window_offsets=ordinal[mask],user_id=user,user_state=state,available=active)
                delta=float(np.max(abs(output['probabilities']-np.array(entry['probabilities']['reliability_minus_'+missing]))))
                if delta>1e-12:raise ValueError('Compiled omission differs')
                omissions.append({'missing_provider':missing,'max_probability_error':delta})
            rows.append({'user':block['user'],'shots_per_class':entry['shots'],'evaluation_trials':len(full['trial_ids']),
                'provider_max_probability_error':provider_error,'calibration_weight_max_error':weight_error,
                'fusion_max_probability_error':error,'omission_cases':omissions})
        print(f'User{user}: portable package replay complete',flush=True)
    if before!=pickle.dumps(bank):raise ValueError('Native calibration or inference changed source package')
    if not verify_only:
        PACKAGE.parent.mkdir(parents=True,exist_ok=True);PACKAGE.write_bytes(packed)
    paths=[protocol,result,Path(__file__),ROOT/'src/emgimu/feature_bank/frozen_emg_provider_bank_v1.py',
           ROOT/'src/emgimu/feature_bank/available_bank_fusion_v1.py',ROOT/'tests/test_frozen_emg_provider_bank_v1.py']
    out={'schema':'frozen_emg_bank_acceptance_v1','source_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in paths},
        'package_path':PACKAGE.relative_to(ROOT).as_posix(),'package_sha256':sha(PACKAGE),'bank_id':bank.bank_id_,
        'python_version':platform.python_version(),'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
        'source_parameters_exactly_recovered':True,'source_trials':len(ids),'classes':list(GESTURES),
        'sample_rate_hz':200,'window_samples':40,'channels':8,'requires_IMU':False,
        'records':rows,'native_provider_cases':10,'native_fusion_cases':40,'native_omission_cases':240,
        'source_state_immutable':True,'default_promoted':False,'physical_validation_proven':False,'completion_proven':False,
        'scope':'Portable six-source-provider checkpoint for the completed EPN62-71 experiment. Only final source fits were recovered and exactly matched recorded parameters; no new model selection, OOF refit or target training. Unlabelled inference accepts pure 8-channel200Hz40-sample windows aggregated by unique native trial IDs and explicit contiguous chronological window offsets. This is cue-trial mean inference, not a streaming detector, current250Hz device adapter or complete F0-F9 bank. Native primary guard remains failed.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print('3/3 Saved source-only portable checkpoint,40 fused cases and240 omissions; no default promotion',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    run(args.verify_only)
