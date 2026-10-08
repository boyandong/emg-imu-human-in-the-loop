"""Offline portable EMG inference and separately labelled calibration."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from .core import FeatureBatch
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1, FrozenEmgUserStateV1


def load_bank(package, acceptance):
    evidence=json.loads(Path(acceptance).read_text(encoding='utf8'))
    packed=Path(package).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=evidence['package_sha256']:
        raise ValueError('Checkpoint SHA-256 differs from acceptance')
    bank=pickle.loads(packed)
    if not isinstance(bank,FrozenEmgProviderBankV1) or bank.bank_id_!=evidence['bank_id']:
        raise ValueError('Checkpoint type or bank identity differs')
    return bank


def load_windows(path):
    # Explicit window metadata, never infer trial boundaries or sampling rate.
    with np.load(path,allow_pickle=False) as data:
        required={'emg','sample_rate_hz','trial_ids','window_offsets'}
        if set(data.files)!=required:
            raise ValueError('Window file must contain exactly emg, sample_rate_hz, trial_ids, window_offsets; labels belong in separate calibration input')
        rate=data['sample_rate_hz']
        if rate.shape!=():raise ValueError('Explicit scalar sampling rate required')
        return FeatureBatch(data['emg'].copy(),float(rate)),data['trial_ids'].copy(),data['window_offsets'].copy()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('predict','calibrate'))
    parser.add_argument('--package',required=True)
    parser.add_argument('--acceptance',required=True)
    parser.add_argument('--windows',required=True,help='NPZ: emg, scalar sample_rate_hz, trial_ids, window_offsets')
    parser.add_argument('--user',required=True)
    parser.add_argument('--output',required=True,help='New output path; an existing file is not overwritten')
    parser.add_argument('--profile',help='Previously saved user calibration profile; predict only')
    parser.add_argument('--labels',help='Separate JSON trial-ID to class-index mapping; calibrate only')
    parser.add_argument('--evaluation-trials',help='JSON list of reserved evaluation IDs; calibrate only')
    parser.add_argument('--providers',nargs='+',help='Explicit available provider names; omitted means all')
    args=parser.parse_args(argv)
    try:
        if args.action=='predict' and (args.labels or args.evaluation_trials):
            raise ValueError('Prediction never accepts labels or calibration evaluation reservations')
        if args.action=='calibrate' and (not args.labels or args.profile):
            raise ValueError('Calibration requires separate labels and does not accept an existing profile')
        bank=load_bank(args.package,args.acceptance)
        batch,ids,offsets=load_windows(args.windows)
        kwargs={'window_offsets':offsets,'user_id':args.user,'available':args.providers}
        if args.action=='calibrate':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            forbidden=() if not args.evaluation_trials else json.loads(Path(args.evaluation_trials).read_text(encoding='utf8'))
            if not isinstance(forbidden,(list,tuple)) or any(not isinstance(v,str) or not v.strip() for v in forbidden):
                raise ValueError('Reserved evaluation IDs must be a string list')
            state=bank.calibrate_user(batch,ids,labels,forbidden_evaluation_trials=forbidden,**kwargs)
            output=pickle.dumps(state,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            state=None if not args.profile else pickle.loads(Path(args.profile).read_bytes())
            if state is not None and not isinstance(state,FrozenEmgUserStateV1):
                raise ValueError('Invalid user profile type')
            result=bank.predict(batch,ids,user_state=state,**kwargs)
            report={'schema':'frozen_emg_offline_prediction_v1','bank_id':bank.bank_id_,
                'user_id':args.user,'trial_ids':list(result['trial_ids']),
                'classes':list(result['classes']),'class_names':list(result['class_names']),
                'predicted_labels':list(result['predicted_labels']),
                'probabilities':result['probabilities'].tolist(),
                'provider_names':list(result['provider_names']),'weights':list(result['weights']),
                'sample_rate_hz':batch.sample_rate_hz,'channels':batch.channels,
                'scope':'Offline cued-trial inference; no physical accuracy or streaming detection claim.'}
            output=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8')
        with Path(args.output).open('xb') as handle:handle.write(output)
        print(f'{args.action}: saved {args.output}',flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':main()
