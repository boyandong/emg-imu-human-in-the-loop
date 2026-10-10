"""Offline enrollment, new-session calibration and unlabeled bank prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .frozen_emg_bank_cli_v1 import load_windows
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1


def load_workflow(package,acceptance):
    evidence=json.loads(Path(acceptance).read_text(encoding='utf8'));packed=Path(package).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=evidence['source_bank_sha256']:raise ValueError('Source bank checksum differs')
    bank=pickle.loads(packed)
    if not isinstance(bank,FrozenEmgProviderBankV1) or bank.bank_id_!=evidence['source_bank_id']:
        raise ValueError('Source bank type/identity differs')
    workflow=PersonalSessionWorkflowV1(bank,**evidence['workflow_config'])
    if workflow.contract_id!=evidence['workflow_contract_id']:raise ValueError('Workflow configuration identity differs')
    return workflow


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported output type: '+type(value).__name__)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('enroll','session','predict'))
    for name in ('package','acceptance','windows','user','session','output','preprocessing'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True,help='Observed channel column identities, in file order')
    parser.add_argument('--labels',help='Calibration only: separate JSON trial-ID to class-label mapping')
    parser.add_argument('--personal-profile')
    parser.add_argument('--session-profile')
    parser.add_argument('--evaluation-trials',help='Calibration only: optional JSON reserved evaluation trial IDs')
    parser.add_argument('--providers',nargs='+',help='Prediction only: available source providers')
    args=parser.parse_args(argv)
    try:
        if args.action=='predict' and (args.labels or args.evaluation_trials):
            raise ValueError('Prediction never accepts evaluation labels or calibration reservations')
        if args.action!='predict' and (not args.labels or args.session_profile or args.providers):
            raise ValueError('Calibration requires separate labels and cannot use session-profile/providers')
        if ((args.action=='enroll' and args.personal_profile) or (args.action=='session' and not args.personal_profile)
                or (args.session_profile and not args.personal_profile)):
            raise ValueError('Enroll creates a personal profile; session calibration and session-profile prediction require it')
        workflow=load_workflow(args.package,args.acceptance)
        batch,ids,offsets=load_windows(args.windows)
        kwargs=dict(window_offsets=offsets,user_id=args.user,session_id=args.session,
            observed_channel_ids=args.channels,preprocessing_id=args.preprocessing)
        personal=None if not args.personal_profile else workflow.load_profile(args.personal_profile,user_id=args.user)
        if args.action!='predict':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            forbidden=() if not args.evaluation_trials else json.loads(Path(args.evaluation_trials).read_text(encoding='utf8'))
            if not isinstance(forbidden,(tuple,list)) or any(not isinstance(t,str) or not t.strip() for t in forbidden):
                raise ValueError('Reserved evaluation trials must be an explicit string list')
            if args.action=='enroll':profile=workflow.enroll_user(batch,ids,labels,forbidden_evaluation_trials=forbidden,**kwargs)
            else:profile=workflow.calibrate_session(batch,ids,labels,personal=personal,forbidden_evaluation_trials=forbidden,**kwargs)
            workflow.save_profile(profile,args.output)
        else:
            session=None if not args.session_profile else workflow.load_profile(args.session_profile,user_id=args.user,
                session_id=args.session,personal=personal)
            if personal is None:
                mapped=workflow._input(batch,observed_channel_ids=args.channels,preprocessing_id=args.preprocessing)
                result=workflow.bank.predict(mapped,ids,window_offsets=offsets,user_id=args.user,available=args.providers)
                result.update(session_id=args.session,personal_profile_id=None,session_profile_id=None,
                    anchor_coordinates={},session_descriptor=None,quality_observations=None,quality_feature_names=(),
                    scope='Source-population zero-target-calibration inference; no personal/session/quality reference is available.')
            else:result=workflow.predict(batch,ids,personal=personal,session=session,available=args.providers,**kwargs)
            result.update(schema='personal_session_offline_prediction_v1',window_trial_ids=ids,
                predicted_labels=[result['class_names'][i] for i in result['probabilities'].argmax(1)],
                normalization_used_by_classifier=False,quality_rejection_enabled=False)
            encoded=(json.dumps(result,ensure_ascii=False,indent=2,default=_json)+'\n').encode('utf8')
            with Path(args.output).open('xb') as f:f.write(encoded)
        print(f'{args.action}: saved {args.output}',flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError,EOFError,zipfile.BadZipFile) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':main()
