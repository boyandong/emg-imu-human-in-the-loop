"""Load paired source-frozen raw/normalized candidates without native recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
from .frozen_emg_bank_cli_v1 import load_windows
from .personal_session_cli_v1 import _json
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .matched_normalized_bank_v1 import MatchedNormalizedWorkflowV1,MatchedNormalizedProviderBankV1


def load_matched_workflow(package,results,mode):
    r=json.loads(Path(results).read_text(encoding='utf8'));metadata=r['source'][mode]
    packed=Path(package).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=metadata['sha256']:raise ValueError('Matched source package checksum differs')
    bank=pickle.loads(packed)
    if bank.bank_id_!=metadata['bank_id']:raise ValueError('Matched source identity differs')
    if mode=='normalized':
        if not isinstance(bank,MatchedNormalizedProviderBankV1):raise ValueError('Normalized source input-domain type differs')
        workflow=MatchedNormalizedWorkflowV1(bank,**r['workflow_config'])
    elif mode=='raw':
        if type(bank) is not FrozenEmgProviderBankV1:raise ValueError('Raw source input-domain type differs')
        workflow=PersonalSessionWorkflowV1(bank,**r['workflow_config'])
    else:raise ValueError('Explicit raw or normalized domain required')
    if workflow.contract_id!=metadata['contract_id']:raise ValueError('Matched workflow contract differs')
    return workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('enroll','session','predict'))
    parser.add_argument('--mode',choices=('raw','normalized'),required=True)
    for name in ('package','results','windows','output','user','session','preprocessing'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True)
    parser.add_argument('--labels')
    parser.add_argument('--personal-profile')
    parser.add_argument('--session-profile')
    parser.add_argument('--providers',nargs='+')
    args=parser.parse_args(argv)
    try:
        if args.action=='predict' and args.labels:raise ValueError('Prediction cannot accept evaluation labels')
        if args.action!='predict' and (not args.labels or args.session_profile or args.providers):
            raise ValueError('Calibration requires separate labels; session-profile/providers are prediction only')
        if args.action=='enroll' and args.personal_profile:raise ValueError('Enroll creates a new personal profile')
        if (args.action=='session' or args.session_profile or (args.action=='predict' and args.mode=='normalized')) and not args.personal_profile:
            raise ValueError('This operation requires a matching personal profile; normalized models have no uncalibrated fallback')
        w=load_matched_workflow(args.package,args.results,args.mode)
        batch,ids,offsets=load_windows(args.windows)
        kwargs=dict(window_offsets=offsets,user_id=args.user,session_id=args.session,
                    observed_channel_ids=args.channels,preprocessing_id=args.preprocessing)
        personal=None if not args.personal_profile else w.load_profile(args.personal_profile,user_id=args.user)
        if args.action!='predict':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            if args.action=='enroll':profile=w.enroll_user(batch,ids,labels,**kwargs)
            else:profile=w.calibrate_session(batch,ids,labels,personal=personal,**kwargs)
            w.save_profile(profile,args.output)
        else:
            session=None if not args.session_profile else w.load_profile(args.session_profile,user_id=args.user,
                session_id=args.session,personal=personal)
            if personal is None:
                mapped=w._input(batch,observed_channel_ids=args.channels,preprocessing_id=args.preprocessing)
                result=w.bank.predict(mapped,ids,window_offsets=offsets,user_id=args.user,available=args.providers)
            else:result=w.predict(batch,ids,personal=personal,session=session,available=args.providers,**kwargs)
            result.update(schema='matched_normalization_prediction_v1',input_mode=args.mode,
                normalization_used_by_classifier=args.mode=='normalized',quality_rejection_enabled=False,
                physical_validation_proven=False)
            with Path(args.output).open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,default=_json)
        print(f'{args.action}/{args.mode}: saved {args.output}',flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError,EOFError,zipfile.BadZipFile) as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
