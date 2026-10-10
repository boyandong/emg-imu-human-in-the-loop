"""Portable opt-in integrated enrollment/session/prediction without evaluation labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
from .frozen_emg_bank_cli_v1 import load_windows
from .personal_session_cli_v1 import load_workflow, _json
from .personal_session_stream_v2 import load_gate
from .personal_session_decision_v1 import PersonalSessionDecisionV1


def load_decision_workflow(package,base_acceptance,policy,acceptance,gate_package,gate_results):
    packed=Path(policy).read_bytes();evidence=json.loads(Path(acceptance).read_text(encoding='utf8'))
    if hashlib.sha256(packed).hexdigest()!=evidence['policy_sha256']:raise ValueError('Decision policy checksum differs')
    config=json.loads(packed);base=load_workflow(package,base_acceptance);gate=load_gate(gate_package,gate_results)
    if config['source_bank_id']!=base.bank.bank_id_ or config['base_contract_id']!=base.contract_id or config['gate_policy_id']!=gate.policy_id:
        raise ValueError('Decision source/gate/configuration identities differ')
    workflow=PersonalSessionDecisionV1(base,anchor_mix=config['anchor_mix'],gate=gate)
    if workflow.policy_id!=evidence['decision_policy_id']:raise ValueError('Decision policy configuration identity differs')
    return workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('enroll','session','predict'))
    for name in ('package','base-acceptance','policy','acceptance','gate-package','gate-results','windows','user','session','output','preprocessing'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True)
    parser.add_argument('--labels');parser.add_argument('--evaluation-trials')
    parser.add_argument('--personal-profile');parser.add_argument('--session-profile')
    parser.add_argument('--providers',nargs='+');parser.add_argument('--raw-windows')
    parser.add_argument('--anchor-mode',choices=('long_term','local','blended'),default='blended')
    parser.add_argument('--quality-mode',choices=('off','structural','soft'),default='off')
    parser.add_argument('--disable-anchor',action='store_true');parser.add_argument('--disable-session-routing',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.action=='predict' and (args.labels or args.evaluation_trials):raise ValueError('Prediction never accepts labels or calibration reservations')
        if args.action!='predict' and (not args.labels or args.providers or args.session_profile or args.raw_windows
                                      or args.disable_anchor or args.disable_session_routing or args.quality_mode!='off'):
            raise ValueError('Calibration needs separate labels; prediction controls cannot change enrollment')
        if ((args.action=='enroll' and args.personal_profile) or (args.action=='session' and not args.personal_profile)
                or (args.session_profile and not args.personal_profile)):raise ValueError('Invalid personal/session profile lifecycle')
        w=load_decision_workflow(args.package,args.base_acceptance,args.policy,args.acceptance,args.gate_package,args.gate_results)
        b,ids,offsets=load_windows(args.windows)
        common=dict(window_offsets=offsets,user_id=args.user,session_id=args.session,observed_channel_ids=args.channels,preprocessing_id=args.preprocessing)
        personal=None if not args.personal_profile else w.load_profile(args.personal_profile,user_id=args.user)
        if args.action!='predict':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            forbidden=[] if not args.evaluation_trials else json.loads(Path(args.evaluation_trials).read_text(encoding='utf8'))
            if args.action=='enroll':profile=w.enroll_user(b,ids,labels,forbidden_evaluation_trials=forbidden,**common)
            else:profile=w.calibrate_session(b,ids,labels,personal=personal,forbidden_evaluation_trials=forbidden,**common)
            w.save_profile(profile,args.output)
        else:
            session=None if not args.session_profile else w.load_profile(args.session_profile,user_id=args.user,session_id=args.session,personal=personal)
            raw=None
            if args.raw_windows:
                import numpy as np
                raw,raw_ids,raw_offsets=load_windows(args.raw_windows)
                if not np.array_equal(ids,raw_ids) or not np.array_equal(offsets,raw_offsets):raise ValueError('Raw/filtered trial/offset axes differ')
            result=w.predict(b,ids,personal=personal,session=session,available=args.providers,use_anchor=not args.disable_anchor,
                use_session_routing=not args.disable_session_routing,anchor_mode=args.anchor_mode,quality_mode=args.quality_mode,raw_batch=raw,**common)
            with Path(args.output).open('xb') as f:
                f.write((json.dumps(result,ensure_ascii=False,indent=2,default=_json)+'\n').encode('utf8'))
        print(args.action+': saved '+args.output,flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError,EOFError,zipfile.BadZipFile) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
