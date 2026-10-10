"""Shared complete-action enrollment, session calibration and label-free inference.

Input NPZ uses the existing full-bout schema. Raw pre-software-highpass samples
are a separate NPZ with identical native axes. Calibration labels are separate
JSON; prediction rejects labels and reservations. Outputs never overwrite.
"""
import argparse
import json
from pathlib import Path
import pickle
import zipfile

from .extended_window_cli_v1 import load_extended_workflow
from .joint_bout_workflow_v1 import JointBoutWorkflowV1
from .personal_session_cli_v1 import _json
from .personal_temporal_cli_v1 import load_bouts


def load_joint_workflow(source_root):
    root=Path(source_root)
    return JointBoutWorkflowV1(load_extended_workflow(root/'song_extended_window_v1/source_bank.pkl',
        root/'song_extended_window_v1/policy.json',root/'SONG_EXTENDED_WINDOW_V1_RESULTS.json',
        root/'song_raw_quality_v1/source_gate.pkl',root/'SONG_RAW_QUALITY_V1_RESULTS.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('enroll','session','predict'))
    for name in ('source-root','bouts','user','session','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--raw-bouts');parser.add_argument('--labels')
    parser.add_argument('--personal-profile');parser.add_argument('--session-profile')
    parser.add_argument('--reserved-evaluation',help='JSON with trial_ids and recording_ids')
    parser.add_argument('--quality-mode',choices=('off','structural','soft'),default='off')
    parser.add_argument('--anchor-mode',choices=('long_term','local','blended'),default='blended')
    parser.add_argument('--disable-anchor',action='store_true')
    parser.add_argument('--disable-session-routing',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.action=='predict' and (args.labels or args.reserved_evaluation):
            raise ValueError('Prediction never accepts labels or calibration reservations')
        if args.action!='predict' and (not args.labels or args.session_profile or args.disable_anchor
                or args.disable_session_routing or args.anchor_mode!='blended'):
            raise ValueError('Calibration needs separate labels and no prediction controls')
        if ((args.action=='enroll' and args.personal_profile)
                or (args.action in ('session','predict') and not args.personal_profile)):
            raise ValueError('Invalid joint personal/session lifecycle')
        w=load_joint_workflow(args.source_root)
        batch=load_bouts(args.bouts)
        raw=None if not args.raw_bouts else load_bouts(args.raw_bouts)
        personal=None if not args.personal_profile else w.load_profile(args.personal_profile,user_id=args.user)
        if args.action!='predict':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            forbidden={}
            if args.reserved_evaluation:
                values=json.loads(Path(args.reserved_evaluation).read_text(encoding='utf8'))
                if (not isinstance(values,dict) or set(values)!={'trial_ids','recording_ids'}
                        or any(not isinstance(v,list) or any(not isinstance(t,str) or not t.strip() for t in v)
                               for v in values.values())):
                    raise ValueError('Explicit reserved evaluation trial and recording lists required')
                forbidden=dict(forbidden_trial_ids=values['trial_ids'],forbidden_recording_ids=values['recording_ids'])
            profile=w.enroll(batch,labels,user_id=args.user,session_id=args.session,personal=personal,
                raw_batch=raw,quality_mode=args.quality_mode,**forbidden)
            w.save_profile(profile,args.output)
        else:
            session=None if not args.session_profile else w.load_profile(args.session_profile,
                user_id=args.user,session_id=args.session,personal=personal)
            result=w.predict(batch,personal=personal,session=session,user_id=args.user,session_id=args.session,
                raw_batch=raw,quality_mode=args.quality_mode,use_anchor=not args.disable_anchor,
                use_session_routing=not args.disable_session_routing,anchor_mode=args.anchor_mode)
            encoded=(json.dumps(result,ensure_ascii=False,indent=2,default=_json)+'\n').encode('utf8')
            with Path(args.output).open('xb') as stream:stream.write(encoded)
        print(args.action+': saved '+args.output,flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError,EOFError,zipfile.BadZipFile) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
