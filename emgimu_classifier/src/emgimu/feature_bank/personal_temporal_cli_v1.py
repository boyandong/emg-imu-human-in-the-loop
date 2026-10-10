"""Separate-process full-bout enrollment and label-free temporal prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .personal_session_cli_v1 import _json
from .personal_temporal_bouts_v1 import TemporalBoutBatchV1, PersonalTemporalBoutsV1


def load_bouts(path):
    with np.load(path,allow_pickle=False) as z:
        required={'samples','offsets','trial_ids','recording_ids','starts','sample_rate_hz',
            'channel_ids','preprocessing_id','boundary_kind'}
        if set(z.files) != required:
            raise ValueError('Exact native full-bout fields required; labels are separate inputs')
        x=z['samples'];offsets=z['offsets']
        if (x.ndim != 2 or offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets)<2
                or offsets[0] != 0 or offsets[-1] != len(x) or np.any(np.diff(offsets)<=0)):
            raise ValueError('Contiguous native samples and complete sequence offsets required')
        batch=TemporalBoutBatchV1(tuple(x[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])),
            tuple(z['trial_ids'].tolist()),tuple(z['recording_ids'].tolist()),tuple(z['starts'].tolist()),
            float(z['sample_rate_hz']),tuple(z['channel_ids'].tolist()),str(z['preprocessing_id']),str(z['boundary_kind']))
    return batch.validate()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('enroll','session','predict'))
    for name in ('config','config-sha256','bouts','user','session','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--personal-profile');parser.add_argument('--session-profile')
    parser.add_argument('--labels');parser.add_argument('--base-probabilities')
    args=parser.parse_args(argv)
    try:
        packed=Path(args.config).read_bytes()
        if hashlib.sha256(packed).hexdigest()!=args.config_sha256:
            raise ValueError('Temporal configuration checksum differs')
        w=PersonalTemporalBoutsV1(**json.loads(packed));batch=load_bouts(args.bouts)
        if args.action=='predict' and args.labels:
            raise ValueError('Prediction never accepts labels')
        if args.action!='predict' and (not args.labels or args.base_probabilities or args.session_profile):
            raise ValueError('Separate calibration labels required; prediction controls cannot alter enrollment')
        if ((args.action=='enroll' and args.personal_profile)
                or (args.action in ('session','predict') and not args.personal_profile)
                or (args.session_profile and not args.personal_profile)):
            raise ValueError('Invalid temporal profile lifecycle')
        personal=None if not args.personal_profile else w.load_profile(args.personal_profile,user_id=args.user)
        if args.action!='predict':
            labels=json.loads(Path(args.labels).read_text(encoding='utf8'))
            p=w.enroll(batch,labels,user_id=args.user,session_id=args.session,personal=personal)
            w.save_profile(p,args.output)
        else:
            session=None if not args.session_profile else w.load_profile(args.session_profile,user_id=args.user)
            base={}
            if args.base_probabilities:
                with np.load(args.base_probabilities,allow_pickle=False) as z:
                    if set(z.files)!={'probabilities','trial_ids','class_names'}:
                        raise ValueError('Base probability inputs need only probabilities/trial/class axes')
                    base=dict(base_probabilities=z['probabilities'],base_trial_ids=z['trial_ids'],base_class_names=z['class_names'])
            result=w.predict(batch,personal=personal,session=session,user_id=args.user,session_id=args.session,**base)
            with Path(args.output).open('xb') as f:
                f.write((json.dumps(result,ensure_ascii=False,indent=2,default=_json)+'\n').encode('utf8'))
        print(args.action+': saved '+args.output,flush=True)
        return 0
    except (ValueError,TypeError,KeyError,OSError,pickle.UnpicklingError,EOFError,zipfile.BadZipFile) as exc:
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
