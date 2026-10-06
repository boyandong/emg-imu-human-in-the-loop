"""Retrospective F8 V3 descriptors on frozen MANUS calibration trials only."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.manus_study import GESTURES
from emgimu.feature_bank.document_session_v3 import DocumentSessionDescriptorV3


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(archive,processed,output):
    source=load_semg_manus_windows(archive,users=range(3,9),sessions=(1,),gestures=GESTURES)
    states={};source_ids={}
    for user in range(3,9):
        mask=np.flatnonzero(source.users==user)
        source_ids[user]=sorted(set(source.trials[mask]))
        states[user]=DocumentSessionDescriptorV3().fit_long_term(source.batch.take(mask),
            source.labels[mask],source.trials[mask],user_id=str(user),
            session_ids=['1']*len(mask),ring_topology=False)
    frozen=pickle.dumps(states);records=[];splits=[]
    for phase,session in (('validation',2),('descriptive_final',3)):
        old_phase='validation' if session==2 else 'final'
        split_path=processed/f'feature_bank_manus_selected_{old_phase}_20260915/split_trial_ids.json'
        splits.append({'phase':phase,'split_sha256':digest(split_path)})
        target=load_semg_manus_windows(archive,users=range(3,9),sessions=(session,),gestures=GESTURES)
        for split in json.loads(split_path.read_text()):
            if not split['shots']:continue
            user=split['user'];cal=set(split['calibration']);evaluation=set(split['evaluation'])
            if cal&evaluation or cal&set(source_ids[user]):raise ValueError('Frozen split overlap')
            mask=np.flatnonzero((target.users==user)&np.isin(target.trials,list(cal)))
            if set(target.trials[mask])!=cal:raise ValueError('Calibration coverage mismatch')
            descriptor=states[user].from_calibration(target.batch.take(mask),target.labels[mask],
                target.trials[mask],user_id=str(user),session_ids=[str(session)]*len(mask))
            records.append({'phase':phase,'user':user,'shots':split['shots'],
                'source_trials':source_ids[user],'calibration_trials':sorted(cal),
                'evaluation_trials':sorted(evaluation),'calibration_windows':len(mask),
                'descriptor':descriptor})
    if pickle.dumps(states)!=frozen:raise AssertionError('Long-term profiles mutated')
    if len(records)!=24 or any(r['descriptor']['phi_dimension']!=180 for r in records):
        raise AssertionError('Unexpected native descriptor inventory')
    root=Path(__file__).resolve().parents[2]
    evidence=['src/emgimu/feature_bank/document_session_v3.py',
              'src/emgimu/feature_bank/document_scale_v3.py',
              'src/emgimu/feature_bank/session_shift_summary.py',
              'src/emgimu/feature_bank/spec_spatial_v3.py',
              'src/emgimu/feature_bank/families.py',
              'src/emgimu/feature_bank/relative_spectrum.py',
              'benchmarks/new_bank_v3/f8_document_manus.py']
    result={'version':'document-f8-v3','dataset':'sEMG-MANUS',
        'archive_sha256':digest(archive),'source_hashes':{p:digest(root/p) for p in evidence},
        'frozen_splits':splits,'users':list(range(3,9)),'source_session':1,
        'target_sessions':[2,3],'profiles':len(records),'dimension':180,
        'dimension_formula':'K*(2H+H*(H-1)/2)+H*(4+C); K=4,H=6,C=8',
        'ring_enabled':False,'evaluation_windows_used_for_descriptor':False,
        'classifier_fit':False,'long_term_state_sha256':hashlib.sha256(frozen).hexdigest(),
        'scope':'Retrospective native same-user session descriptors; no routing, fatigue or live-accuracy claim.',
        'records':records}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('profiles','dimension','ring_enabled','classifier_fit')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('processed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.archive,a.processed,a.output)
