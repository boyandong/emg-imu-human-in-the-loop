"""Class-matched session shifts from existing MANUS source and calibration trial IDs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.feature_bank.manus_study import GESTURES
from emgimu.feature_bank.session_shift_summary import FamilySessionShiftSummary


def export(archive,processed,output):
    source=load_semg_manus_windows(archive,users=range(3,9),sessions=(1,),gestures=GESTURES)
    profiles={};states={};rows=[];audits=[]
    for user in range(3,9):
        mask=np.flatnonzero(source.users==user)
        states[user]=FamilySessionShiftSummary().fit_long_term(source.batch.take(mask),source.labels[mask],source.trials[mask],ring_topology=True)
    before=pickle.dumps(states)
    for phase,session in (('validation',2),('final',3)):
        path=processed/f'feature_bank_manus_selected_{phase}_20260915/split_trial_ids.json'
        splits=json.loads(path.read_text());target=load_semg_manus_windows(archive,users=range(3,9),sessions=(session,),gestures=GESTURES)
        for split in splits:
            if not split['shots']:continue
            cal_ids=set(split['calibration']);user=split['user'];shots=split['shots']
            if cal_ids&set(split['evaluation']) or cal_ids&set(source.trials):raise AssertionError('Calibration/source/evaluation overlap')
            mask=np.flatnonzero((target.users==user)&np.isin(target.trials,list(cal_ids)))
            if set(target.trials[mask])!=cal_ids:raise AssertionError('Calibration trial coverage mismatch')
            result=states[user].from_calibration(target.batch.take(mask),target.labels[mask],target.trials[mask])
            profiles[f'{phase}_{user}_{shots}']=result
            for label,values in result['classes'].items():
                row={'dataset':'semg_manus','phase':phase,'subject':user,'source_session':1,'target_session':session,
                    'shots_per_class':shots,'gesture':label,**values,'rest_noise_shift_available':False,
                    'calibration_trials':len(cal_ids),'evaluation_samples_used':False}
                row['channel_variance_residual_json']=json.dumps(row['channel_variance_residual_json'])
                rows.append(row)
        audits.append({'phase':phase,'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if pickle.dumps(states)!=before:raise AssertionError('Session summary mutated long-term source state')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'family_specific_session_shifts.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'family_specific_session_shifts.json').write_text(json.dumps(profiles,indent=2))
    audit={'status':'ok','profiles':len(profiles),'class_matched_rows':len(rows),'sources':audits,
        'long_term_state_sha256':hashlib.sha256(before).hexdigest(),'evaluation_samples_used':False,
        'classifier_fit':False,'rest_noise':'N/A: subset has six active finger classes',
        'scope':'family-specific diagnostic session signatures; not new fitted fusion predictions'}
    (output/'family_specific_session_shift_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:audit[k] for k in ('status','profiles','class_matched_rows','evaluation_samples_used')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('processed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.archive,a.processed,a.output)
