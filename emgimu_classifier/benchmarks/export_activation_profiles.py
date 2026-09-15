"""Reuse recorded source-force calibration IDs; do not open unseen-force recordings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from emgimu.datasets.libemg_force import load_libemg_force_windows
from emgimu.feature_bank.activation_profile import PersonalActivationProfile


def export(raw,processed,output):
    rows=[];profiles={};audits=[]
    for phase,users in (('validation',(7,8)),('final',(9,10))):
        source=processed/f'feature_bank_force_probability_{phase}_20260915/split_trial_ids.json'
        splits=json.loads(source.read_text());data=load_libemg_force_windows(raw,subjects=users,conditions=('Ramp',))
        for split in splits:
            user=split['user'];shots=split['shots'];ids=set(split['calibration'])
            if ids&set(split['evaluation']):raise AssertionError('Calibration/evaluation overlap')
            if not shots:
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'subject':user,'shots_per_class':0,
                    'condition':'Ramp','gesture':'ALL_ACTIVE','q10':'N/A','q50':'N/A','q90':'N/A','pattern_spread':'N/A',
                    'windows':0,'trials':0,'status':'no target calibration'})
                continue
            mask=(data.subjects==user)&np.isin(data.trials,list(ids));cal=data.take(np.flatnonzero(mask))
            if set(cal.trials)!=ids or set(cal.conditions)!= {'Ramp'}:raise AssertionError('Profile must use recorded Ramp calibration only')
            profile=PersonalActivationProfile().fit_calibration(cal.batch,cal.labels,cal.trials,rest_label=0).profile_
            profiles[f'{phase}_{user}_{shots}']=profile
            for label,values in profile['groups'].items():
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'subject':user,'shots_per_class':shots,
                    'condition':'Ramp','gesture':label,**{k:values[k] if values[k] is not None else 'N/A' for k in ('q10','q50','q90','pattern_spread','windows','trials')},
                    'status':'calibration-only signal range; no mechanical-force estimate'})
        for user in users:
            rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'subject':user,'shots_per_class':5,
                'condition':'Ramp','gesture':'ALL_ACTIVE','q10':'N/A','q50':'N/A','q90':'N/A','pattern_spread':'N/A',
                'windows':0,'trials':0,'status':'unsupported: only four Ramp trials/class'})
        audits.append({'phase':phase,'split_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'calibration_only_conditions':['Ramp'],'unseen_force_recordings_opened':False})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'personal_activation_profiles.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'personal_activation_profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')
    audit={'status':'ok','profiles':len(profiles),'rows':len(rows),'split_sources':audits,
        'original_info_sha256':hashlib.sha256((raw/'Info.txt').read_bytes()).hexdigest(),
        'native_rest_mapping':'C1/adapter label0 follows No Movement as first class in original Info.txt',
        'scope':'observed Ramp 20–80% MVC-feedback activation range; not unconstrained natural-product envelope or fatigue',
        'classifier_fit':False,'evaluation_samples_used':False}
    (output/'activation_profile_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','profiles','rows','evaluation_samples_used')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('processed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.raw,a.processed,a.output)
