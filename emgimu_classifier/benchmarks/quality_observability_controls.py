"""Frozen quality observation controls on recorded Ramp calibration windows only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.libemg_force import load_libemg_force_windows
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


def export(raw,processed,output):
    source=load_libemg_force_windows(raw,subjects=range(1,7),conditions=('Ramp',))
    # Info.txt documents lowpass filtering but does not establish upstream highpass availability.
    family=QualityObservabilityFamily(pre_highpass_available=False,ring_topology=True).fit(source.batch)
    before=pickle.dumps(family);source_rms=np.sqrt(np.mean(source.batch.emg.astype(float)**2,axis=1));amplitude=np.median(source_rms,axis=0)
    rows=[];split_hashes=[]
    for phase,users in (('validation',(7,8)),('final',(9,10))):
        path=processed/f'feature_bank_force_probability_{phase}_20260915/split_trial_ids.json';splits=json.loads(path.read_text())
        data=load_libemg_force_windows(raw,subjects=users,conditions=('Ramp',))
        for user in users:
            split=next(s for s in splits if s['user']==user and s['shots']==1)
            if set(split['calibration'])&set(split['evaluation']):raise AssertionError('Calibration leakage')
            mask=(data.subjects==user)&np.isin(data.trials,split['calibration']);x=data.batch.emg[mask].copy()
            if set(data.trials[mask])!=set(split['calibration']):raise AssertionError('Calibration coverage mismatch')
            contiguous=x.copy();contiguous[:,:80,2]=contiguous[:,0:1,2]
            fragmented=x.copy()
            for step in range(1,159,2):fragmented[:,step,2]=fragmented[:,step-1,2]
            time=np.arange(x.shape[1])/data.batch.sample_rate_hz
            low=x+5*amplitude[None,None,:]*np.sin(2*np.pi*5*time)[None,:,None]
            for scenario,values in (('uncorrupted',x),('contiguous_flatline_ch3',contiguous),('fragmented_flatline_ch3',fragmented),('synthetic_5Hz',low)):
                features=family.transform(FeatureBatch(values,data.batch.sample_rate_hz));names=family.feature_names
                columns=[i for i,n in enumerate(names) if '.low_frequency_power_ratio.' in n]
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'subject':user,'condition':'Ramp_cal1','scenario':scenario,
                    'windows':len(x),'feature_dimension':len(names),
                    'legacy_flatline_fraction_ch3':float(features[:,names.index('F9.flatline_fraction.ch3')].mean()),
                    'longest_flatline_ratio_ch3':float(features[:,names.index('F9v2.longest_flatline_ratio.ch3')].mean()),
                    'mean_low_frequency_power_ratio':float(features[:,columns].mean()) if family.availability_['low_frequency_pre_highpass'] else 'N/A',
                    'mean_legacy_quality_score':float(features[:,names.index('F9.mean_quality')].mean()),
                    'adc_clipping_available':False,'pre_highpass_low_frequency_available':family.availability_['low_frequency_pre_highpass'],'synthetic':scenario!='uncorrupted'})
        split_hashes.append({'phase':phase,'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if pickle.dumps(family)!=before:raise AssertionError('Quality controls mutated frozen source state')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'quality_observability_controls.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit={'status':'ok','rows':len(rows),'source_subjects':list(range(1,7)),'source_conditions':['Ramp'],
        'source_state_sha256':hashlib.sha256(before).hexdigest(),'split_sources':split_hashes,
        'feature_names':names,'availability':family.availability_,'evaluation_samples_used':False,'classifier_fit':False,
        'pre_highpass_evidence':'unknown upstream highpass; Info.txt specifies lowpass and loader applies no highpass, but absence of upstream filtering is not proven; native low-frequency ratio remains unavailable',
        'original_info_sha256':hashlib.sha256((raw/'Info.txt').read_bytes()).hexdigest(),
        'limits':'diagnostic synthetic observation controls; no labelled hardware-noise performance or new quality fusion claim'}
    (output/'quality_observability_control_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'status':'ok','rows':len(rows),'feature_dimension':len(names),'evaluation_samples_used':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('processed',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.raw,a.processed,a.output)
