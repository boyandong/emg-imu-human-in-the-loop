"""Reverify matched input domains, native probabilities and saved profiles; no fits."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from benchmarks.song_real8.song_matched_normalization_v1 import ROOT,HERE,PROTOCOL,RESULT,OUT,MODES,subset,trial_axis
from benchmarks.song_real8.song_personal_session_workflow_v1 import select,windows,CLASSES,CHANNELS,PREPROCESSING
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.matched_normalization_cli_v1 import load_matched_workflow

ACCEPTANCE=ROOT/'feature_bank/SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(item,ids):
    batch=subset(item,ids);x=batch['batch'].emg.astype(np.float64);y=batch['hand']
    center=np.median(x[y=='neutral'].reshape(-1,8),axis=0)
    scale=np.quantile(np.abs(x[y!='neutral']-center).reshape(-1,8),.95,axis=0)
    return center,scale


def run():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));r=json.loads(RESULT.read_text(encoding='utf8'))
    if sha(PROTOCOL)!=r['protocol_sha256']:raise ValueError('Protocol changed')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen artifact changed: '+name)
    native={}
    for s in ('S01','S02','S03','S04'):
        item=load_session(Path(p['source_folder'])/f'2026-09-18_{s}',s,filter_mode='causal')
        if item['audit']['sha256']!=p['hdf5_sha256'][s]:raise ValueError('Native recording changed')
        item['batch']=FeatureBatch(item['batch'].emg,250.);native[s]=item
    for s in ('S01','S02'):
        ids,y=trial_axis(native[s]);ci,ev=select(ids,y,5,p['selection_seed'])
        if ids[ci].tolist()!=r['source_normalization_ids'][s]:raise ValueError('Source calibration reservation differs')
        if not set(ids[ev])<=set(r['source_model_fit_ids']):raise ValueError('Source fit reservation differs')
        center,scale=oracle(native[s],ids[ci])
        np.testing.assert_array_equal(center,r['source_normalizers'][s]['center'])
        np.testing.assert_array_equal(scale,r['source_normalizers'][s]['scale'])
    previous=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    for name in ('evaluation_ids','personal_calibration_ids','reserved_current_ids'):
        if r[name]!=previous[name]:raise ValueError('Matched target reservation differs')
    current=native['S04'];eb,ei,eo=windows(current,r['evaluation_ids']);cids,cy=trial_axis(current)
    with (OUT/'predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    checked=0;maximum=0.;normalization_error=0.
    with np.load(OUT/'readouts.npz',allow_pickle=False) as saved:
        for mode in MODES:
            w=load_matched_workflow(ROOT/r['source'][mode]['path'],RESULT,mode)
            source_before=pickle.dumps(w.bank);personal=w.load_profile(ROOT/r['profiles'][mode]['personal'],user_id='Song')
            personal_before=pickle.dumps(personal)
            pc,ps=oracle(native['S03'],r['personal_calibration_ids'])
            np.testing.assert_array_equal(personal.normalizer.center_,pc);np.testing.assert_array_equal(personal.normalizer.scale_,ps)
            if mode=='normalized':
                if list(w.bank.source_model_fit_trials_)!=r['source_model_fit_ids']:raise ValueError('Package source fit domain differs')
                if {s:list(v) for s,v in w.bank.source_normalization_trials_.items()}!=r['source_normalization_ids']:
                    raise ValueError('Package source normalization provenance differs')
            for shots in (0,1,2,5):
                session=None;center,scale=pc,ps
                if shots:
                    ci,_=select(cids,cy,shots,p['selection_seed']);expected=cids[ci].tolist()
                    session=w.load_profile(ROOT/r['profiles'][mode]['sessions'][str(shots)],user_id='Song',session_id='S04',personal=personal)
                    if list(session.calibration_trials)!=expected:raise ValueError('Current calibration reservation differs')
                    center,scale=oracle(current,expected)
                    np.testing.assert_array_equal(session.normalizer.center_,center);np.testing.assert_array_equal(session.normalizer.scale_,scale)
                kwargs=dict(window_offsets=eo,personal=personal,session_id='S04',user_id='Song',
                    observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
                full=w.predict(eb,ei,session=session,**kwargs);long_only=w.predict(eb,ei,**kwargs)
                transformed=eb
                if mode=='normalized':
                    transformed=FeatureBatch((eb.emg.astype(np.float64)-center)/(scale+1e-10),250.)
                    actual=w.classifier_input(eb,session=session,**kwargs)
                    normalization_error=max(normalization_error,float(np.max(abs(transformed.emg-actual.emg))))
                    np.testing.assert_array_equal(actual.emg,saved[f'normalized_emg_{shots}'])
                    np.testing.assert_array_equal(center,saved[f'center_{shots}']);np.testing.assert_array_equal(scale,saved[f'scale_{shots}'])
                readout=w.bank.predict_providers(transformed,ei,window_offsets=eo,user_id='Song')
                for g,q in readout['probabilities'].items():np.testing.assert_allclose(q,saved[f'{mode}_{shots}_{g}'],rtol=0,atol=1e-12)
                states=r['states'][f'{mode}_{shots}'];weights=np.asarray(states['weights'])
                probabilities=readout['probabilities'];independent=sum(weights[i]*probabilities[g] for i,g in enumerate(w.bank.providers_))
                np.testing.assert_allclose(independent,full['probabilities'],rtol=0,atol=1e-12)
                arms=dict(F0=probabilities['F0'],population=w.bank.predict(transformed,ei,window_offsets=eo,user_id='Song')['probabilities'],
                          personal=long_only['probabilities'],session=full['probabilities'])
                for omitted in w.bank.providers_:
                    available=[g for g in w.bank.providers_ if g!=omitted]
                    remaining=np.array([weights[w.bank.providers_.index(g)] for g in available]);remaining/=remaining.sum()
                    arms['session_minus_'+omitted]=sum(remaining[i]*probabilities[g] for i,g in enumerate(available))
                for arm,q in arms.items():
                    records=[row for row in rows if row['mode']==mode and int(row['shots'])==shots and row['arm']==arm]
                    if [row['trial_id'] for row in records]!=r['evaluation_ids']:raise ValueError('Saved prediction axis differs')
                    reference=np.array([[float(row['p_'+c]) for c in CLASSES] for row in records])
                    maximum=max(maximum,float(np.max(abs(q-reference))));checked+=len(records)
                if pickle.dumps(w.bank)!=source_before or pickle.dumps(personal)!=personal_before:raise ValueError('Evaluation mutated source/personal state')
    if maximum>1e-12 or normalization_error!=0. or checked!=80*124:raise ValueError('Matched verification incomplete')
    sources=[RESULT,PROTOCOL,Path(__file__),ROOT/'src/emgimu/feature_bank/matched_normalization_cli_v1.py',
             ROOT/'tests/test_matched_normalization_cli_v1.py',ROOT/'tests/test_matched_normalization_delivery_v1.py']
    a=dict(schema='song_matched_normalization_acceptance_v1',source_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in sources},
        artifact_sha256=r['artifact_sha256'],checked_cells=80,checked_trial_probabilities=checked,
        maximum_probability_error=maximum,maximum_normalization_error=normalization_error,
        source_model_fit_trials=len(r['source_model_fit_ids']),source_normalization_trials=sum(map(len,r['source_normalization_ids'].values())),
        source_state_immutable=True,personal_state_immutable=True,primary_pass=r['primary_pass'],
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=p['scope'])
    ACCEPTANCE.write_text(json.dumps(a,indent=2)+'\n',encoding='utf8',newline='\n')
    print(f'Verified all80 cells/{checked} native trial probabilities and exact calibration-only normalization; no refit',flush=True)


if __name__=='__main__':run()
