"""Native full-bout descriptor replay, using previously frozen template candidate IDs."""
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.feature_bank.unibo_sequence_temporal import envelope_path,complete_paths
from emgimu.feature_bank.temporal import PathSignatureFamily
from emgimu.feature_bank.document_path_v3 import DocumentTemporalTemplatesV3,DocumentPathSignatureV3

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    protocol_path=HERE/'F5_PATH_UNIBO_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text())
    source=Path(protocol['frozen_source']);final=Path(protocol['frozen_final']);data=Path(protocol['dataset_root'])
    for name,digest in protocol['frozen_files'].items():
        base=source if name.startswith('source/') else final
        if sha(base/name.split('/',1)[1])!=digest:raise ValueError('Frozen parent changed')
    splits=json.loads((source/'split_trial_ids.json').read_text())
    candidates={i for s in splits for i in s['source_template_candidates']}
    metas=json.loads((source/'bout_metadata.json').read_text())
    wanted=[r for r in metas if r['id'] in candidates or r['day']==6]
    wanted+=json.loads((final/'bout_metadata.json').read_text())
    if len({r['id'] for r in wanted})!=len(wanted):raise ValueError('Duplicate bout metadata')
    paths={};groups=defaultdict(list)
    for row in wanted:groups[row['trial']].append(row)
    files={p.stem:p for p in (data/'trials').rglob('*.npz')}
    if not files:raise PermissionError('Native cache is missing or inaccessible')
    print('[1/3] reconstructing only frozen source candidates and held-out complete envelopes',flush=True)
    for trial_id,rows in groups.items():
        path=files[trial_id];digest=sha(path)
        trial=load_benchmark_trial(path,expected_channels=4,expected_rate_hz=200.)
        if not trial.benchmark_eligible:raise ValueError('Ineligible native trial')
        for row in rows:
            if digest!=row['raw_sha256']:raise ValueError('Native file hash differs')
            start,end=row['start'],row['end']
            if not np.all(trial.hand_label[start:end]==row['label']):raise ValueError('Native bout labels differ')
            paths[row['id']]={**row,'path':envelope_path(trial.emg[start:end])}
    states,temperatures=pickle.loads((source/'fitted_states.pkl').read_bytes());frozen=pickle.dumps((states,temperatures))
    rows=[];selection=[]
    for split in splits:
        user=split['user'];ids=split['source_template_candidates'];cal=[paths[i] for i in ids]
        if len(ids)!=20 or any(r['day']>5 or r['user']!=user for r in cal):raise ValueError('Source candidates differ')
        fitted=DocumentTemporalTemplatesV3().fit(complete_paths(cal),[r['label'] for r in cal],trial_ids=ids)
        signature=DocumentPathSignatureV3().fit(complete_paths(cal))
        old_signature=PathSignatureFamily().fit(complete_paths(cal))
        old_template=states[user][1]['dtw'][0]
        local_frozen=pickle.dumps((fitted,signature,old_signature))
        selection.append({'user':user,'candidate_ids':ids,'medoid_ids':list(fitted.medoid_trial_ids_),
                          'classes':fitted.classes_.tolist(),'source_band':fitted.band_})
        print(f'[2/3] {user}: frozen 20 candidates; replay Days6-8',flush=True)
        for phase,days in (('validation',(6,)),('descriptive_final',(7,8))):
            target=[paths[r['id']] for r in wanted if r['user']==user and r['day'] in days]
            if {r['trial'] for r in target}&{r['trial'] for r in cal}:raise ValueError('Native trial overlap')
            target_batch=complete_paths(target)
            new=fitted.transform(target_batch,trial_ids=[r['id'] for r in target])
            old=old_template.transform(target_batch)
            sig=signature.transform(target_batch);old_sig=old_signature.transform(target_batch)
            for r,dist,baseline,s,old_s in zip(target,new,old,sig,old_sig):
                rows.append({'phase':phase,'id':r['id'],'trial':r['trial'],'user':user,'day':r['day'],
                    'label':r['label'],'duration_seconds':r['duration_seconds'],
                    **{f'dtw_{c}':float(v) for c,v in zip(fitted.classes_,dist)},
                    **{f'legacy_dtw_{c}':float(v) for c,v in zip(fitted.classes_,baseline)},
                    **{f'signature_{i}':float(v) for i,v in enumerate(s)},
                    'signature_legacy_max_abs_difference':float(np.max(np.abs(s-old_s))),
                    'prediction':int(fitted.classes_[np.argmin(dist)]),
                    'legacy_prediction':int(fitted.classes_[np.argmin(baseline)])})
        if pickle.dumps((fitted,signature,old_signature))!=local_frozen:raise ValueError('Local source state mutated')
    if pickle.dumps((states,temperatures))!=frozen:raise ValueError('Legacy parent state mutated')
    path=HERE/'F5_PATH_UNIBO_ROWS.csv'
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    by_phase={}
    for phase in ('validation','descriptive_final'):
        selected=[r for r in rows if r['phase']==phase]
        by_phase[phase]={'bouts':len(selected),'users':len({r['user'] for r in selected}),
            'changed_nearest_template_decisions':sum(r['prediction']!=r['legacy_prediction'] for r in selected),
            'new_bout_accuracy':float(np.mean([r['prediction']==r['label'] for r in selected])),
            'legacy_bout_accuracy':float(np.mean([r['legacy_prediction']==r['label'] for r in selected]))}
    if by_phase['descriptive_final']['bouts']!=3391:raise ValueError('Frozen final inventory differs')
    evidence=['src/emgimu/feature_bank/document_path_v3.py','benchmarks/new_bank_v3/f5_path_unibo_run.py']
    result={'protocol_sha256':sha(protocol_path),'row_sha256':sha(path),'rows':len(rows),
        'source_hashes':{p:sha(ROOT/p) for p in evidence},'source_selection':selection,
        'by_phase':by_phase,'signature_dimension':20,'dtw_dimension':4,
        'classifier_fit':False,'target_labels_used_for_fitting':False,
        'max_signature_legacy_abs_difference':max(r['signature_legacy_max_abs_difference'] for r in rows),
        'scope':protocol['scope']}
    (HERE/'F5_PATH_UNIBO_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('[3/3] '+json.dumps(by_phase),flush=True)


if __name__=='__main__':run()
