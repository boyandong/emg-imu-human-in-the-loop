"""Condition-level robustness from saved force predictions, without fitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from emgimu.datasets.libemg_force import FILE_RE
from .screening import metrics


def trial_conditions(trials, labels, users):
    conditions=[]
    for trial,label,user in zip(trials,labels,users):
        match=FILE_RE.fullmatch(str(trial)+'.csv')
        if match is None or int(match['label'])-1!=label or int(match['subject'])!=user:
            raise ValueError('saved trial metadata does not match native identifier')
        conditions.append(match['condition'])
    return np.asarray(conditions)


def run(source:Path,output:Path):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    splits=json.loads((source/'split_trial_ids.json').read_text())
    rows=[];pooled={}
    path=source/'heldout_predictions.npz'
    with np.load(path,allow_pickle=False) as saved:
        y,u,trials=saved['labels'],saved['users'],saved['trials']
        conditions=trial_conditions(trials,y,u)
        for split in splits:
            user,shots=split['user'],split['shots']
            ev=np.flatnonzero(np.isin(trials,split['evaluation']))
            if set(split['calibration'])&set(split['evaluation']) or not np.all(u[ev]==user):
                raise ValueError('invalid evaluation partition')
            if len(ev)!=len(split['evaluation']):raise ValueError('missing evaluation trial')
            prefix=f'{user}_{shots}_'
            for key in saved.files:
                if not key.startswith(prefix):continue
                method=key[len(prefix):];p=saved[key]
                if len(p)!=len(ev):raise ValueError('probability/trial length mismatch')
                for condition in sorted(set(conditions[ev])):
                    mask=conditions[ev]==condition
                    rows.append({'dataset':'libemg_contraction_intensity','phase':manifest['phase'],
                        'subject':user,'condition':condition,'shots_per_class':shots,'method':method,
                        'evaluation_trials':int(mask.sum()),'aggregation':'native subject-condition trial mean classifier',
                        **metrics(y[ev][mask],p[mask],np.ones(mask.sum()))})
                    pooled.setdefault((shots,method,condition),[]).append((y[ev][mask],p[mask]))
        for (shots,method,condition),values in pooled.items():
            labels=np.concatenate([v[0] for v in values]);p=np.concatenate([v[1] for v in values])
            rows.append({'dataset':'libemg_contraction_intensity','phase':manifest['phase'],'subject':'ALL',
                'condition':condition,'shots_per_class':shots,'method':method,'evaluation_trials':len(labels),
                'aggregation':'pooled target-user trials within condition',**metrics(labels,p,np.ones(len(labels)))})
    summary=[]
    for shots,method in sorted({(r['shots_per_class'],r['method']) for r in rows}):
        cells=[r for r in rows if r['subject']=='ALL' and r['shots_per_class']==shots and r['method']==method]
        worst=min(cells,key=lambda r:r['macro_f1'])
        native=[r for r in rows if r['subject']!='ALL' and r['shots_per_class']==shots and r['method']==method]
        worst_native=min(native,key=lambda r:r['macro_f1'])
        summary.append({'dataset':'libemg_contraction_intensity','phase':manifest['phase'],'shots_per_class':shots,'method':method,
            'mean_condition_macro_f1':float(np.mean([r['macro_f1'] for r in cells])),
            'minimum_condition_macro_f1':worst['macro_f1'],'worst_condition':worst['condition'],
            'minimum_subject_condition_macro_f1':worst_native['macro_f1'],
            'worst_subject':worst_native['subject'],'worst_subject_condition':worst_native['condition'],
            'conditions_evaluated':len(cells)})
    output.mkdir(parents=True)
    for name,values in (('ablation_full_bank',rows),('force_worst_condition',summary)):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (output/'run_manifest.json').write_text(json.dumps({'source_run':source.name,'phase':manifest['phase'],
        'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'classifier_or_family_fit':False,
        'scope':'eleven native target-force conditions; minimum condition is not minimum across failure dimensions',
        'evaluation_trial_ids':trials.tolist()},indent=2))
    print(json.dumps({'status':'ok','condition_cells':len(rows),'robustness_summary_rows':len(summary)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
