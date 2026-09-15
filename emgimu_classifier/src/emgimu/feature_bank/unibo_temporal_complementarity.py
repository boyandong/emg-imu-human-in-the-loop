"""Original G5 versus reference temporal model errors on held-out UniBo Day 6."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from emgimu.datasets.unibo_baseline import HAND_NAMES
from .unibo_study import _metrics
from .core_incremental_oof import write

PAIRS=(('reference_F0_plus_validated_G5','reference_F0_plus_reference_F5'),
    ('validated_G0_plus_validated_G5','validated_G0_plus_reference_F5'),
    ('reference_F0','reference_F0_plus_validated_G5'),
    ('reference_F0','reference_F0_plus_reference_F5'))


def complementarity(y,a,b,w):
    if len(y)!=len(a) or len(y)!=len(b) or len(y)!=len(w) or np.any(w<=0):
        raise ValueError('Misaligned predictions or nonpositive weights')
    w=w/w.sum();ca=a.argmax(1)==y;cb=b.argmax(1)==y
    ea=(~ca).astype(float);eb=(~cb).astype(float)
    ma=np.sum(w*ea);mb=np.sum(w*eb)
    va=np.sum(w*(ea-ma)**2);vb=np.sum(w*(eb-mb)**2)
    return {'error_correlation':float(np.sum(w*(ea-ma)*(eb-mb))/np.sqrt(va*vb)) if va>1e-15 and vb>1e-15 else 'N/A',
        'disagreement_rate':float(np.sum(w*(a.argmax(1)!=b.argmax(1)))),
        'a_correct_b_wrong':float(np.sum(w*(ca&~cb))),
        'a_wrong_b_correct':float(np.sum(w*(~ca&cb))),
        'both_wrong':float(np.sum(w*(~ca&~cb))),
        'both_correct':float(np.sum(w*(ca&cb)))}


def run(source,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    splits=json.loads((source/'split_trial_ids.json').read_text())
    if manifest['target_days']!=[6] or manifest['train_days']!=[1,2,3,4,5]:
        raise AssertionError('Requires the frozen independent Day-6 comparison')
    if set(splits['train'])&set(splits['validation']):raise AssertionError('Trial leakage')
    with np.load(source/'heldout_predictions.npz',allow_pickle=False) as z:
        y=z['labels'];u=z['subjects'];trials=z['trials'];posture=z['posture'];days=z['days']
        if set(trials)!=set(splits['validation']) or set(days)!={'d06'}:
            raise AssertionError('Held-out trial/day coverage changed')
        # Exactly reproduce established subject/day -> trial -> truth segment -> window weights.
        w=np.zeros(len(y));groups=np.char.add(np.char.add(u.astype(str),'/'),days.astype(str))
        for group in set(groups):
            group_mask=groups==group;group_trials=np.unique(trials[group_mask])
            for trial in group_trials:
                trial_mask=group_mask&(trials==trial);labels=np.unique(y[trial_mask])
                for label in labels:
                    mask=trial_mask&(y==label)
                    w[mask]=1/(len(set(groups))*len(group_trials)*len(labels)*mask.sum())
        w*=len(y)
        originals=list(csv.DictReader((source/'feature_family_results.csv').open(encoding='utf-8')))
        metrics_checked=0
        for name in manifest['specs']:
            reference=next(r for r in originals if r['subject']=='ALL' and r['condition']=='ALL' and r['model']==name)
            score=_metrics(y,z[name],w)
            for metric in ('macro_f1','accuracy','log_loss','brier','ece'):
                np.testing.assert_allclose(score[metric],float(reference[metric]),atol=1e-12,rtol=1e-10)
                metrics_checked+=1
        cells=[('ALL','ALL',np.ones(len(y),bool))]
        cells.extend((user,'ALL',u==user) for user in sorted(set(u)))
        cells.extend(('ALL',f'posture_{p}',posture==p) for p in sorted(set(posture)))
        cells.extend(('ALL',f'class_{HAND_NAMES[int(label)]}',y==label) for label in sorted(set(y)))
        rows=[]
        for a,b in PAIRS:
            for user,condition,mask in cells:
                rows.append({'dataset':'unibo_inail','phase':'validation','subject':user,'condition':condition,
                    'calibration_budget':0,'family_a':a,'family_b':b,'evaluation_windows':int(mask.sum()),
                    'evaluation_trials':len(set(trials[mask])),'weighting':'established hierarchical segment weights; subset renormalized',
                    'scope':'common-baseline concatenated-feature classifiers; not isolated G5 versus F5 specialists',
                    **complementarity(y[mask],z[a][mask],z[b][mask],w[mask])})
    output.mkdir(parents=True)
    write(output/'error_complementarity.csv',rows)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    evidence={'dataset':'unibo_inail','phase':'validation','source_run':source.name,'pairs':PAIRS,
        'classifier_or_family_fit':False,'target_days':[6],'train_days':[1,2,3,4,5],
        'source_artifact_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
            ('heldout_predictions.npz','feature_family_results.csv','run_manifest.json','split_trial_ids.json')},
        'limits':'window errors with hierarchical trial weighting; windows correlated, no independent-window significance claim; no standalone-family or DTW comparison; final days not opened'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))
    (output/'replay_audit.json').write_text(json.dumps({'status':'ok','original_metrics_checked':metrics_checked,'rows':len(rows),'target_windows':len(y)},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows),'original_metrics_checked':metrics_checked}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.source,a.output)
