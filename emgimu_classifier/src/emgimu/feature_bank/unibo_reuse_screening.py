"""Day-6 comparison of existing UniBo G0/G5 versus the new reference families."""
from pathlib import Path
import argparse
import csv
import json
import pickle
import numpy as np
from emgimu.datasets.unibo_physiology import load_chronological_raw_windows,chronological_fold
from .unibo_full_fusion import batch,weights,classifier
from .unibo_study import _metrics
from .families import LocalDetailFamily,TemporalFormFamily
from .validated_unibo import ValidatedUniBoFamily


def run(dataset,output):
    if output.exists():raise FileExistsError(output)
    print('[1/3] source days1-5; validation day6 only',flush=True)
    data=load_chronological_raw_windows(dataset,range(1,7));train,target=chronological_fold(data,(1,2,3,4,5),6,6)
    if set(train.trial_id)&set(target.trial_id):raise AssertionError('trial leakage')
    families={'F0_reference':LocalDetailFamily(),'F0_validated_G0':ValidatedUniBoFamily('G0'),
        'F5_reference':TemporalFormFamily(),'F5_validated_G5':ValidatedUniBoFamily('G5')}
    tw,vw=weights(train),weights(target);a={};b={};dimensions={};states={};predictions={};rows=[]
    for name,f in families.items():
        print(f'[2/3] {name}',flush=True)
        f.fit(batch(train),train.labels,tw) if isinstance(f,ValidatedUniBoFamily) else f.fit(batch(train),train.labels)
        before=pickle.dumps(f);a[name]=f.transform(batch(train));b[name]=f.transform(batch(target))
        if pickle.dumps(f)!=before:raise AssertionError('validation changed family state')
        dimensions[name]=a[name].shape[1]
    specs={'reference_F0':('F0_reference',),'validated_G0':('F0_validated_G0',),
        'reference_F0_plus_reference_F5':('F0_reference','F5_reference'),
        'reference_F0_plus_validated_G5':('F0_reference','F5_validated_G5'),
        'validated_G0_plus_validated_G5':('F0_validated_G0','F5_validated_G5'),
        'validated_G0_plus_reference_F5':('F0_validated_G0','F5_reference')}
    for name,members in specs.items():
        s,m=classifier(np.concatenate([a[n] for n in members],1),train.labels,tw)
        p=m.predict_proba(s.transform(np.concatenate([b[n] for n in members],1)));states[name]=(s,m,members);predictions[name]=p
        cells=[('ALL','ALL',np.ones(len(target),bool))]
        cells.extend(('ALL',f'posture_{v}',target.posture==v) for v in range(1,5))
        cells.extend((u,'ALL',target.subject_id==u) for u in sorted(set(target.subject_id)))
        for u,condition,mask in cells:
            rows.append({'dataset':'unibo_inail','phase':'validation','subject':u,'condition':condition,
                'feature_family':'+'.join(members),'model':name,'calibration_budget':0,
                'feature_dimension':sum(dimensions[n] for n in members),**_metrics(target.labels[mask],p[mask],vw[mask])})
    print('[3/3] saving compatibility comparison',flush=True);output.mkdir(parents=True)
    with (output/'feature_family_results.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'fitted_states.pkl').open('wb') as h:pickle.dump((families,states),h)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions,labels=target.labels,trials=target.trial_id,
        subjects=target.subject_id,days=target.session_id,posture=target.posture)
    (output/'split_trial_ids.json').write_text(json.dumps({'train':sorted(set(train.trial_id)),'validation':sorted(set(target.trial_id))},indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'target_days':[6],'train_days':[1,2,3,4,5],'dimensions':dimensions,
        'specs':specs,'legacy_source':'emgimu.datasets.unibo_physiology.PhysiologyFeatureTransformer',
        'native_channels':4,'sample_rate_hz':200,'final_days_opened':False,
        'scope':'validated native UniBo reuse; historical DS2 formula equivalence remains unavailable'},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.dataset,a.output)
