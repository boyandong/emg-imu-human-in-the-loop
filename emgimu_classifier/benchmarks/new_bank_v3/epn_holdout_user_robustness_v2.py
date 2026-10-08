"""Descriptive user robustness of frozen predictions; no feature/model selection."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,log_loss

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METRICS=('macro_f1','log_loss')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    source=HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'
    result=json.loads(source.read_text(encoding='utf8'))
    users=sorted({b['user'] for b in result['blocks']})
    scores={};identities={}
    for b in result['blocks']:
        identities[b['user'],b['shots']]=b['evaluation_ids']
        for method in ('euclidean','mahalanobis'):
            p=np.asarray(b['probabilities'][method]); y=np.asarray(b['labels'])
            metrics={'macro_f1':float(f1_score(y,p.argmax(1),labels=range(6),average='macro',zero_division=0)),
                     'log_loss':float(log_loss(y,p,labels=range(6)))}
            for key,value in metrics.items():
                if abs(value-b['scores'][method][key])>1e-12:raise ValueError('Frozen native score mismatch')
            scores[b['user'],b['shots'],method]=metrics
    if any(identities[u,10]!=identities[u,20] for u in users):raise ValueError('Budget evaluations differ')
    rows=[]
    for shots in (10,20):
        for method in ('euclidean','mahalanobis'):
            for metric in METRICS:
                values=np.array([scores[u,shots,method][metric] for u in users])
                worse=values.min() if metric=='macro_f1' else values.max()
                rows.append({'shots_per_class':shots,'method':method,'metric':metric,'users':len(users),
                             'equal_user_mean':float(values.mean()),'sample_std_ddof1':float(values.std(ddof=1)),
                             'minimum':float(values.min()),'maximum':float(values.max()),'worst_user_value':float(worse),
                             'worst_users':[u for u,v in zip(users,values) if v==worse],
                             'user_values':dict(zip(map(str,users),map(float,values)))})
    comparisons=[]
    for kind in ('method_at_fixed_budget','budget_at_fixed_method'):
        for setting in ((10,20) if kind=='method_at_fixed_budget' else ('euclidean','mahalanobis')):
            for metric in METRICS:
                details=[]
                for u in users:
                    a,b=(scores[u,setting,'euclidean'][metric],scores[u,setting,'mahalanobis'][metric]) if kind=='method_at_fixed_budget' else (scores[u,10,setting][metric],scores[u,20,setting][metric])
                    gain=b-a if metric=='macro_f1' else a-b
                    details.append({'user':u,'base':a,'alternative':b,'gain_positive_is_better':gain})
                gain=np.array([d['gain_positive_is_better'] for d in details])
                comparisons.append({'kind':kind,'setting':setting,'metric':metric,
                                    'base':'euclidean' if kind=='method_at_fixed_budget' else '10 shots/class',
                                    'alternative':'mahalanobis' if kind=='method_at_fixed_budget' else '20 shots/class',
                                    'wins':int(np.sum(gain>1e-12)),'ties':int(np.sum(abs(gain)<=1e-12)),
                                    'losses':int(np.sum(gain < -1e-12)),
                                    'equal_user_mean_gain':float(gain.mean()),'minimum_user_gain':float(gain.min()),
                                    'maximum_user_gain':float(gain.max()),'details':details})
    csv_path=HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.csv'
    scalar=[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v) for k,v in row.items()} for row in rows]
    with csv_path.open('w',encoding='utf8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(scalar[0]),lineterminator='\n');writer.writeheader();writer.writerows(scalar)
    payload={'schema':'epn_holdout_user_robustness_v2','source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (source,Path(__file__))},
             'csv_sha256':sha(csv_path),'aggregate_rows':rows,'paired_comparisons':comparisons,
             'interpretation':'Mahalanobis improves worst-user F1 and reduces between-user sample SD versus Euclidean at each tested budget. Increasing Mahalanobis calibration from10 to20 raises mean F1 but harms3/10 users and lowers worst-user F1; do not recommend longer calibration from pooled gains alone.',
             'scope':'Equal-user descriptive metrics in one frozen public cohort. Both methods are personally calibrated; no uncalibrated baseline, cross-session proof, seven-axis R_min or population confidence interval is implied.',
             'default_promoted':False,'own_device_efficacy_proven':False,'completion_proven':False}
    (HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print('Saved8 user-robustness rows and8 paired comparisons; no refit',flush=True)

if __name__=='__main__':run()
