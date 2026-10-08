import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'

def test_independent_user_confusion_and_variation_statistics():
    audit=json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json').read_text(encoding='utf8'))
    result=json.loads((HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json').read_text(encoding='utf8'))
    for path,digest in audit['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    table=HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.csv'
    assert hashlib.sha256(table.read_bytes()).hexdigest()==audit['csv_sha256']
    with table.open(encoding='utf8',newline='') as stream:assert len(list(csv.DictReader(stream)))==8
    independent={}
    for b in result['blocks']:
        y=np.asarray(b['labels'])
        for method in ('euclidean','mahalanobis'):
            p=np.asarray(b['probabilities'][method]);prediction=p.argmax(1)
            f=[]
            for label in range(6):
                tp=np.sum((y==label)&(prediction==label));support=np.sum(y==label);called=np.sum(prediction==label)
                f.append(2*tp/(support+called) if support+called else 0)
            independent[b['user'],b['shots'],method,'macro_f1']=sum(f)/6
            independent[b['user'],b['shots'],method,'log_loss']=-sum(np.log(p[i,label]) for i,label in enumerate(y))/len(y)
    for row in audit['aggregate_rows']:
        users=sorted(map(int,row['user_values']));v=[independent[u,row['shots_per_class'],row['method'],row['metric']] for u in users]
        mean=sum(v)/len(v);sd=(sum((x-mean)**2 for x in v)/(len(v)-1))**.5
        assert abs(mean-row['equal_user_mean'])<1e-12 and abs(sd-row['sample_std_ddof1'])<1e-12
        assert abs(min(v)-row['minimum'])<1e-12 and abs(max(v)-row['maximum'])<1e-12
        worst=min(v) if row['metric']=='macro_f1' else max(v)
        assert abs(worst-row['worst_user_value'])<1e-12
        assert row['worst_users']==[u for u,x in zip(users,v) if abs(x-worst)<1e-12]
    for comparison in audit['paired_comparisons']:
        differences=[]
        for d in comparison['details']:
            metric=comparison['metric'];setting=comparison['setting'];u=d['user']
            keys=((u,setting,'euclidean',metric),(u,setting,'mahalanobis',metric)) if comparison['kind']=='method_at_fixed_budget' else ((u,10,setting,metric),(u,20,setting,metric))
            a,b=[independent[k] for k in keys];gain=b-a if metric=='macro_f1' else a-b
            assert abs(d['gain_positive_is_better']-gain)<1e-12
            differences.append(gain)
        assert comparison['wins']==sum(g>1e-12 for g in differences)
        assert comparison['losses']==sum(g < -1e-12 for g in differences)
        assert comparison['wins']+comparison['ties']+comparison['losses']==10
    c=next(c for c in audit['paired_comparisons'] if c['kind']=='budget_at_fixed_method' and c['setting']=='mahalanobis' and c['metric']=='macro_f1')
    assert c['losses']==3 and c['wins']==7
    assert not audit['default_promoted'] and not audit['own_device_efficacy_proven'] and not audit['completion_proven']
