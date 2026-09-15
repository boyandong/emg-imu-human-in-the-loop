"""Inspect inter-user variation and class summaries without changing source metrics."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics


def build(results):
    groups={};hashes={};seen={}
    for filename in ('calibration_curve.csv','ablation_full_bank.csv','feature_family_results.csv'):
        path=results/filename;hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='utf-8-sig',newline='') as h:
            for row in csv.DictReader(h):
                try:int(row['subject']);float(row['macro_f1']);float(row['log_loss'])
                except (ValueError,KeyError,TypeError):continue
                method=row.get('method') or row.get('model') or row.get('feature_family') or row.get('family')
                budget=row.get('shots_per_class') or row.get('calibration_budget') or 'N/A'
                key=(row['run_id'],row.get('dataset',''),method,budget,row.get('condition',''),
                     row.get('scenario',''),row.get('protocol',''),row.get('feature_family') or row.get('family') or '',
                     row.get('n0',''),row.get('reliability_temperature',''),row.get('feature_bank',''))
                identity=(key,row['subject'])
                metric=(float(row['macro_f1']),float(row['log_loss']))
                if identity in seen:
                    if seen[identity]!=metric:raise ValueError(f'Conflicting duplicate subject metrics: {identity}')
                    continue
                seen[identity]=metric;groups.setdefault(key,[]).append(row)
    output=[]
    for key,rows in groups.items():
        f1=[float(r['macro_f1']) for r in rows];loss=[float(r['log_loss']) for r in rows]
        classes=[]
        for row in rows:
            try:parsed=json.loads(row.get('per_class_f1_json',''))
            except (ValueError,TypeError):continue
            if isinstance(parsed,dict) and all(isinstance(v,(int,float)) for v in parsed.values()):classes.append(parsed)
        class_summary='N/A'
        if len(classes)==len(rows) and all(set(c)==set(classes[0]) for c in classes):
            class_summary=json.dumps({h:statistics.mean(c[h] for c in classes) for h in classes[0]},sort_keys=True)
        output.append(dict(zip(('run_id','dataset','method','calibration_budget','condition','scenario','protocol','feature_family',
                                'n0','reliability_temperature','feature_bank'),key))|
                      {'subjects':len(rows),'subject_ids_json':json.dumps(sorted(int(r['subject']) for r in rows)),
                       'mean_user_macro_f1':statistics.mean(f1),'std_user_macro_f1':statistics.pstdev(f1),
                       'min_user_macro_f1':min(f1),'max_user_macro_f1':max(f1),'mean_user_log_loss':statistics.mean(loss),
                       'per_class_mean_user_f1_json':class_summary})
    write(results/'per_subject_analysis.csv',output)
    comparisons=[]
    for run in ('feature_bank_epn_probability_final_20260915','feature_bank_epn_selected_final_20260915'):
        for budget in ('0','1','2','5'):
            selected=[r for r in output if r['run_id']==run and r['calibration_budget']==budget and r['condition']=='cross_user']
            def one(method):
                found=[r for r in selected if r['method']==method]
                if len(found)!=1:raise ValueError(f'Ambiguous cross-user summary {run} {budget} {method}')
                return found[0]
            anchor=one('full');base=one('without_F7_anchor')
            if anchor['subject_ids_json']!=base['subject_ids_json']:raise ValueError('Different anchor user populations')
            comparisons.append({'run_id':run,'shots_per_class':budget,'subjects':anchor['subjects'],
                'without_anchor_mean_f1':base['mean_user_macro_f1'],'anchor_mean_f1':anchor['mean_user_macro_f1'],
                'without_anchor_std_f1':base['std_user_macro_f1'],'anchor_std_f1':anchor['std_user_macro_f1'],
                'without_anchor_min_f1':base['min_user_macro_f1'],'anchor_min_f1':anchor['min_user_macro_f1'],
                'delta_std_f1':anchor['std_user_macro_f1']-base['std_user_macro_f1'],
                'delta_mean_f1':anchor['mean_user_macro_f1']-base['mean_user_macro_f1']})
    write(results/'anchor_cross_user_variation.csv',comparisons)
    audit={'status':'ok','subject_condition_summaries':len(output),'anchor_comparisons':len(comparisons),
           'source_csv_sha256':hashes,'scope':'descriptive population standard deviation over observed users; no confidence/significance claim; unchanged source metric rows',
           'limitations':'three final EPN users; budgets change evaluation trial sets; class summaries N/A when source class values unavailable'}
    (results/'per_subject_analysis_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','subject_condition_summaries','anchor_comparisons')}))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path)
    args=parser.parse_args();build(args.results)
