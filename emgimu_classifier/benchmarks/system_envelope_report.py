"""Descriptive envelope of the frozen benchmark-specific experts, not one universal model."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def load(root,run,name='feature_family_results.csv'):
    with (root/run/name).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def build(root:Path,output:Path)->None:
    rows=[]
    specs=(('wearing','feature_bank_electrode_shift_audited_final_20260915','ALL','F0+F3_Ring','trial_'),
        ('day','feature_bank_unibo_audited_final_20260915','ALL','F0+F5_Temporal','day_'),
        ('posture','feature_bank_emg_fmg_audited_final_20260915','ALL','F0+F2c_SPD','position_'),
        ('speed','feature_bank_manus_final_20260915','SAME_USERS','F0+F2c_SPD',None))
    for factor,run,subject,family,prefix in specs:
        source=[r for r in load(root,run) if r['subject']==subject and (factor!='posture' or r['scenario']=='position_shift')]
        base=next(r for r in source if r['feature_family']=='F0' and r['condition']=='ALL')
        candidate=next(r for r in source if r['feature_family']==family and r['condition']=='ALL')
        worst=[]
        for name in ('F0',family):
            cells=[r for r in source if r['feature_family']==name and r['condition']!='ALL' and (prefix is None or r['condition'].startswith(prefix))]
            worst.append(min(float(r['macro_f1']) for r in cells))
        rows.append({'factor':factor,'run_id':run,'candidate_model':family,'baseline_R':float(base['macro_f1']),
            'candidate_R':float(candidate['macro_f1']),'baseline_worst_condition':worst[0],
            'candidate_worst_condition':worst[1],'delta_R':float(candidate['macro_f1'])-float(base['macro_f1'])})
    force=next(r for r in load(root,'feature_bank_force_product_final_20260915','calibration_curve.csv') if r['subject']=='ALL' and r['condition']=='ALL' and r['shots_per_class']=='0')
    rows.append({'factor':'force','run_id':'feature_bank_force_product_final_20260915','candidate_model':force['feature_bank'],
        'baseline_R':'N/A','candidate_R':float(force['macro_f1']),'baseline_worst_condition':'N/A','candidate_worst_condition':'N/A','delta_R':'N/A'})
    raw=[float(r['zero_shot_same_eval_macro_f1']) for r in load(root,'feature_bank_epn_calibration_diagnostics_final_20260915_v2','per_family_calibration_gain.csv') if r['family']=='F0' and r['shots_per_class']=='0']
    full=next(r for r in load(root,'feature_bank_full_fusion_final_20260915','calibration_curve.csv') if r['subject']=='ALL' and r['shots_per_class']=='0' and r['method']=='full')
    baseline=float(np.mean(raw));candidate=float(full['macro_f1'])
    rows.append({'factor':'user','run_id':'feature_bank_full_fusion_final_20260915','candidate_model':'eight-provider uniform zero-shot fusion',
        'baseline_R':baseline,'candidate_R':candidate,'baseline_worst_condition':'N/A','candidate_worst_condition':'N/A','delta_R':candidate-baseline})
    rows.append({'factor':'quality','run_id':'N/A','candidate_model':'N/A','baseline_R':'N/A','candidate_R':'N/A',
        'baseline_worst_condition':'N/A','candidate_worst_condition':'N/A','delta_R':'N/A'})
    rows.sort(key=lambda r:('force','wearing','day','user','posture','speed','quality').index(r['factor']))
    available=[r['candidate_R'] for r in rows if r['candidate_R']!='N/A']
    paired=[r for r in rows if r['baseline_R']!='N/A' and r['candidate_R']!='N/A']
    summary={'scope':'frozen benchmark-specific expert package; different native tasks and aggregations; not one universally evaluated full model',
        'mean_R_available':float(np.mean(available)),'R_min_available':float(min(available)),
        'paired_factors':[r['factor'] for r in paired],
        'paired_baseline_mean':float(np.mean([r['baseline_R'] for r in paired])),
        'paired_candidate_mean':float(np.mean([r['candidate_R'] for r in paired])),
        'paired_baseline_R_min':float(min(r['baseline_R'] for r in paired)),
        'paired_candidate_R_min':float(min(r['candidate_R'] for r in paired)),
        'missing':'force final F0 reference and real-quality benchmark; multi-failure universal full-system comparison incomplete',
        'warning':'descriptive mean across unlike tasks; no common-population accuracy inference; session/speed/reapplication confounds remain'}
    with (output/'system_robustness_envelope.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'system_robustness_envelope.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();build(a.root,a.output)
