"""Build labelled held-out feature deltas without averaging unlike datasets."""
from pathlib import Path
import argparse
import csv
import json

SOURCES=(
 ('force','feature_bank_force_audited_validation_20260915','ALL','ALL',None),
 ('wearing','feature_bank_electrode_shift_validation_20260915','ALL','ALL',None),
 ('day','feature_bank_unibo_validation_20260915','ALL','ALL',None),
 ('user','feature_bank_epn612_trial_validation_20260915','ALL','cross_user',None),
 ('posture','feature_bank_emg_fmg_validation_20260915','ALL','ALL','position_shift'),
 ('speed','feature_bank_manus_validation_20260915','SAME_USERS','ALL',None),
)
FACTORS=('force','wearing','day','user','posture','speed','quality')


def build(root:Path,output:Path)->None:
    cells=[];vectors={}
    for factor,run,subject,condition,scenario in SOURCES:
        with (root/run/'feature_family_results.csv').open(encoding='utf-8-sig',newline='') as handle:
            rows=[r for r in csv.DictReader(handle) if r['subject']==subject and (scenario is None or r.get('scenario')==scenario)]
        baseline={r['condition']:r for r in rows if r['feature_family']=='F0'}
        for row in rows:
            family=row['feature_family']
            if not family.startswith('F0+') or len(family.split('+'))!=2:continue
            core=baseline.get(row['condition'])
            if core is None:raise ValueError(f'unmatched baseline {run}: {row}')
            name=family.removeprefix('F0+')
            delta=float(row['macro_f1'])-float(core['macro_f1'])
            cells.append({'family':name,'factor':factor,'run_id':run,'condition':row['condition'],
                'baseline_macro_f1':core['macro_f1'],'candidate_macro_f1':row['macro_f1'],
                'delta_macro_f1':delta,'delta_logloss':float(core['log_loss'])-float(row['log_loss']),
                'delta_brier':float(core['brier'])-float(row['brier'])})
            if row['condition']==condition:
                vectors.setdefault(name,{f:'N/A' for f in FACTORS})[factor]=delta
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('robustness_cells',cells),('robustness_vectors',[{'family':name,**v} for name,v in sorted(vectors.items())])):
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'robustness_vector_definitions.json').write_text(json.dumps({
        'metric':'candidate minus matched F0 validation macro-F1; no cross-dataset average',
        'sources':SOURCES,'quality':'N/A: no matched additive-family quality screen',
        'confounds':{'day':'UniBo day includes reapplication','speed':'MANUS session 1→2 with speed cells; not isolated speed intervention',
                    'posture':'EMG-FMG limb positions; external load mixed in source/target'},
        'topology':'family names retained; four-channel UniBo excludes ring family'},indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','families':len(vectors),'condition_cells':len(cells)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();build(args.root,args.output)
