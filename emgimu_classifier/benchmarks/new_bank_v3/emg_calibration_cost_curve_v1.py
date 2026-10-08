"""Plot immutable performance versus actual extracted calibration exposure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.new_bank_v3.emg_window_bank_v1 import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'EMG_CALIBRATION_COST_CURVE_V1.json'


def run():
    cost_path=HERE/'EMG_CALIBRATION_BURDEN_V1.json'
    cost=json.loads(cost_path.read_text(encoding='utf8'))
    for path,digest in cost['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Changed cost evidence')
    runs=[('EMG_F0_F7_BANK_V1','F0_F7','EPN users 42-51: F0 + F7'),
          ('EMG_CALIBRATED_FUSION_V1','reliability_bank','EPN users 62-71: reliability fusion')]
    sources={cost_path.relative_to(ROOT).as_posix():sha(cost_path)};cells=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    notes=[]
    for row,(run_id,method,title) in enumerate(runs):
        path=HERE/(run_id+'_RESULTS.json');r=json.loads(path.read_text(encoding='utf8'))
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        for arm in (('F0',method) if row==0 else ('F0','uniform_bank',method)):
            budgets=(0,) if arm in ('F0','uniform_bank') else ((1,2,5) if row==0 else (0,1,2,5))
            for shots in budgets:
                costs=[c for c in cost['records'] if c['run_id']==run_id and c['feature_bank']==arm and c['shots_per_class']==shots]
                if len(costs)!=10:raise ValueError('Missing ten-user native cost cells')
                score=r['scores'][str(shots)][arm]['pooled']
                cells.append({'run_id':run_id,'feature_bank':arm,'shots_per_class':shots,
                    'users':10,'evaluation_trials':1200,'actual_calibration_trials_per_user':costs[0]['used_calibration_trials'],
                    'used_signal_seconds_mean_per_user':float(np.mean([c['used_signal_seconds'] for c in costs])),
                    'used_signal_seconds_min_per_user':min(c['used_signal_seconds'] for c in costs),
                    'used_signal_seconds_max_per_user':max(c['used_signal_seconds'] for c in costs),
                    'full_recording_seconds_min_per_user':min(c['used_trials_full_recording_seconds'] for c in costs),
                    'full_recording_seconds_max_per_user':max(c['used_trials_full_recording_seconds'] for c in costs),
                    'macro_f1':score['macro_f1'],'log_loss':score['log_loss']})
        points=sorted([c for c in cells if c['run_id']==run_id and c['feature_bank']==method],key=lambda c:c['shots_per_class'])
        for column,(metric,heading) in enumerate((('macro_f1','Macro-F1 (higher is better)'),('log_loss','Log loss (lower is better)'))):
            ax=axes[row,column]
            ax.plot([p['used_signal_seconds_mean_per_user'] for p in points],[p[metric] for p in points],marker='o',lw=2.2,color='#167a69',label=method)
            base=next(c for c in cells if c['run_id']==run_id and c['feature_bank']=='F0')
            ax.axhline(base[metric],color='#475569',ls='--',lw=1.5,label='F0: 0 target trials')
            if row:
                uniform=next(c for c in cells if c['run_id']==run_id and c['feature_bank']=='uniform_bank')
                ax.axhline(uniform[metric],color='#a16207',ls=':',lw=1.5,label='Uniform: 0 target trials')
            for p in points:ax.annotate(f"{p['shots_per_class']}/class",(p['used_signal_seconds_mean_per_user'],p[metric]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set_title(title+'\n'+heading,fontsize=10);ax.set_xlabel('Mean used EMG exposure per user (seconds)')
            ax.set_xticks([0,4.8,9.6,24]);ax.set_xlim(-1.5,26);ax.margins(y=.2);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
        five=next(c for c in points if c['shots_per_class']==5)
        notes.append(f"Users {42 if row==0 else 62}-{51 if row==0 else 71}: 5/class =30 trials; complete stored records {five['full_recording_seconds_min_per_user']:.1f}-{five['full_recording_seconds_max_per_user']:.1f} s/user.")
    fig.suptitle('Calibration benefit and burden: separate frozen public cohorts',fontsize=13)
    fig.supxlabel('\n'.join(notes)+'\nExposure is not physical session wall time. Hardware setup, guided rest and elapsed time are unmeasured.',fontsize=9)
    figures=[]
    for suffix in ('svg','png'):
        path=HERE/('EMG_CALIBRATION_COST_CURVE_V1.'+suffix);fig.savefig(path,dpi=160)
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
        figures.append(path)
    plt.close(fig)
    for path in (Path(__file__),ROOT/'tests/test_emg_calibration_cost_curve_v1.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    payload={'schema':'emg_calibration_cost_curve_v1','source_sha256':sources,'cells':cells,
        'figure_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in figures},
        'interpretation':'F0/F7 mixture has limited gains; six-provider reliability fusion improves five-shot F1 but worsens probability loss. Calibration benefit is not monotone. Separate cohorts and models are not pooled. Zero-shot F7 is unavailable and is not plotted as F0/F7.',
        'few_second_calibration_proven':False,'physical_wall_time_proven':False,'default_promoted':False,'completion_proven':False}
    OUT.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8',newline='\n')
    print(f'Saved {len(cells)} performance/cost cells and two figure formats; no refit',flush=True)


if __name__=='__main__':run()
