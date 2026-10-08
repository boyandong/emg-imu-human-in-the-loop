"""Plot frozen holdout performance against measured signal exposure, without refitting."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    result_path = HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'
    burden_path = HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json'
    result = json.loads(result_path.read_text())
    burden = json.loads(burden_path.read_text())
    for path, digest in burden['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Calibration cost evidence changed: '+path)
    costs = {(r['subject'],r['shots_per_class']):r for r in burden['records']}
    rows = []
    for block in result['blocks']:
        cost = costs[block['user'],block['shots']]
        for method in ('euclidean','mahalanobis'):
            rows.append({'subject':block['user'],'shots_per_class':block['shots'],
                         'method':method,'calibration_trials':cost['used_calibration_trials'],
                         'signal_seconds':cost['used_signal_seconds'],
                         'full_recording_seconds':cost['used_trials_full_recording_seconds'],
                         'macro_f1':block['scores'][method]['macro_f1'],
                         'log_loss':block['scores'][method]['log_loss']})
    fig, axes = plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    colors = {'euclidean':'#64748b','mahalanobis':'#167a69'}
    for axis, metric, title in zip(axes,('macro_f1','log_loss'),('Macro-F1 (higher is better)','Log loss (lower is better)')):
        for method,color in colors.items():
            for user in sorted({r['subject'] for r in rows}):
                points = sorted([r for r in rows if r['subject']==user and r['method']==method],key=lambda r:r['shots_per_class'])
                axis.plot([r['signal_seconds'] for r in points],[r[metric] for r in points],color=color,alpha=.2,lw=1)
            axis.plot([48,96],[result['scores'][str(s)][method][metric] for s in (10,20)],color=color,marker='o',lw=2.5,label=method.title()+' (pooled trials)')
        axis.set_title(title)
        axis.set_xlabel('Used EMG exposure per user (seconds)')
        axis.set_xticks([48,96],['48 s\n10 shots/class','96 s\n20 shots/class'])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('EPN612: frozen users 32-41, six native classes')
    fig.supxlabel('Faint lines: individual users. Stored recordings: 296-299 s / 592-598 s. Device elapsed time unmeasured.',fontsize=8)
    figures = []
    for suffix in ('svg','png'):
        output = HERE/('MAHALANOBIS_EPN_HOLDOUT_V2_COST_CURVE.'+suffix)
        fig.savefig(output,dpi=160)
        if suffix == 'svg':
            output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
        figures.append(output)
    plt.close(fig)
    payload = {'schema':'epn_holdout_cost_curve_v2',
               'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (result_path,burden_path,Path(__file__))},
               'figure_sha256':{p.name:sha(p) for p in figures},'rows':rows,
               'calibration_requirements':{
                   'all_task_gestures_required':True,'required_native_gesture_classes':6,
                   'basis':'Both prototypes and within-class covariance require labelled target calibration for all six classes.',
                   'each_target_session':'N/A: repeated-session transfer not evaluated in this cohort',
                   'target_force_required':'N/A: force labels and unseen-force transfer not controlled in this experiment',
                   'target_posture_required':'N/A: posture labels and unseen-posture transfer not controlled in this experiment',
                   'physical_session_seconds':'N/A: setup, prompts, rest and elapsed device time unmeasured'},
               'pooled_scores':result['scores'],
               'scope':'New independent 10/20-shot confirmation only; do not extend curves to unsupported budgets or equate exposure with full recording/physical time.',
               'few_second_calibration_proven':False,'device_efficacy_proven':False,'default_promoted':False}
    (HERE/'MAHALANOBIS_EPN_HOLDOUT_V2_COST_CURVE.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print('Saved frozen calibration-cost curve: 40 method/user/budget rows',flush=True)

if __name__ == '__main__':
    run()
