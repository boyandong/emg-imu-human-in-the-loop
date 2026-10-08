import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'


def test_cost_curves_use_actual_method_costs_native_scores_and_unavailable_zero_shot_anchor():
    d=json.loads((HERE/'EMG_CALIBRATION_COST_CURVE_V1.json').read_text(encoding='utf8'))
    for path,digest in {**d['source_sha256'],**d['figure_sha256']}.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    cost=json.loads((HERE/'EMG_CALIBRATION_BURDEN_V1.json').read_text(encoding='utf8'))
    results={run:json.loads((HERE/(run+'_RESULTS.json')).read_text(encoding='utf8')) for run in {c['run_id'] for c in d['cells']}}
    assert len(d['cells'])==10 and len(d['figure_sha256'])==2
    assert not d['few_second_calibration_proven'] and not d['physical_wall_time_proven'] and not d['default_promoted'] and not d['completion_proven']
    assert not any(c['feature_bank']=='F0_F7' and c['shots_per_class']==0 for c in d['cells'])
    for cell in d['cells']:
        run=cell['run_id'];arm=cell['feature_bank'];shots=cell['shots_per_class']
        r=results[run]
        rows=[c for c in cost['records'] if (c['run_id'],c['feature_bank'],c['shots_per_class'])==(run,arm,shots)]
        assert len(rows)==cell['users']==10 and cell['evaluation_trials']==1200
        assert all(c['used_calibration_trials']==cell['actual_calibration_trials_per_user'] for c in rows)
        assert cell['used_signal_seconds_mean_per_user']==np.mean([c['used_signal_seconds'] for c in rows])
        for prefix,field in [('used_signal','used_signal_seconds'),('full_recording','used_trials_full_recording_seconds')]:
            assert cell[prefix+'_seconds_min_per_user']==min(c[field] for c in rows)
            assert cell[prefix+'_seconds_max_per_user']==max(c[field] for c in rows)
        for metric in ('macro_f1','log_loss'):assert cell[metric]==r['scores'][str(shots)][arm]['pooled'][metric]
        if arm in ('F0','uniform_bank'):
            assert cell['actual_calibration_trials_per_user']==cell['used_signal_seconds_mean_per_user']==0
        else:assert cell['actual_calibration_trials_per_user']==6*shots
