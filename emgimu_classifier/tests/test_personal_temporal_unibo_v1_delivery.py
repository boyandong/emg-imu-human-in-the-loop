"""Native retained cells, frozen source reuse and explicit failure of promotion."""
import json
import csv
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.verify_personal_temporal_unibo_v1 import verify,metrics

ROOT=Path(__file__).resolve().parents[1]


def test_native_temporal_delivery_reconstructs_every_cell_and_acceptance():
    actual=verify()
    saved=json.loads((ROOT/'feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json').read_text())
    assert actual==saved
    assert actual['retained_predictions']==187044 and actual['arm_cells']==924
    assert actual['matched_evaluation_bouts']==4251
    assert actual['native_inrun_DP_medoid_signature_probability_oracle_max_error']<1e-12
    assert actual['user_loss_wins']==1 and not actual['primary_pass']
    assert actual['primary_guards']['nonworse_macro_f1']
    assert not actual['primary_guards']['lower_log_loss']
    assert not actual['default_promoted'] and not actual['physical_validation_proven']


def test_weighted_metric_oracle_uses_trial_mass_not_bout_count():
    y=np.array([0,0,1,1]);q=np.array([[.8,.2],[.3,.7],[.4,.6],[.7,.3]])
    w=np.array([.1,.4,.2,.3]);result=metrics(y,q,w,['a','b'])
    assert abs(result['accuracy']-.3)<1e-14
    assert abs(result['per_class_recall']['a']-.2)<1e-14
    assert abs(result['per_class_recall']['b']-.4)<1e-14
    assert abs(result['macro_f1']-(2*.1/(.5+.4)+2*.2/(.5+.6))/2)<1e-14


def test_canonical_temporal_rows_preserve_weighted_scores_errors_and_costs():
    result=json.loads((ROOT/'benchmarks/new_bank_v3/PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json').read_text())
    blocks={(b['user'],f"Day{b['day']}",str(b['shots'])):b for b in result['blocks']}
    def rows(name):
        with (ROOT/'feature_bank/delivery/new_bank_v3'/name).open(encoding='utf8',newline='') as f:
            return [r for r in csv.DictReader(f) if r['run_id']=='personal_temporal_unibo_v1']
    family=rows('feature_family_results.csv');errors=rows('error_complementarity.csv');ablations=rows('ablation_full_bank.csv')
    assert len(family)==924 and len(errors)==1092 and len(ablations)==168
    with np.load(ROOT/'benchmarks/new_bank_v3/personal_temporal_unibo_v1/readouts.npz',allow_pickle=False) as arrays:
        for row in family:
            b=blocks[row['subject'],row['session/domain'],row['calibration_budget']];arm=row['feature_family']
            q=arrays[b['key']+'_'+arm];w=np.array(b['weights']);y=np.array(b['labels'])
            s=metrics(y,q,w,['neutral','index_pinch','fist','open_hand'])
            for k in ('accuracy','macro_f1','log_loss','brier'):
                assert abs(float(row[k])-s[k])<1e-12
            notes=json.loads(row['metadata_notes_json'])
            assert notes['actual_additional_temporal_calibration_trials'][arm]==b['predictive_calibration_cost'][arm]
            assert notes['oracle_complete_boundaries']
        for row in errors:
            b=blocks[row['subject'],row['session/domain'],row['calibration_budget']]
            w=np.array(b['weights']);w/=w.sum();y=np.array(b['labels'])
            ea=arrays[b['key']+'_'+row['family_a']].argmax(1)!=y
            eb=arrays[b['key']+'_'+row['family_b']].argmax(1)!=y
            for mask,field in ((~ea&eb,'a_correct_b_wrong_probability'),(ea&~eb,'a_wrong_b_correct_probability')):
                assert abs(float(row[field])-w[mask].sum())<1e-12
            assert abs(float(row['disagreement_rate'])-w[ea!=eb].sum())<1e-12
        for row in ablations:
            b=blocks[row['subject'],row['session/domain'],row['calibration_budget']]
            assert int(row['full_target_calibration_trials_per_user'])==20+4*b['shots']
            assert 'not renormalization' in json.loads(row['metadata_notes_json'])['branch_ablation']
