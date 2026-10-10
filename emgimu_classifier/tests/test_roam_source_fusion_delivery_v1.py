"""Saved source-only policy and target result provenance; no native reruns."""
import hashlib
import json
from pathlib import Path
import numpy as np
from emgimu.feature_bank.source_probability_fusion_v1 import SourceProbabilityFusionV1

ROOT=Path(__file__).resolve().parents[1]


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_fusion_native_source_and_target_artifacts_match_frozen_protocol():
    p=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_PROTOCOL.json')
    s=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_SOURCE_RESULTS.json')
    t=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_TARGET_RESULTS.json')
    for name,digest in {**p['source_sha256'],**p['source_artifact_sha256'],**p['target_artifact_sha256'],
                        **s['artifacts_sha256'],**t['artifacts_sha256']}.items():assert sha(ROOT/name)==digest,name
    assert s['protocol_sha256']==t['protocol_sha256']==sha(ROOT/'benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_PROTOCOL.json')
    assert t['source_result_sha256']==sha(ROOT/'benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_SOURCE_RESULTS.json')
    assert len(p['native_source_recordings'])==72 and sum(len(r['cue_intervals']) for r in p['native_source_recordings'])==648
    assert s['source_budget_rows']==972 and s['source_query_trials']==324
    assert not s['target_probability_arrays_read'] and not s['existing_source_classifiers_refitted']


def test_source_policy_excludes_targets_and_keeps_all_budgets_as_paired_scenarios():
    s=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_SOURCE_RESULTS.json')
    t=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_TARGET_RESULTS.json')
    old=read('benchmarks/new_bank_v3/ROAM_NATIVE_JOINT_V1_RESULTS.json')
    policy=SourceProbabilityFusionV1.from_manifest(read('benchmarks/new_bank_v3/roam_source_fusion_v1/source/policy.json'))
    assert policy.policy_id==s['policy_id']==t['policy_id'] and len(policy.source_trial_ids)==324
    assert not set(policy.source_trial_ids)&{q for u in old['subjects'] for q in u['query_ids']}
    assert len(s['blocks'])==54 and len(t['cells'])==330 and t['predictions']==5940
    assert not t['target_models_profiles_or_policy_refitted'] and t['source_policy_committed_before_target_apply']
    assert s['diagnostic']['equal_user_weighting'] and not s['diagnostic']['source_training_loss_is_unbiased_evaluation']
    arrays=np.load(ROOT/'benchmarks/new_bank_v3/roam_source_fusion_v1/target/readouts.npz',allow_pickle=False)
    for cell in t['cells']:
        key=f'u{cell["user"]}_s{cell["shots"]}_{cell["arm"]}'
        q=arrays[key];unknown=arrays[key+'_rejected']
        assert q.shape==(18,3) and int(unknown.sum())==cell['unknown_trials']
        assert cell['unique_calibration_trials']==(0 if cell['arm']=='population' else 6+3*cell['shots'])
        if unknown.all():assert cell['accuracy']==0. and cell['macro_f1']==0.
    arrays.close()


def test_source_and_target_independent_acceptance_preserve_all_guards():
    s=read('feature_bank/ROAM_SOURCE_FUSION_V1_SOURCE_ACCEPTANCE.json')
    a=read('feature_bank/ROAM_SOURCE_FUSION_V1_ACCEPTANCE.json')
    t=read('benchmarks/new_bank_v3/ROAM_SOURCE_FUSION_V1_TARGET_RESULTS.json')
    assert s['full_source_replay_blocks']==54 and s['source_query_users_excluded_from_representation_fit']
    assert s['calibration_query_recordings_disjoint'] and s['read_only_no_fitting']
    for key in ('maximum_manual_classifier_error','maximum_independent_temporal_error','maximum_frozen_workflow_replay_error'):
        assert s[key]<1e-12
    assert s['simplex_stationarity_gap']<1e-6 and a['maximum_target_composition_error']<1e-12
    assert a['primary_guards']==t['primary_guards'] and a['primary_pass']==all(v for g in t['primary_guards'].values() for v in g.values())
    assert a['missing_provider_Unknown_scored_wrong'] and not a['target_refitted']
    for key in ('source_training_loss_is_unbiased_evaluation','default_promoted','physical_validation_proven','completion_proven'):
        assert not a[key]
