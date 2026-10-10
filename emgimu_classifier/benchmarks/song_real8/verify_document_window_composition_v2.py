"""Read-only V2 audit with sklearn dtype-preserving scaler parameter casts.

The frozen V1 verifier assumed float64 scaler parameters during float32
operations. The installed sklearn casts parameters BEFORE each operation.
Native experiment, checkpoint, probabilities and original verifier stay frozen."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from benchmarks.song_real8_study import load_session
from benchmarks.song_real8.song_personal_session_workflow_v1 import windows
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_window_composition_v1 import (
    GROUPS, CLASSES, TYPES, FAMILY_GROUPS, load_document_workflow)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    protocol=HERE/'DOCUMENT_WINDOW_COMPOSITION_V1_PROTOCOL.json'
    result=HERE/'DOCUMENT_WINDOW_COMPOSITION_V1_RESULTS.json'
    p=json.loads(protocol.read_text(encoding='utf8'));r=json.loads(result.read_text(encoding='utf8'))
    assert r['protocol_sha256']==sha(protocol)
    for name,digest in {**p['source_sha256'],**p['artifact_sha256'],**r['artifact_sha256']}.items():
        assert sha(ROOT/name)==digest,name
    out=HERE/'document_window_composition_v1'
    w=load_document_workflow(out/'source_bank.pkl',out/'policy.json',
        HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    for g,types in zip(GROUPS,TYPES): assert tuple(type(f) for f in w.bank.families_[g])==types
    native=load_session(Path(p['source_folder'])/'2026-09-18_S04','S04',filter_mode='causal')
    assert native['audit']['sha256']==p['hdf5_sha256']['S04']
    native['batch']=FeatureBatch(native['batch'].emg,250.)
    b,ids,offsets=windows(native,r['evaluation_ids'])
    q=np.load(out/'readouts.npz',allow_pickle=False)
    before=pickle.dumps(w);provider_error=0.;fusion_error=0.
    for g in GROUPS:
        columns=[]
        for f in w.bank.families_[g]:
            values=f.transform(b)
            columns.append(np.stack([values[ids==t].mean(0) for t in r['evaluation_ids']]))
        scaler,model=w.bank.models_[g]
        # Direct classifier matrix product, independent of bank.predict_providers.
        x=np.concatenate(columns,axis=1)
        z=x.copy();z-=scaler.mean_.astype(z.dtype);z/=scaler.scale_.astype(z.dtype)
        logits=z@model.coef_.T+model.intercept_
        logits-=logits.max(1,keepdims=True)
        raw=np.exp(logits);raw/=raw.sum(1,keepdims=True)
        adjusted=np.log(np.maximum(raw,1e-15))/w.bank.temperatures_[g]
        adjusted-=adjusted.max(1,keepdims=True)
        direct=np.exp(adjusted);direct/=direct.sum(1,keepdims=True)
        provider_error=max(provider_error,float(abs(direct-q['provider_'+g]).max()))
    cells={(c['shots'],c['arm']):c for c in r['cells']}
    assert len(cells)==96
    y=np.array([CLASSES.index(c) for c in r['evaluation_labels']])
    for (shots,arm),c in cells.items():
        prob=q[f'shots{shots}_{arm}'];prediction=prob.argmax(1)
        assert prob.shape==(124,4) and np.isfinite(prob).all()
        np.testing.assert_allclose(prob.sum(1),1.,rtol=0,atol=1e-12)
        np.testing.assert_allclose([c['accuracy'],c['log_loss'],c['brier']],
            [(prediction==y).mean(),-np.log(np.maximum(prob[np.arange(124),y],1e-15)).mean(),
             np.mean((prob-np.eye(4)[y])**2)],rtol=0,atol=1e-14)
        f1=[]
        for i,label in enumerate(CLASSES):
            tp=np.sum((prediction==i)&(y==i));fn=np.sum((prediction!=i)&(y==i));fp=np.sum((prediction==i)&(y!=i))
            f1.append(2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0.)
            assert c['recall'][label]==tp/(tp+fn)
        np.testing.assert_allclose(c['macro_f1'],np.mean(f1),rtol=0,atol=1e-14)
        weight_key=f'shots{shots}_{arm}_weights'
        if weight_key in q:
            active=[g for g in GROUPS if f'shots{shots}_{arm}_provider_{g}' in q]
            if arm.startswith('minus_provider_'): assert arm[len('minus_provider_'):] not in active
            if arm.startswith('minus_family_'): assert not set(FAMILY_GROUPS[arm[len('minus_family_'):]])&set(active)
            weights=q[weight_key];np.testing.assert_allclose(weights.sum(1),1.,rtol=0,atol=1e-14)
            direct=sum(weights[:,j,None]*q[f'shots{shots}_{arm}_provider_{g}'] for j,g in enumerate(active))
            fusion_error=max(fusion_error,float(abs(direct-prob).max()))
            assert int(q[f'shots{shots}_{arm}_rejected'].sum())==c['unknown_trials']
        expected=4*shots if arm not in ('new_population','new_uniform') else 0
        assert c['current_calibration_trials']==expected
    with (out/'predictions.csv').open(encoding='utf8',newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==11904
    for row in rows:
        prob=q[f'shots{row["shots"]}_{row["arm"]}'][r['evaluation_ids'].index(row['trial_id'])]
        np.testing.assert_array_equal(prob,np.array([float(row['p_'+g]) for g in CLASSES]))
    previous=json.loads((HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    assert previous['evaluation_ids']==r['evaluation_ids'] and previous['evaluation_labels']==r['evaluation_labels']
    assert previous['personal_calibration_ids']==r['personal_calibration_ids']
    assert set(w.bank.policy_.source_trials).isdisjoint(r['evaluation_ids']+r['personal_calibration_ids']+r['reserved_current_ids'])
    old=np.load(HERE/'song_extended_window_v1/readouts.npz',allow_pickle=False)
    old_bank=pickle.loads((HERE/'song_extended_window_v1/source_bank.pkl').read_bytes())
    parameter_error=0.
    for g in GROUPS:
        for first,second in [(w.bank.models_[g][0].mean_,old_bank.models_[g][0].mean_),
                (w.bank.models_[g][0].scale_,old_bank.models_[g][0].scale_),
                (w.bank.models_[g][1].coef_,old_bank.models_[g][1].coef_),
                (w.bank.models_[g][1].intercept_,old_bank.models_[g][1].intercept_)]:
            parameter_error=max(parameter_error,float(abs(first-second).max()))
        assert w.bank.temperatures_[g]==old_bank.temperatures_[g]
    np.testing.assert_array_equal(w.bank.policy_.population,old_bank.policy_.population)
    comparisons=dict(new_reliability='seven_reliability',new_F8='seven_F8',new_full='seven_full_structural',
        full_minus_F7='seven_minus_F7',full_minus_F8='seven_minus_F8',full_minus_F9='seven_minus_F9',
        new_population='seven_population',new_uniform='seven_uniform')
    comparisons.update({'minus_provider_'+g:'full_minus_'+g for g in GROUPS})
    for shots in p['current_shots']:
        for arm,old_arm in comparisons.items():
            np.testing.assert_array_equal(q[f'shots{shots}_{arm}'],old[f'shots{shots}_{old_arm}_probabilities'])
        for arm,old_arm in [('old_reliability','seven_reliability'),('old_full','seven_full_structural')]:
            np.testing.assert_array_equal(q[f'shots{shots}_{arm}'],old[f'shots{shots}_{old_arm}_probabilities'])
        assert r['profiles'][str(shots)]['calibration_ids']==previous['profiles'][str(shots)]['calibration_ids'] if shots else not r['profiles']['0']['calibration_ids']
    assert max(provider_error,fusion_error)<1e-12 and parameter_error==0. and pickle.dumps(w)==before
    return dict(schema='document_window_composition_v1_acceptance',verification_version=2,
        verifier_sha256=sha(Path(__file__)),native_runner_unchanged=True,
        supersedes_v1_scaler_parameter_cast_assumption=True,protocol_sha256=sha(protocol),
        result_sha256=sha(result),source_bank_sha256=sha(out/'source_bank.pkl'),policy_sha256=sha(out/'policy.json'),
        cells=96,predictions=11904,held_out_trials=124,provider_count=7,
        maximum_provider_probability_error=provider_error,maximum_fusion_probability_error=fusion_error,
        existing_source_classifier_parameter_max_error=parameter_error,matched_existing_probability_cells=4*len(comparisons),
        added_joint_F2_family_removal_cells=4,independent_rebuild_is_not_a_new_accuracy_gain=True,
        exact_implementations_verified=True,source_calibration_evaluation_disjoint=True,
        old_control_probabilities_preserved=True,primary_guards=r['primary_guards'],primary_pass=r['primary_pass'],
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=r['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    evidence=check()
    if args.write:
        target=ROOT/'feature_bank/DOCUMENT_WINDOW_COMPOSITION_V1_ACCEPTANCE.json'
        with target.open('x',encoding='utf8',newline='\n') as stream:json.dump(evidence,stream,indent=2);stream.write('\n')
    print(json.dumps(evidence,separators=(',',':')))
