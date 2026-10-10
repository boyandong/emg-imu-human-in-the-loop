"""Independent read-only source replay, simplex KKT and target composition.

No representation/classifier/profile/temperature/weight fit or optimizer call.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.roam_source_fusion_v1 import (
    ROOT,HERE,PROTOCOL,INPUT,SOURCE,TARGET,OUT,OLD,PROVIDERS,OLD_NAMES,CLASSES,
    GROUPS,sha,source_data,merge,build)
from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import manual_source,temporal_oracle
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from emgimu.feature_bank.source_probability_fusion_v1 import SourceProbabilityFusionV1

SOURCE_RECEIPT=ROOT/'feature_bank/ROAM_SOURCE_FUSION_V1_SOURCE_ACCEPTANCE.json'
TARGET_RECEIPT=ROOT/'feature_bank/ROAM_SOURCE_FUSION_V1_ACCEPTANCE.json'


def checked_inputs():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in {**p['source_sha256'],**p['source_artifact_sha256']}.items():assert sha(ROOT/name)==digest,name
    return p


def source_verify():
    p=checked_inputs();r=json.loads(SOURCE.read_text(encoding='utf8'));inputs=json.loads(INPUT.read_text(encoding='utf8'))
    assert r['protocol_sha256']==sha(PROTOCOL)
    for name,digest in r['artifacts_sha256'].items():assert sha(ROOT/name)==digest,name
    policy=SourceProbabilityFusionV1.from_manifest(json.loads((OUT/'source/policy.json').read_text(encoding='utf8')))
    assert policy.policy_id==r['policy_id'] and policy.source_protocol_id==sha(PROTOCOL)
    assert policy.inference_bank_id==inputs['inference_bank_id'] and policy.providers==PROVIDERS
    data=source_data();assert len(data)==72 and sum(len(d.batch.trial_ids) for d in data.values())==648
    assert json.loads(json.dumps([d.receipt for d in data.values()]))==p['native_source_recordings']
    banks={};fold_by_user={};bank_before={}
    for i,fold in enumerate(inputs['source_folds']):
        bank=pickle.loads((OUT/f'source/bank_fold{i}.pkl').read_bytes());banks[i]=build(bank)
        bank_before[i]=pickle.dumps(banks[i]);families,models=pickle.loads((ROOT/fold['path']).read_bytes())
        assert tuple(bank.policy_.source_trials)==tuple(fold['fit_ids']) and bank.sensor_contract_==(200.,40,8)
        assert bank.temperatures_==inputs['source_temperatures']
        for g in GROUPS:
            s,m=bank.models_[g];rs,rm=models[g]
            for a in ('mean_','scale_','var_'):np.testing.assert_array_equal(getattr(s,a),getattr(rs,a))
            for a in ('coef_','intercept_','classes_'):np.testing.assert_array_equal(getattr(m,a),getattr(rm,a))
            assert all(set(v.native_source_trials_)==set(fold['fit_ids']) for v in bank.families_[g])
        for u in fold['held_users']:fold_by_user[u]=i
    arrays=np.load(OUT/'source/readouts.npz',allow_pickle=False);all_q={g:[] for g in PROVIDERS};labels=[];trials=[]
    source_error=temporal_error=replay_error=0.
    for u in range(1,19):
        w=banks[fold_by_user[u]];user=f'ROAM_source_s{u}';query=merge([data[u,c] for c in ('unsupported','reaching')])
        assert all(f'/s{u}/' not in t for t in w.window.bank.policy_.source_trials)
        qb,qi,qo,_=native_windows(query.batch,40,8)
        axis,raw=manual_source(w.window.bank.families_,w.window.bank.models_,qb,qi,w.window.bank.temperatures_)
        order=[axis.index(t) for t in query.batch.trial_ids]
        raw={g:q[order] for g,q in raw.items()}
        qsource=sum(w.window.bank.policy_.population[i]*raw[g] for i,g in enumerate(GROUPS))
        for shots in (0,1,2):
            block=next(b for b in r['blocks'] if b['user']==u and b['shots']==shots)
            assert tuple(block['query_ids'])==query.batch.trial_ids
            assert [query.labels[t] for t in query.batch.trial_ids]==block['query_labels']
            personal=w.load_profile(ROOT/block['personal_path'],user_id=user)
            session=None if block['session_path'] is None else w.load_profile(ROOT/block['session_path'],
                user_id=user,session_id='hanging',personal=personal)
            for profile,condition,count in [(personal,'resting',6),(session,'hanging',3*shots)]:
                if profile is None:continue
                assert len(profile.calibration.trial_ids)==count
                assert set(profile.calibration.recording_ids).isdisjoint(query.batch.recording_ids)
                original=data[u,condition].batch
                for t,x in zip(profile.calibration.trial_ids,profile.calibration.sequences):
                    np.testing.assert_array_equal(x,original.sequences[original.trial_ids.index(t)])
                assert profile.calibration_cost['counted_once'] and profile.calibration_cost['native_sample_rate_hz']==200.
            base=w.window.predict(qb,qi,window_offsets=qo,personal=personal.window,
                session=None if session is None else session.window,user_id=user,session_id='hanging',
                observed_channel_ids=query.batch.channel_ids,preprocessing_id=query.batch.preprocessing_id,
                use_anchor=False,use_session_routing=False)
            qreliable=base['probabilities'][order]
            replay=w.predict(query.batch,personal=personal,session=session,user_id=user,session_id='hanging')
            ld,ls=temporal_oracle(query.batch,personal.temporal)
            dtw,sig=ld,ls
            if session is not None:
                d,s=temporal_oracle(query.batch,session.temporal);dtw=.5*(ld+d);sig=.5*(ls+s)
            components=dict(source_window=qsource,reliability_window=qreliable,
                F7F8_window=replay['temporal']['arms']['base'],DTW=dtw,signature=sig)
            for g,q in components.items():
                stored=arrays[f'u{u}_s{shots}_{g}'];error=float(abs(q-stored).max())
                if g=='source_window':source_error=max(source_error,error)
                elif g in ('DTW','signature'):temporal_error=max(temporal_error,error)
                else:replay_error=max(replay_error,error)
                all_q[g].append(stored)
            labels.extend(block['query_labels']);trials.extend(block['query_ids'])
        if u%3==0:print(f'Independent source replay {u}/18 users',flush=True)
    for i,w in banks.items():assert pickle.dumps(w)==bank_before[i]
    all_q={g:np.concatenate(v) for g,v in all_q.items()}
    for g,h in zip(PROVIDERS,policy.source_probability_sha256):
        import hashlib
        assert hashlib.sha256(np.ascontiguousarray(all_q[g],dtype=float).tobytes()).hexdigest()==h
    assert len(trials)==972 and len(set(trials))==324 and set(policy.source_trial_ids)==set(trials)
    assert not set(trials)&set(inputs['forbidden_target_query_ids'])
    y=np.array([CLASSES.index(c) for c in labels]);truth=np.column_stack([all_q[g][np.arange(len(y)),y] for g in PROVIDERS])
    weights=np.array(policy.weights);reg=p['regularization'];pred=truth@weights
    loss=float(-np.log(np.maximum(pred,1e-15)).mean());objective=loss+reg*float(weights@weights)
    gradient=-(truth/np.maximum(pred,1e-15)[:,None]).mean(0)+2*reg*weights
    gap=float(weights@gradient-gradient.min())
    assert gap<1e-6 and abs(objective-r['diagnostic']['objective'])<1e-12
    assert abs(loss-r['diagnostic']['source_log_loss'])<1e-12
    assert max(source_error,temporal_error,replay_error)<1e-12
    assert not r['target_probability_arrays_read'] and not r['existing_source_classifiers_refitted']
    return dict(schema='roam_source_fusion_v1_source_acceptance',source_result_sha256=sha(SOURCE),
        protocol_sha256=sha(PROTOCOL),policy_sha256=sha(OUT/'source/policy.json'),verifier_sha256=sha(Path(__file__)),
        native_source_recordings=72,source_cue_intervals=648,independent_source_queries=324,source_budget_rows=972,
        full_source_replay_blocks=54,maximum_manual_classifier_error=source_error,
        maximum_independent_temporal_error=temporal_error,maximum_frozen_workflow_replay_error=replay_error,
        independently_verified_source_objective=objective,simplex_stationarity_gap=gap,
        source_query_users_excluded_from_representation_fit=True,calibration_query_recordings_disjoint=True,
        existing_source_classifiers_refitted=False,source_policy_training_loss_is_unbiased_evaluation=False,
        target_probability_arrays_read=False,read_only_no_fitting=True,scope=r['scope'])


def metric(labels,q,rejected):
    truth=np.array([CLASSES.index(c) for c in labels]);pred=q.argmax(1);pred[rejected]=3
    cm=np.zeros((4,4),int);np.add.at(cm,(truth,pred),1)
    denom=cm.sum(0)[:3]+cm.sum(1)[:3]
    f1=np.divide(2*np.diag(cm)[:3],denom,out=np.zeros(3),where=denom>0)
    return dict(trials=len(labels),accuracy=float(np.mean(truth==pred)),macro_f1=float(f1.mean()),
        log_loss=float(-np.log(np.maximum(q[np.arange(len(q)),truth],1e-15)).mean()),
        brier=float(np.square(q-np.eye(3)[truth]).sum()/(len(q)*3)),unknown_trials=int(rejected.sum()),
        recall={c:float(cm[i,i]/cm[i].sum()) for i,c in enumerate(CLASSES)},confusion_with_Unknown=cm.tolist())


def target_verify():
    p=checked_inputs();s=json.loads(SOURCE.read_text(encoding='utf8'));a=json.loads(SOURCE_RECEIPT.read_text(encoding='utf8'))
    r=json.loads(TARGET.read_text(encoding='utf8'));old=json.loads(OLD.read_text(encoding='utf8'))
    assert a['source_result_sha256']==sha(SOURCE) and a['verifier_sha256']==sha(Path(__file__))
    assert a['policy_sha256']==sha(OUT/'source/policy.json') and a['full_source_replay_blocks']==54
    for name,digest in {**p['target_artifact_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/name)==digest,name
    assert r['source_result_sha256']==sha(SOURCE) and r['protocol_sha256']==sha(PROTOCOL)
    policy=SourceProbabilityFusionV1.from_manifest(json.loads((OUT/'source/policy.json').read_text(encoding='utf8')))
    assert policy.policy_id==s['policy_id']==r['policy_id'] and policy.inference_bank_id==old['source_bank_id']
    saved=np.load(OUT/'target/readouts.npz',allow_pickle=False);original=np.load(HERE/'roam_native_joint_v1/readouts.npz',allow_pickle=False)
    expected={};maximum=0.;seen_rows=set();weights=np.array(policy.weights)
    for subject in old['subjects']:
        u=subject['user'];labels=subject['query_labels'];assert not set(subject['query_ids'])&set(policy.source_trial_ids)
        for shots in (0,1,2):
            key=f'u{u}_s{shots}_';components={g:original[key+n] for g,n in zip(PROVIDERS,OLD_NAMES)}
            arms={}
            for arm,omitted in [('source_selected',None),*[('minus_'+g,g) for g in PROVIDERS]]:
                keep=[i for i,g in enumerate(PROVIDERS) if g!=omitted];mass=weights[keep].sum()
                rejected=np.full(18,mass<=1e-15)
                q=np.full((18,3),1/3) if mass<=1e-15 else sum(weights[i]*components[PROVIDERS[i]] for i in keep)/mass
                q/=q.sum(1,keepdims=True);arms[arm]=(q,rejected)
            for arm,old_name in [('fixed_joint','joint_full'),('window_full','window_full'),('population','population'),('reliability','window_reliability')]:
                arms[arm]=(original[key+old_name],np.zeros(18,bool))
            arms['uniform_components']=(np.mean(list(components.values()),axis=0),np.zeros(18,bool))
            for arm,(q,rejected) in arms.items():
                stored=saved[key+arm];maximum=max(maximum,float(abs(q-stored).max()))
                np.testing.assert_array_equal(rejected,saved[key+arm+'_rejected'])
                expected[u,shots,arm]=(stored,rejected)
                cell=next(c for c in r['cells'] if (c['user'],c['shots'],c['arm'])==(u,shots,arm))
                for k,v in metric(labels,stored,rejected).items():
                    if isinstance(v,float):assert abs(v-cell[k])<1e-12
                    else:assert v==cell[k]
                assert cell['unique_calibration_trials']==(0 if arm=='population' else 6+3*shots)
    assert len(expected)==len(r['cells'])==330 and maximum<1e-12
    subjects={s['user']:s for s in old['subjects']}
    for cell in r['aggregates']:
        users=range(19,24) if cell['phase']=='validation' else range(24,29) if cell['phase']=='descriptive_final' else range(19,29)
        pairs=[expected[u,cell['shots'],cell['arm']] for u in users]
        q=np.concatenate([v[0] for v in pairs]);rejected=np.concatenate([v[1] for v in pairs])
        labels=[c for u in users for c in subjects[u]['query_labels']]
        for k,v in metric(labels,q,rejected).items():
            if isinstance(v,float):assert abs(v-cell[k])<1e-12
            else:assert v==cell[k]
    with (OUT/'target/predictions.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            u,shots=int(row['user']),int(row['shots']);arm=row['arm'];t=row['trial_id'];key=(u,shots,arm,t)
            assert key not in seen_rows;seen_rows.add(key);position=subjects[u]['query_ids'].index(t)
            q,rejected=expected[u,shots,arm]
            assert row['predicted']==('Unknown' if rejected[position] else CLASSES[q[position].argmax()])
            assert row['true_label']==subjects[u]['query_labels'][position]
            np.testing.assert_array_equal(q[position],[float(row['p_'+c]) for c in CLASSES])
    assert len(seen_rows)==r['predictions']==5940
    selected=next(c for c in r['aggregates'] if (c['phase'],c['shots'],c['arm'])==('validation',2,'source_selected'))
    guards={};wins={}
    for arm in ('fixed_joint','population'):
        base=next(c for c in r['aggregates'] if (c['phase'],c['shots'],c['arm'])==('validation',2,arm))
        wins[arm]=sum(metric(subjects[u]['query_labels'],*expected[u,2,'source_selected'])['log_loss']<
            metric(subjects[u]['query_labels'],*expected[u,2,arm])['log_loss'] for u in range(19,24))
        guards[arm]=dict(lower_log_loss=selected['log_loss']<base['log_loss'],lower_brier=selected['brier']<base['brier'],
            nonworse_macro_f1=selected['macro_f1']>=base['macro_f1'],nonworse_all_class_recall=all(selected['recall'][c]>=base['recall'][c] for c in CLASSES),
            at_least3_user_loss_wins=wins[arm]>=3)
    assert guards==r['primary_guards'] and wins==r['user_loss_wins']
    assert r['primary_pass']==all(v for g in guards.values() for v in g.values())
    assert not r['target_models_profiles_or_policy_refitted'] and not r['default_promoted'] and not r['completion_proven']
    return dict(schema='roam_source_fusion_v1_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),target_result_sha256=sha(TARGET),policy_sha256=sha(OUT/'source/policy.json'),
        verifier_sha256=sha(Path(__file__)),independent_source_queries=324,independent_target_queries=180,
        source_budget_rows=972,target_cells=330,target_predictions=5940,maximum_target_composition_error=maximum,
        source_simplex_stationarity_gap=a['simplex_stationarity_gap'],primary_guards=guards,primary_pass=r['primary_pass'],
        user_loss_wins=wins,source_policy_committed_before_target_apply=True,target_refitted=False,read_only_no_fitting=True,
        missing_provider_Unknown_scored_wrong=True,source_training_loss_is_unbiased_evaluation=False,
        previously_inspected_target_users=True,default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=r['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    receipt=source_verify() if args.source_only else target_verify()
    if args.write_receipt:
        with (SOURCE_RECEIPT if args.source_only else TARGET_RECEIPT).open('x',encoding='utf8',newline='\n') as stream:
            json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))
