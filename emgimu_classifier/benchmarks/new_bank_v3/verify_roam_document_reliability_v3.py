"""Independent native calibration, source CV and cached target probability audit.

Never calls new reliability calculations/composition, model.predict or fitting.
Frozen representation transforms and inherited verified target readouts are used.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from benchmarks.new_bank_v3.roam_document_reliability_v3 import ROOT,HERE,PROTOCOL,SOURCE,TARGET,OUT,PREVIOUS,INPUT,SOURCE_RECEIPT,TARGET_RECEIPT,ARMS
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha,write,build,merge,CLASSES,GROUPS,ARCHIVE
from benchmarks.new_bank_v3.roam_source_fusion_v1 import source_data
from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import manual_source
from benchmarks.new_bank_v3.verify_roam_source_fusion_v1 import metric
from benchmarks.new_bank_v3.verify_roam_native_continuous_v2 import close
from benchmarks.new_bank_v3.verify_roam_precision_transition_v2 import same
from benchmarks.song_real8.verify_integrated_decision_v1 import routing_oracle


def read(p):return json.loads(p.read_text(encoding='utf8'))


def checked():
    p=read(PROTOCOL)
    for rel,h in {**p['source_sha256'],**p['source_artifact_sha256']}.items():assert sha(ROOT/rel)==h,rel
    return p


def features_oracle(bank,profile):
    b,ids,_,_=native_windows(profile.calibration,bank.sensor_contract_[1],round(.04*bank.sensor_contract_[0]))
    axis=tuple(np.unique(ids));labels=dict(profile.calibration_labels);y=np.array([labels[t] for t in axis]);features={}
    for g in GROUPS:
        columns=[]
        for family in bank.families_[g]:
            values=family.transform(b);columns.append(np.stack([values[ids==t].mean(0) for t in axis]))
        x=np.concatenate(columns,axis=1);scaler,_=bank.models_[g]
        x-=scaler.mean_.astype(x.dtype);x/=scaler.scale_.astype(x.dtype)
        features[g]=(x,y,axis)
    return features


def reliability_oracle(calibration,population,n0,tau):
    logs=[];ids=None
    for g in GROUPS:
        x,y,axis=calibration[g];x=np.asarray(x,float)
        assert len(axis)==len(set(axis)) and set(y)==set(CLASSES)
        assert ids is None or axis==ids;ids=axis
        means=[np.mean(x[y==c],axis=0) for c in CLASSES]
        between=sum(float(np.sqrt(np.square(means[i]-means[j]).sum())) for i in range(3) for j in range(i+1,3))/3
        within=sum(float(np.sqrt(np.square(x[y==c]-means[i]).sum(1)).mean()) for i,c in enumerate(CLASSES))/3
        logs.append(np.log(between/(within+1e-10)+1e-10)/tau)
    v=np.exp(np.asarray(logs)-max(logs));v/=v.sum();alpha=n0/(n0+len(ids))
    return alpha*np.asarray(population)+(1-alpha)*v


def hierarchy(long,current,population,n0,tau):
    w=reliability_oracle(long,population,n0,tau)
    if current is not None:
        assert set(next(iter(long.values()))[2]).isdisjoint(next(iter(current.values()))[2])
        w=reliability_oracle(current,w,n0,tau)
    return w


def source_verify():
    p=checked();assert sha(ARCHIVE)==p['archive_sha256'];r=read(SOURCE);policy=read(OUT/'source/policy.json');inputs=read(INPUT)
    assert r['protocol_sha256']==policy['source_protocol_sha256']==sha(PROTOCOL)
    for rel,h in r['artifacts_sha256'].items():assert sha(ROOT/rel)==h
    assert policy['source_cache_sha256']==sha(OUT/'source/features_probabilities.npz')
    assert policy['inference_bank_id']==inputs['inference_bank_id'] and policy['classes']==list(CLASSES) and policy['providers']==list(GROUPS)
    cache=np.load(OUT/'source/features_probabilities.npz',allow_pickle=False);data=source_data()
    banks={i:pickle.loads((HERE/f'roam_source_fusion_v1/source/bank_fold{i}.pkl').read_bytes()) for i in range(3)}
    frozen={i:pickle.dumps(v) for i,v in banks.items()};max_feature_error=max_probability_error=0.;independent_calibrations={}
    assert len(r['blocks'])==54 and len(r['candidates'])==16
    for u in p['source_users']:
        rows=[b for b in r['blocks'] if b['user']==u];assert len(rows)==3 and {v['shots'] for v in rows}=={0,1,2}
        bank=banks[rows[0]['fold']];w=build(bank);user=f'ROAM_source_s{u}'
        assert all(f'/s{u}/' not in t for t in bank.policy_.source_trials)
        query=merge([data[u,c] for c in ('unsupported','reaching')]);qb,qi,_,_=native_windows(query.batch,40,8)
        axis,raw=manual_source(bank.families_,bank.models_,qb,qi,bank.temperatures_)
        order=[axis.index(t) for t in query.batch.trial_ids]
        for g,q in raw.items():max_probability_error=max(max_probability_error,float(abs(q[order]-cache[f'u{u}_raw_'+g]).max()))
        personal=w.load_profile(ROOT/rows[0]['personal_path'],user_id=user)
        for row in rows:
            assert tuple(row['query_ids'])==query.batch.trial_ids and row['query_labels']==[query.labels[t] for t in query.batch.trial_ids]
            session=None if row['session_path'] is None else w.load_profile(ROOT/row['session_path'],user_id=user,session_id='hanging',personal=personal)
            for profile,meta in ((personal,row['long']),(session,row['current'])):
                if profile is None:assert meta is None;continue
                assert set(query.batch.recording_ids).isdisjoint(profile.calibration.recording_ids)
                if meta['prefix'] in independent_calibrations:continue
                cal=features_oracle(bank,profile);independent_calibrations[meta['prefix']]=cal
                for g,(x,y,ids) in cal.items():
                    assert tuple(meta['trial_ids'])==ids and meta['labels']==y.tolist()
                    max_feature_error=max(max_feature_error,float(abs(x-cache[meta['prefix']+g]).max()))
    assert all(pickle.dumps(banks[i])==frozen[i] for i in banks) and max_feature_error<1e-12 and max_probability_error<1e-12
    rebuilt=[]
    for i,config in enumerate(p['source_candidates']):
        rows=[]
        for block in r['blocks']:
            long=independent_calibrations[block['long']['prefix']]
            current=None if block['current'] is None else independent_calibrations[block['current']['prefix']]
            weights=hierarchy(long,current,inputs['source_population'],config['n0'],config['temperature'])
            q=sum(weights[j]*cache[block['query_prefix']+'raw_'+g] for j,g in enumerate(GROUPS));q/=q.sum(1,keepdims=True)
            m=metric(block['query_labels'],q,np.zeros(len(q),bool));rows.append(dict(user=block['user'],**m))
        users=[dict(user=u,log_loss=float(np.mean([v['log_loss'] for v in rows if v['user']==u])),brier=float(np.mean([v['brier'] for v in rows if v['user']==u]))) for u in p['source_users']]
        summary=dict(index=i,config=config,per_user=users,mean_log_loss=float(np.mean([v['log_loss'] for v in users])),mean_brier=float(np.mean([v['brier'] for v in users])))
        same(summary,r['candidates'][i]);rebuilt.append(summary)
    selected=sorted(rebuilt,key=lambda c:(c['mean_log_loss'],c['mean_brier'],c['index']))[0]
    assert selected['index']==policy['selected_candidate']==r['selected_candidate'] and selected['config']==r['selected_config']
    assert policy['n0']==selected['config']['n0'] and policy['temperature']==selected['config']['temperature']
    assert set(policy['source_query_recording_ids'])=={d.receipt['member'] for d in data.values()}
    assert len({t for b in r['blocks'] for t in b['query_ids']})==324 and sum(len(b['query_ids']) for b in r['blocks'])==972
    assert not any(r[k] for k in ('existing_classifiers_refitted','target_arrays_read','source_scores_are_unbiased'))
    return dict(schema='roam_document_reliability_v3_source_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        policy_sha256=sha(OUT/'source/policy.json'),verifier_sha256=sha(Path(__file__)),selected_config=selected['config'],
        source_query_trials=324,source_budget_rows=972,source_users=18,candidates_verified=16,
        maximum_independent_calibration_feature_error=max_feature_error,maximum_independent_probability_error=max_probability_error,
        independent_exact_hierarchy_and_source_selection=True,target_arrays_read=False,classifiers_refitted=False,
        source_scores_are_unbiased=False,read_only_no_fitting=True,native_experiment_rerun=False,scope=r['scope'])


def target_verify():
    p=checked();r=read(TARGET);s=read(SOURCE);receipt=read(SOURCE_RECEIPT);policy=read(OUT/'source/policy.json')
    assert r['protocol_sha256']==receipt['protocol_sha256']==sha(PROTOCOL)
    assert r['source_result_sha256']==receipt['source_result_sha256']==sha(SOURCE)
    assert r['source_acceptance_sha256']==sha(SOURCE_RECEIPT) and r['policy_sha256']==receipt['policy_sha256']==sha(OUT/'source/policy.json')
    assert receipt['verifier_sha256']==sha(Path(__file__))
    for rel,h in {**p['target_artifact_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/rel)==h,rel
    inherited_receipt=read(ROOT/'feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json')
    assert inherited_receipt['result_sha256']==sha(HERE/'ROAM_NATIVE_JOINT_V1_RESULTS.json')
    inherited=np.load(HERE/'roam_native_joint_v1/readouts.npz',allow_pickle=False);saved=np.load(OUT/'target/readouts.npz',allow_pickle=False)
    bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes());w=build(bank);before=pickle.dumps(w)
    max_error=0.;expected={};assert len(r['blocks'])==30 and len(r['cells'])==210 and len(r['aggregates'])==63
    for row in r['blocks']:
        u=row['user'];user=f'ROAM_s{u}';personal=w.load_profile(ROOT/row['personal_path'],user_id=user)
        session=None if row['session_path'] is None else w.load_profile(ROOT/row['session_path'],user_id=user,session_id='hanging',personal=personal)
        for profile in (personal,session):
            if profile is not None:
                assert set(row['query_ids']).isdisjoint(profile.calibration.trial_ids)
                assert set(row['query_recording_ids']).isdisjoint(profile.calibration.recording_ids)
                assert set(policy['source_query_recording_ids']).isdisjoint(profile.calibration.recording_ids)
        long=features_oracle(bank,personal);current=None if session is None else features_oracle(bank,session)
        weights=hierarchy(long,current,policy['population'],policy['n0'],policy['temperature'])
        np.testing.assert_allclose(weights,row['weights'],rtol=0,atol=1e-12)
        routing=np.ones(7)
        if session is not None:
            lp=personal.window.decision;sp=session.window.decision
            routing=routing_oracle({g:lp.base.anchors[g].prototypes_ for g in GROUPS},
                {g:sp.base.anchors[g]['local'].prototypes_ for g in GROUPS},lp.spd_prototypes,sp.spd_local,lp,sp)
        routed=weights*routing;routed/=routed.sum();np.testing.assert_allclose(routed,row['routed_weights'],atol=1e-12,rtol=0)
        assert row['long_calibration_trials']==6 and row['current_calibration_trials']==3*row['shots']
        prefix=f'u{u}_s{row["shots"]}_'
        raw={g:inherited[f'u{u}_source_'+g] for g in GROUPS};heads={g:inherited[prefix+'decision_'+g] for g in GROUPS}
        reliability=sum(weights[i]*raw[g] for i,g in enumerate(GROUPS));reliability/=reliability.sum(1,keepdims=True)
        window=sum(routed[i]*heads[g] for i,g in enumerate(GROUPS));window/=window.sum(1,keepdims=True)
        joint=.75*window+.125*inherited[prefix+'DTW_blended']+.125*inherited[prefix+'signature_blended'];joint/=joint.sum(1,keepdims=True)
        values={name:inherited[prefix+key] for name,key in dict(source_window='population',legacy_reliability='window_reliability',legacy_window='window_full',legacy_joint='joint_full').items()}
        values.update(document_reliability=reliability,document_window=window,document_joint=joint)
        for arm,q in values.items():max_error=max(max_error,float(abs(q-saved[prefix+arm]).max()));expected[u,row['shots'],arm]=q
    assert max_error<1e-12 and pickle.dumps(w)==before and len(expected)==210 and len(saved.files)==210
    for cell in r['cells']:
        row=next(b for b in r['blocks'] if (b['user'],b['shots'])==(cell['user'],cell['shots']))
        q=expected[cell['user'],cell['shots'],cell['arm']]
        same({k:v for k,v in cell.items() if k not in ('user','shots','arm','target_calibration_trials')},metric(row['query_labels'],q,np.zeros(len(q),bool)))
        assert cell['target_calibration_trials']==(0 if cell['arm']=='source_window' else 6+3*cell['shots'])
    for cell in r['aggregates']:
        users=range(19,24) if cell['phase']=='validation' else range(24,29) if cell['phase']=='descriptive_final' else range(19,29)
        y=[c for b in r['blocks'] if b['user'] in users and b['shots']==cell['shots'] for c in b['query_labels']]
        q=np.concatenate([expected[u,cell['shots'],cell['arm']] for u in users])
        same({k:v for k,v in cell.items() if k not in ('phase','shots','arm')},metric(y,q,np.zeros(len(q),bool)))
    ag={(v['phase'],v['shots'],v['arm']):v for v in r['aggregates']};cu={(v['user'],v['shots'],v['arm']):v for v in r['cells']}
    full=ag['validation',2,'document_joint'];guards={};wins={}
    for arm in ('legacy_joint','source_window'):
        base=ag['validation',2,arm];wins[arm]=sum(cu[u,2,'document_joint']['log_loss']<cu[u,2,arm]['log_loss'] for u in range(19,24))
        guards[arm]=dict(lower_log_loss=full['log_loss']<base['log_loss'],lower_brier=full['brier']<base['brier'],
            nonworse_macro_f1=full['macro_f1']>=base['macro_f1'],nonworse_all_recalls=all(full['recall'][c]>=base['recall'][c] for c in CLASSES),at_least3_user_loss_wins=wins[arm]>=3)
    assert guards==r['primary_guards'] and wins==r['validation_user_loss_wins'] and all(all(v.values()) for v in guards.values())==r['primary_pass']
    assert r['source_policy_committed_before_target'] and not r['target_fitted_or_selected'] and not r['target_query_feature_inference_repeated']
    assert not r['default_promoted'] and not r['completion_proven']
    return dict(schema='roam_document_reliability_v3_acceptance',protocol_sha256=sha(PROTOCOL),source_result_sha256=sha(SOURCE),
        source_acceptance_sha256=sha(SOURCE_RECEIPT),target_result_sha256=sha(TARGET),policy_sha256=sha(OUT/'source/policy.json'),
        verifier_sha256=sha(Path(__file__)),selected_config=s['selected_config'],independent_query_trials=180,budget_blocks=30,
        cells=210,prediction_rows=3780,maximum_independent_probability_error=max_error,independent_calibration_hierarchy_routing_composition_and_metrics=True,
        primary_guards=guards,primary_pass=r['primary_pass'],validation_user_loss_wins=wins,
        two_shot_summary=[v for v in r['aggregates'] if v['shots']==2],target_query_feature_inference_repeated=False,
        target_fitted_or_selected=False,source_policy_committed_before_target=True,read_only_no_fitting=True,native_experiment_rerun=False,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=r['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    result=source_verify() if args.source_only else target_verify()
    if args.write_receipt:write(SOURCE_RECEIPT if args.source_only else TARGET_RECEIPT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scope','two_shot_summary')}))
