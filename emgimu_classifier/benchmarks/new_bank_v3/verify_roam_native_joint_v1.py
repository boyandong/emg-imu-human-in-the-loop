"""Read-only native replay and independent probability/metric equations.

Never fits a representation, classifier, temperature, profile or fusion policy.
Reuses frozen feature transforms; their individual equations have separate
formula fixtures. This verifier checks this study's axes, composition and scope.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from benchmarks.new_bank_v3.roam_native_joint_v1 import (
    ROOT,HERE,PROTOCOL,RESULT,OUT,ARCHIVE,load,merge,build,CLASSES,GROUPS,ZERO_COST)
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def softmax(x):
    x=np.asarray(x,float);v=np.exp(x-x.max(1,keepdims=True));return v/v.sum(1,keepdims=True)


def manual_source(families,models,batch,ids,temperatures=None):
    axis=tuple(np.unique(ids));probabilities={}
    for group in GROUPS:
        blocks=[]
        for family in families[group]:
            values=family.transform(batch)
            blocks.append(np.stack([values[ids==t].mean(0) for t in axis]))
        x=np.concatenate(blocks,axis=1);scaler,model=models[group]
        # StandardScaler casts fitted parameters to the float32 input dtype.
        z=x.copy();z-=scaler.mean_.astype(z.dtype);z/=scaler.scale_.astype(z.dtype)
        q=softmax(z@model.coef_.T+model.intercept_)
        if temperatures is not None:q=softmax(np.log(np.maximum(q,1e-15))/temperatures[group])
        probabilities[group]=q
    return axis,probabilities


def metric(labels,q):
    truth=np.array([CLASSES.index(c) for c in labels]);pred=q.argmax(1)
    cm=np.zeros((3,3),int)
    np.add.at(cm,(truth,pred),1)
    f1=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(3,float),where=(cm.sum(0)+cm.sum(1))>0)
    return dict(trials=len(truth),accuracy=float(np.mean(truth==pred)),macro_f1=float(f1.mean()),
        log_loss=float(-np.log(np.maximum(q[np.arange(len(q)),truth],1e-15)).mean()),
        brier=float(np.square(q-np.eye(3)[truth]).sum()/(len(q)*3)),confusion=cm.tolist(),
        recall={c:float(cm[i,i]/cm[i].sum()) for i,c in enumerate(CLASSES)})


def dtw(a,b,band):
    table={(0,0):(0.,0)}
    for i in range(1,len(a)+1):
        for j in range(max(1,i-band),min(len(b),i+band)+1):
            cost,n=min([table.get(k,(np.inf,0)) for k in ((i-1,j-1),(i-1,j),(i,j-1))])
            table[i,j]=(cost+float(np.sqrt(np.square(a[i-1]-b[j-1]).sum())),n+1)
    cost,n=table[len(a),len(b)]
    return cost/n


def temporal_oracle(batch,profile):
    envelopes=np.array([[np.sqrt(np.square(part.astype(float)).mean(0)) for part in np.array_split(x,32)] for x in batch.sequences])
    paths=envelopes/(np.linalg.norm(envelopes,axis=2,keepdims=True)+1e-10)
    delta=np.diff(paths-paths[:,:1],axis=1);prefix=np.cumsum(delta,axis=1)-delta
    signature=np.concatenate([delta.sum(1),(np.einsum('ntc,ntd->ncd',prefix,delta)+
        .5*np.einsum('ntc,ntd->ncd',delta,delta)).reshape(len(paths),-1)],axis=1)
    distance=np.array([[dtw(x,t,3) for t in profile.templates] for x in paths])
    geometry=np.array([[dtw(a,b,3) for b in profile.templates] for a in profile.templates])
    temperature=geometry[np.triu_indices(3,1)].mean()
    dq=np.full_like(distance,1/3) if temperature<=1e-10 else softmax(-distance/temperature)
    z=(signature-profile.signature_mean)/profile.signature_scale
    dist=np.linalg.norm(z[:,None]-profile.signature_prototypes[None],axis=2)
    geometry=np.linalg.norm(profile.signature_prototypes[:,None]-profile.signature_prototypes[None],axis=2)
    temperature=geometry[np.triu_indices(3,1)].mean()
    sq=np.full_like(dist,1/3) if temperature<=1e-10 else softmax(-dist/temperature)
    return dq,sq


def verify():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));r=json.loads(RESULT.read_text(encoding='utf8'))
    assert sha(PROTOCOL)==r['protocol_sha256'] and sha(ARCHIVE)==p['archive_sha256']
    for name,digest in {**p['source_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/name)==digest,name
    data=load();assert json.loads(json.dumps([d.receipt for d in data.values()]))==p['native_recordings']
    assert len(data)==58 and sum(len(d.batch.trial_ids) for d in data.values())==522
    assert all(not d.receipt['excluded_intervals'] and not d.receipt['resampled'] for d in data.values())
    assert p['source_users']==list(range(1,19)) and p['validation_users']==list(range(19,24))
    assert p['descriptive_users']==list(range(24,29)) and p['current_shots']==[0,1,2]
    source_ids={t for u in range(1,19) for t in data[u,'resting'].batch.trial_ids}
    bank=pickle.loads((OUT/'source_bank.pkl').read_bytes());w=build(bank);before=pickle.dumps(w)
    assert set(bank.policy_.source_trials)==source_ids and bank.policy_.n0==4. and bank.policy_.temperature==.5
    assert bank.sensor_contract_==(200.,40,8) and bank.classes_==CLASSES and w.window.gate is None
    policy=json.loads((OUT/'policy.json').read_text(encoding='utf8'))
    assert policy['joint_contract_id']==w.contract_id and policy['bank_sha256']==sha(OUT/'source_bank.pkl')
    source_error=0.;oof={g:[] for g in GROUPS};oof_y=[];seen=set()
    for fold in r['source_oof']:
        families,models=pickle.loads((ROOT/fold['path']).read_bytes())
        held=fold['held_users'];train_ids=source_ids-set(fold['ids'])
        assert train_ids==set(fold['fit_ids']) and not seen.intersection(fold['ids']);seen.update(fold['ids'])
        for views in families.values():
            for family in views:assert set(family.native_source_trials_)==train_ids
        assert set(families['F2b'][0].source_trial_ids_)==train_ids
        held_data=merge([data[u,'resting'] for u in held]);batch,ids,_,_=native_windows(held_data.batch,40,40)
        axis,prob=manual_source(families,models,batch,ids)
        assert axis==tuple(fold['ids']) and [held_data.labels[t] for t in axis]==fold['labels']
        for g in GROUPS:
            error=float(np.max(np.abs(prob[g]-fold['raw'][g])));source_error=max(source_error,error)
            oof[g].append(softmax(np.log(np.maximum(prob[g],1e-15))/r['source_temperatures'][g]))
        oof_y.extend(fold['labels'])
    assert seen==source_ids and len(oof_y)==162
    losses=np.array([metric(oof_y,np.concatenate(oof[g]))['log_loss'] for g in GROUPS])
    np.testing.assert_allclose(losses,[r['source_oof_losses'][g] for g in GROUPS],rtol=0,atol=1e-12)
    population=np.exp(-losses+losses.min());population/=population.sum()
    np.testing.assert_allclose(population,bank.policy_.population,rtol=0,atol=1e-12)
    saved=np.load(OUT/'readouts.npz',allow_pickle=False);expected={};max_error=0.;temporal_error=0.;replay_error=0.
    for subject in r['subjects']:
        u=subject['user'];user=f'ROAM_s{u}';query=merge([data[u,c] for c in ('unsupported','reaching')])
        assert query.batch.trial_ids==tuple(subject['query_ids']) and [query.labels[t] for t in query.batch.trial_ids]==subject['query_labels']
        assert len(query.batch.trial_ids)==18 and not source_ids.intersection(query.batch.trial_ids)
        personal=w.load_profile(ROOT/subject['personal_path'],user_id=user)
        wanted=tuple(t for c in CLASSES for t in [t for t in data[u,'resting'].batch.trial_ids if data[u,'resting'].labels[t]==c][:2])
        assert personal.calibration.trial_ids==wanted and personal.profile_id==subject['personal_id']
        query_records=set(query.batch.recording_ids)
        assert query_records.isdisjoint(personal.calibration.recording_ids)
        qb,qi,qo,tails=native_windows(query.batch,40,8)
        assert tails==subject['query_window_tail_samples'] and all(t<8 for t in tails)
        axis,raw=manual_source(bank.families_,bank.models_,qb,qi,bank.temperatures_)
        order=[axis.index(t) for t in query.batch.trial_ids];raw={g:q[order] for g,q in raw.items()}
        for g,q in raw.items():source_error=max(source_error,float(np.max(abs(q-saved[f'u{u}_source_{g}']))))
        ld,ls=temporal_oracle(query.batch,personal.temporal)
        for shots in (0,1,2):
            prefix=f'u{u}_s{shots}_';budget=subject['budgets'][str(shots)];session=None
            if shots:
                session=w.load_profile(ROOT/budget['session_path'],user_id=user,session_id='hanging',personal=personal)
                wanted=tuple(t for c in CLASSES for t in [t for t in data[u,'hanging'].batch.trial_ids if data[u,'hanging'].labels[t]==c][:shots])
                assert session.calibration.trial_ids==wanted and query_records.isdisjoint(session.calibration.recording_ids)
            for profile in (personal,session):
                if profile is None:continue
                assert not source_ids.intersection(profile.calibration.trial_ids)
                cost=profile.calibration_cost
                assert cost['native_signal_samples']==sum(map(len,profile.calibration.sequences))
                assert cost['native_signal_seconds']==cost['native_signal_samples']/200 and cost['counted_once'] and not cost['resampled']
                assert cost['unique_native_calibration_trials']==(6 if profile is personal else 3*shots)
                original=data[u,'resting' if profile is personal else 'hanging'].batch
                for trial,signal in zip(profile.calibration.trial_ids,profile.calibration.sequences):
                    np.testing.assert_array_equal(signal,original.sequences[original.trial_ids.index(trial)])
            dtwq,sigq=(ld,ls) if session is None else temporal_oracle(query.batch,session.temporal)
            if session is not None:dtwq=.5*(ld+dtwq);sigq=.5*(ls+sigq)
            for name,q in [('DTW_blended',dtwq),('signature_blended',sigq),('DTW_long',ld),('signature_long',ls)]:
                temporal_error=max(temporal_error,float(np.max(abs(q-saved[prefix+name]))))
            common=dict(personal=personal,user_id=user,session_id='hanging',session=session)
            replay=w.predict(query.batch,**common)
            replay_error=max(replay_error,float(np.max(abs(replay['probabilities']-saved[prefix+'joint_full']))))
            win=replay['window'];reorder=[win['trial_ids'].index(t) for t in query.batch.trial_ids]
            weights=np.array(win['weights']);np.testing.assert_array_equal(weights,saved[prefix+'window_weights'])
            providers={g:win['decision_provider_probabilities'][g][reorder] for g in GROUPS}
            for g,q in providers.items():np.testing.assert_array_equal(q,saved[prefix+'decision_'+g])
            qwin=sum(weights[i]*providers[g] for i,g in enumerate(GROUPS));qwin/=qwin.sum(1,keepdims=True)
            other={}
            for name,anchor,routing in [('window_reliability',False,False),('window_F7',True,False),('window_F8',False,True)]:
                z=w.window.predict(qb,qi,window_offsets=qo,personal=personal.window,session=None if session is None else session.window,
                    user_id=user,session_id='hanging',observed_channel_ids=query.batch.channel_ids,
                    preprocessing_id=query.batch.preprocessing_id,use_anchor=anchor,use_session_routing=routing)
                other[name]=z['probabilities'][reorder]
            arms=dict(population=sum(population[i]*raw[g] for i,g in enumerate(GROUPS)),uniform=np.mean(list(raw.values()),axis=0),
                single_F0=raw['F0'],single_CSP=raw['F2b'],window_full=qwin,**other)
            arms.update(joint_full=.75*qwin+.125*dtwq+.125*sigq,joint_DTW=.75*qwin+.25*dtwq,
                joint_signature=.75*qwin+.25*sigq,joint_uniform=.75*qwin+.25/3,joint_long_templates=.75*qwin+.125*ld+.125*ls,
                joint_minus_window_F7=.75*other['window_F8']+.125*dtwq+.125*sigq,
                joint_minus_window_F8=.75*other['window_F7']+.125*dtwq+.125*sigq,joint_minus_temporal=qwin)
            removals={**{'minus_provider_'+g:(g,) for g in GROUPS},'minus_family_F0':('F0',),'minus_family_F1':('F1',),
                'minus_family_F2':('F2ac','F2b'),'minus_family_F3':('F3b',),'minus_family_F4':('F4abc',),'minus_family_F5':('F5window',)}
            for name,omitted in removals.items():
                keep=[i for i,g in enumerate(GROUPS) if g not in omitted]
                q=sum(weights[i]*providers[GROUPS[i]] for i in keep)/weights[keep].sum();q/=q.sum(1,keepdims=True)
                arms[name]=q if name=='minus_family_F5' else .75*q+.125*dtwq+.125*sigq
            assert set(arms)==set(budget['arms']) and len(arms)==29
            for arm,q in arms.items():
                stored=saved[prefix+arm];max_error=max(max_error,float(np.max(abs(q-stored))))
                expected[u,shots,arm]=stored
                cell=next(c for c in r['cells'] if c['user']==u and c['shots']==shots and c['arm']==arm)
                for k,v in metric(subject['query_labels'],stored).items():
                    if isinstance(v,float):assert abs(v-cell[k])<1e-12,(u,shots,arm,k)
                    else:assert v==cell[k],(u,shots,arm,k)
                assert cell['unique_calibration_trials']==(0 if arm in ZERO_COST else 6+3*shots)
        print(f'Native replay verified user {u-18}/10',flush=True)
    assert pickle.dumps(w)==before and len(expected)==len(r['cells'])==870
    subjects={s['user']:s for s in r['subjects']};seen_rows=set()
    with (OUT/'predictions.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            u,shots=int(row['user']),int(row['shots']);arm=row['arm'];t=row['trial_id'];key=(u,shots,arm,t)
            assert key not in seen_rows;seen_rows.add(key)
            pos=subjects[u]['query_ids'].index(t);q=expected[u,shots,arm][pos]
            assert row['true_label']==subjects[u]['query_labels'][pos] and row['predicted']==CLASSES[q.argmax()]
            np.testing.assert_array_equal(q,[float(row['p_'+c]) for c in CLASSES])
    assert len(seen_rows)==r['prediction_rows']==15660
    for cell in r['aggregates']:
        users=range(19,24) if cell['phase']=='validation' else range(24,29) if cell['phase']=='descriptive_final' else range(19,29)
        q=np.concatenate([expected[u,cell['shots'],cell['arm']] for u in users]);y=[c for u in users for c in subjects[u]['query_labels']]
        for k,v in metric(y,q).items():
            if isinstance(v,float):assert abs(v-cell[k])<1e-12,(cell['phase'],cell['shots'],cell['arm'],k)
            else:assert v==cell[k]
    base=next(c for c in r['aggregates'] if (c['phase'],c['shots'],c['arm'])==('validation',2,'window_full'))
    full=next(c for c in r['aggregates'] if (c['phase'],c['shots'],c['arm'])==('validation',2,'joint_full'))
    wins=sum(metric(subjects[u]['query_labels'],expected[u,2,'joint_full'])['log_loss']<
        metric(subjects[u]['query_labels'],expected[u,2,'window_full'])['log_loss'] for u in range(19,24))
    guards=dict(lower_log_loss=full['log_loss']<base['log_loss'],lower_brier=full['brier']<base['brier'],
        nonworse_macro_f1=full['macro_f1']>=base['macro_f1'],nonworse_all_class_recalls=all(full['recall'][c]>=base['recall'][c] for c in CLASSES),
        at_least3_of5_user_loss_wins=wins>=3)
    assert guards==r['primary_guards'] and all(guards.values())==r['primary_pass'] and wins==r['validation_user_loss_wins']
    assert max(source_error,max_error,temporal_error,replay_error)<1e-12
    return dict(schema='roam_native_joint_v1_acceptance',protocol_sha256=sha(PROTOCOL),result_sha256=sha(RESULT),
        verifier_sha256=sha(Path(__file__)),native_recordings=58,native_cue_intervals=522,independent_queries=180,
        native_cells=870,predictions=15660,read_only_no_fitting=True,source_user_folds_disjoint=True,
        calibration_query_recordings_disjoint=True,profiles_preserve_all_calibration_samples=True,
        maximum_manual_source_probability_error=source_error,maximum_composition_probability_error=max_error,
        maximum_independent_temporal_probability_error=temporal_error,maximum_saved_profile_replay_error=replay_error,
        primary_guards=guards,primary_pass=all(guards.values()),validation_user_loss_wins=wins,
        native_sample_rate_hz=200.,channels=8,classes=CLASSES,raw_quality_policy_available=False,
        resampled=False,previously_inspected_public_users=True,autonomous_segmentation_proven=False,
        physiological_boundaries_proven=False,physical_validation_proven=False,default_promoted=False,
        completion_proven=False,scope=r['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    receipt=verify()
    if args.write_receipt:
        with (ROOT/'feature_bank/ROAM_NATIVE_JOINT_V1_ACCEPTANCE.json').open('x',encoding='utf8',newline='\n') as stream:
            json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))
