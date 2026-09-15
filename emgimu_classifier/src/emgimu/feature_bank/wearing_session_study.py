"""Matched before/after wearing calibration with immutable personal profiles.

The native trial_1..4 domains are wearing conditions, not invented calendar days.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from .wearing_full_fusion import load
from .force_full_fusion import aggregate
from .force_nested_oof import temperature_probability
from .calibration import SessionSignature,late_fusion
from .session_anchor import SessionPrototypeAnchor
from .quality_observability import QualityObservabilityFamily
from .electrode_shift_study import _metrics
from .unibo_temporal_complementarity import complementarity
from .core_incremental_oof import write
from .screening import SEED

FAMILIES=('F0','F3_Ring')
METHODS=('population_uniform','population_session_context','anchor_long_uniform',
    'anchor_local_uniform','anchor_blended_uniform','population_plus_local_anchor')


def calibration_indices(labels,shots,user,domain):
    labels=np.asarray(labels)
    if shots<0:raise ValueError('Negative calibration budget')
    counts=[int(np.sum(labels==h)) for h in range(5)]
    if any(count<=shots for count in counts):
        raise ValueError('Budget must leave at least one evaluation trial per native class')
    rng=np.random.default_rng(SEED+user*10+int(domain[-1]))
    return np.array([i for h in range(5) for i in rng.permutation(np.flatnonzero(labels==h))[:shots]],dtype=int)


def anchor_probability(anchor,features):
    distances=anchor._distances(features)
    logits=-distances/max(anchor.similarity_scale_,1e-10)
    logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);return p/p.sum(1,keepdims=True)


def condition_predictions(population,features,labels,cal,ev,profiles,signatures):
    snapshots=pickle.dumps((profiles,signatures))
    uniform=np.full(len(FAMILIES),1/len(FAMILIES));context=uniform.copy()
    vectors={};beta={};anchor_probs={mode:{} for mode in ('long_term','local','blended')}
    for index,name in enumerate(FAMILIES):
        x=features[name]
        if len(cal):
            vector=signatures[name].from_session_calibration(x[cal],labels[cal])
            vectors[name]=vector
            # Identical fixed rule to the previously prespecified MANUS
            # session-context control; no tuning on wearing validation/final.
            context[index]*=np.clip((vector[5:10].mean()+1)/2,.05,1.)
        for mode in anchor_probs:
            anchor,b=profiles[name].from_calibration(x[cal] if len(cal) else None,
                labels[cal] if len(cal) else None,mode=mode)
            anchor_probs[mode][name]=anchor_probability(anchor,x[ev]);beta[f'{name}_{mode}']=b.tolist()
    context/=context.sum()
    available={name:p[ev] for name,p in population.items()}
    predictions={'population_uniform':late_fusion(available,FAMILIES,uniform),
        'population_session_context':late_fusion(available,FAMILIES,context)}
    for mode,probabilities in anchor_probs.items():
        predictions[f'anchor_{"long" if mode=="long_term" else mode}_uniform']=late_fusion(probabilities,FAMILIES,uniform)
    shots=len(cal)/5;alpha=shots/(shots+2)
    predictions['population_plus_local_anchor']=(1-alpha)*predictions['population_uniform']+alpha*predictions['anchor_local_uniform']
    if snapshots!=pickle.dumps((profiles,signatures)):
        raise AssertionError('Session calibration mutated long-term personal profile')
    return predictions,vectors,context,beta,anchor_probs['local']['F3_Ring']


def run(archive,parent,output,phase):
    if output.exists():raise FileExistsError(output)
    users=(15,16,17) if phase=='validation' else (18,19,20)
    manifest=json.loads((parent/'run_manifest.json').read_text())
    if manifest['phase']!=phase or not set(FAMILIES)<=set(manifest['families']):
        raise ValueError('Wrong frozen wearing provider package')
    original_splits=json.loads((parent/'split_trial_ids.json').read_text())
    old_states=pickle.loads((parent/'fitted_states.pkl').read_bytes())
    before=pickle.dumps(old_states)
    rows=[];pairs=[];splits=[];vectors=[];states={};arrays={};collected={}
    for user in users:
        print(f'[{users.index(user)+1}/3] user {user}: immutable source profiles; per-wearing-domain cal0/1',flush=True)
        source=load(archive,user,('training',));target=load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
        source_features={};target_features={};population={};profiles={};signatures={}
        for name in FAMILIES:
            family,scaler,model=old_states[(user,name)]
            a,sy,_,st=aggregate(family.transform(source.batch),source)
            b,y,_,trials=aggregate(family.transform(target.batch),target)
            if set(st)!=set(original_splits[str(user)]['train']) or set(trials)!=set(original_splits[str(user)]['test']):
                raise AssertionError('Frozen source/target trial coverage changed')
            a=scaler.transform(a);b=scaler.transform(b)
            source_features[name]=a;target_features[name]=b
            profiles[name]=SessionPrototypeAnchor().fit_long_term(a,sy)
            signatures[name]=SessionSignature().fit_long_term(a,sy)
            population[name]=temperature_probability(model.predict_proba(b),manifest['temperatures'][f'{user}_{name}'])
            arrays[f'{user}_features_{name}']=b;arrays[f'{user}_population_{name}']=population[name]
        domains=np.array([trial.split('/')[-2] for trial in trials])
        states[user]=(profiles,signatures)
        arrays[f'{user}_labels']=y;arrays[f'{user}_trials']=trials
        # Diagnostic fit uses source before-wearing data only, with unknown ADC
        # and pre-highpass availability explicitly masked. No quality gating.
        quality=QualityObservabilityFamily(ring_topology=True).fit(source.batch)
        source_q=np.mean(quality.transform(source.batch),axis=0)
        for domain in sorted(set(domains)):
            domain_indices=np.flatnonzero(domains==domain);dy=y[domain_indices]
            for shots in (0,1):
                local_cal=calibration_indices(dy,shots,user,domain)
                cal=domain_indices[local_cal];ev=domain_indices[~np.isin(np.arange(len(dy)),local_cal)]
                if set(trials[cal])&set(trials[ev]) or set(st)&set(trials[cal]) or set(st)&set(trials[ev]):
                    raise AssertionError('Source/calibration/evaluation trial leakage')
                p,signature,context,beta,ring_anchor=condition_predictions(population,target_features,y,cal,ev,profiles,signatures)
                splits.append(dict(user=user,domain=domain,shots=shots,train=st.tolist(),
                    calibration=trials[cal].tolist(),evaluation=trials[ev].tolist()))
                q_summary=None
                if len(cal):
                    selected=target.take(np.flatnonzero(np.isin(target.trials,trials[cal])))
                    local_q=np.mean(quality.transform(selected.batch),axis=0)
                    q_summary=dict(mean_absolute_quality_shift=float(np.mean(np.abs(local_q-source_q))),
                        availability=quality.availability_,interpretation='calibration/source observability difference; not hardware fault labels')
                vectors.append(dict(user=user,domain=domain,shots=shots,
                    session_signature={name:dict(vector=v.tolist(),names=signatures[name].feature_names) for name,v in signature.items()},
                    context_weights=context.tolist(),prototype_beta=beta,quality_summary=q_summary))
                for method,probability in p.items():
                    row=dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                        shots_per_class=shots,method=method,feature_bank='F0|F3_Ring_reference',
                        calibration_trials=len(cal),evaluation_trials=len(ev),supported=True,
                        **_metrics(y[ev],probability))
                    rows.append(row);collected.setdefault((domain,shots,method),[]).append(row)
                    arrays[f'{user}_{domain}_{shots}_{method}']=probability
                for a,b,pa,pb in (('population_uniform','population_session_context',p['population_uniform'],p['population_session_context']),
                        ('F3_Ring_population','F3_Ring_local_anchor',population['F3_Ring'][ev],ring_anchor)):
                    pairs.append(dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                        calibration_budget=shots,family_a=a,family_b=b,
                        scope='reference ring, not historical validated RLCS; same held-out trials',
                        **complementarity(y[ev],pa,pb,np.ones(len(ev)))))
            for unsupported in (2,5):
                rows.append(dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                    shots_per_class=unsupported,method='unsupported_budget',feature_bank='F0|F3_Ring_reference',
                    calibration_trials='',evaluation_trials='',supported=False,
                    reason='Only two native repetitions/class/wearing-domain; budget leaves no evaluation trial'))
    for (domain,shots,method),values in collected.items():
        mean={metric:float(np.mean([v[metric] for v in values])) for metric in ('macro_f1','accuracy','log_loss','brier','ece')}
        classes=[json.loads(v['per_class_f1_json']) for v in values]
        rows.append({**values[0],'subject':'ALL','calibration_trials':sum(v['calibration_trials'] for v in values),
            'evaluation_trials':sum(v['evaluation_trials'] for v in values),**mean,
            'per_class_f1_json':json.dumps({h:float(np.mean([v[h] for v in classes])) for h in classes[0]}),
            'aggregation':'mean individual-user scores; budgets have different excluded evaluation trials'})
    if before!=pickle.dumps(old_states):raise AssertionError('Target calibration changed frozen provider package')
    output.mkdir(parents=True)
    # Union schema preserves unsupported rows as missing measurements.
    import csv
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'calibration_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(output/'error_complementarity.csv',pairs)
    with (output/'fitted_profiles.pkl').open('wb') as handle:pickle.dump(states,handle)
    np.savez_compressed(output/'heldout_predictions.npz',**arrays)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'session_signatures.json').write_text(json.dumps(vectors,indent=2))
    (output/'run_manifest.json').write_text(json.dumps(dict(phase=phase,users=users,families=FAMILIES,
        parent_run=parent.name,parent_sha256={name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in
            ('fitted_states.pkl','run_manifest.json','split_trial_ids.json')},
        raw_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),classifier_fit=False,
        prototype_beta='source class count/(source class count+calibration class count)',
        context_rule='uniform source-provider weights times clipped (mean class cosine+1)/2; prespecified MANUS rule',
        population_anchor_alpha='shots/(shots+2), fixed budget rule',
        probability_calibration='reused before-wearing repetition-held-out OOF source temperatures',
        evidence_boundary='same-user before/after native wearing conditions, not calendar session/day labels; reference ring is not historical RLCS; unknown ADC/pre-highpass masked; no device accuracy claim'),indent=2))
    replay(output,parent)


def replay(output,parent=None):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if parent is not None:
        for name,digest in manifest['parent_sha256'].items():
            if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=digest:
                raise AssertionError('Frozen before-wearing source package changed')
    profiles=pickle.loads((output/'fitted_profiles.pkl').read_bytes())
    splits=json.loads((output/'split_trial_ids.json').read_text());checked=0;error=0.
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            user=split['user'];domain=split['domain'];shots=split['shots']
            trials=saved[f'{user}_trials'];y=saved[f'{user}_labels']
            cal=np.flatnonzero(np.isin(trials,split['calibration']));ev=np.flatnonzero(np.isin(trials,split['evaluation']))
            if set(split['train'])&set(split['calibration']) or set(split['calibration'])&set(split['evaluation']):
                raise AssertionError('Replayed source/calibration/evaluation overlap')
            features={name:saved[f'{user}_features_{name}'] for name in FAMILIES}
            population={name:saved[f'{user}_population_{name}'] for name in FAMILIES}
            p,_,_,_,_=condition_predictions(population,features,y,cal,ev,*profiles[user])
            for method,probability in p.items():
                reference=saved[f'{user}_{domain}_{shots}_{method}']
                error=max(error,float(np.max(np.abs(probability-reference))))
                np.testing.assert_allclose(probability,reference,atol=1e-12,rtol=0);checked+=1
    result=dict(status='ok',prediction_arrays_replayed=checked,maximum_absolute_probability_error=error,
        long_term_profiles_immutable=True,classifier_fit=False,parent_source_hashes_verified=parent is not None,
        scope='Source/profile and held-out prediction replay; no device-level inference claim')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def summarize(output):
    """Add mean user/domain scores; never pool different wearing calibrations."""
    import csv
    path=output/'calibration_curve.csv'
    with path.open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle));fields=list(rows[0])
    rows=[row for row in rows if row['condition']!='ALL']
    for shots in (0,1):
        for method in METHODS:
            selected=[row for row in rows if row['subject']!='ALL' and row['method']==method and int(row['shots_per_class'])==shots]
            scores={metric:float(np.mean([float(row[metric]) for row in selected])) for metric in ('macro_f1','accuracy','log_loss','brier','ece')}
            classes=[json.loads(row['per_class_f1_json']) for row in selected]
            rows.append({**selected[0],'subject':'ALL','condition':'ALL',**scores,
                'calibration_trials':sum(int(row['calibration_trials']) for row in selected),
                'evaluation_trials':sum(int(row['evaluation_trials']) for row in selected),
                'per_class_f1_json':json.dumps({h:float(np.mean([row[h] for row in classes])) for h in classes[0]}),
                'aggregation':'mean 12 individual user/wearing-domain scores; five cal trials per domain at one-shot'})
    unsupported=next(row for row in rows if row['supported']=='False')
    rows.extend({**unsupported,'subject':'ALL','condition':'ALL','shots_per_class':shots} for shots in (2,5))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'summary_rows':14,'scope':'mean user/domain; no calibration pooled across domains'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path)
    parser.add_argument('parent',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=('validation','final'),required=True)
    parser.add_argument('--replay',action='store_true');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.replay:replay(args.output,args.parent)
    elif args.summarize:summarize(args.output)
    else:
        run(args.archive,args.parent,args.output,args.phase)
        summarize(args.output)
