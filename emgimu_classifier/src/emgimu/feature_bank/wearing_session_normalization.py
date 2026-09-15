"""Native wearing study with source OOF-calibrated integrated session updates."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.electrode_shift import PATH_RE
from .session_pipeline import SessionCalibrationPipeline,FAMILIES,BRANCHES
from .wearing_full_fusion import load
from .wearing_session_study import calibration_indices
from .force_full_fusion import aggregate
from .force_nested_oof import fit_temperature,temperature_probability
from .calibration import late_fusion
from .electrode_shift_study import _metrics


def fuse(raw,temperatures,weights=None,session_present=False):
    uniform=np.ones(2)/2;result={}
    for branch in BRANCHES:
        effective=branch if session_present or branch in ('source_model','source_long_anchor') else 'source_long_anchor' if 'anchor' in branch else 'source_model'
        probabilities={name:temperature_probability(raw[branch][name],temperatures[f'{effective}_{name}']) for name in FAMILIES}
        result[branch]=late_fusion(probabilities,FAMILIES,uniform)
        if branch=='session_model':result['session_context']=late_fusion(probabilities,FAMILIES,uniform if weights is None else weights)
    return result


def run(archive,parent,output,phase):
    if output.exists():raise FileExistsError(output)
    users=(15,16,17) if phase=='validation' else (18,19,20)
    parent_manifest=json.loads((parent/'run_manifest.json').read_text())
    if parent_manifest['phase']!=phase:raise ValueError('Wrong raw source provider phase')
    old=pickle.loads((parent/'fitted_states.pkl').read_bytes())
    models={};sessions={};saved={};splits=[];descriptors=[];rows=[];temperatures={}
    for user in users:
        print(f'[{users.index(user)+1}/3] {user}: before-wearing OOF, integrated rest/scale/quality/profile updates',flush=True)
        source=load(archive,user,('training',));target=load(archive,user,('trial_1','trial_2','trial_3','trial_4'))
        reps=np.array([int(PATH_RE.fullmatch(trial)['rep']) for trial in source.trials]);rep_values=sorted(set(reps))
        _,sy,_,st=aggregate(np.zeros((len(source.labels),1)),source)
        oof={f'{branch}_{name}':np.zeros((len(st),5)) for branch in BRANCHES for name in FAMILIES}
        folds=[];fold_states={}
        for i,rep in enumerate(rep_values):
            cal_rep=rep_values[(i+1)%len(rep_values)]
            long=source.take(np.flatnonzero(~np.isin(reps,(rep,cal_rep))))
            cal=source.take(np.flatnonzero(reps==cal_rep));ev=source.take(np.flatnonzero(reps==rep))
            pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(long)
            session=pipeline.calibrate_session(cal)
            raw,y,trials=pipeline.predict(ev,session)
            index=np.array([np.flatnonzero(st==trial).item() for trial in trials]);np.testing.assert_array_equal(sy[index],y)
            for branch in BRANCHES:
                for name in FAMILIES:oof[f'{branch}_{name}'][index]=raw[branch][name]
            folds.append(dict(evaluation_repetition=int(rep),calibration_repetition=int(cal_rep),
                train=sorted(set(long.trials)),calibration=sorted(set(cal.trials)),evaluation=sorted(set(ev.trials))))
            fold_states[int(rep)]=(pipeline,session)
        t={key:fit_temperature(p,sy) for key,p in oof.items()}
        temperatures[str(user)]=t
        pipeline=SessionCalibrationPipeline(rest_label=2,ring_topology=True).fit_long_term(source)
        models[user]=(pipeline,fold_states)
        raw_population={}
        for name in FAMILIES:
            family,scaler,model=old[(user,name)]
            x,y,_,trials=aggregate(family.transform(target.batch),target)
            raw_population[name]=temperature_probability(model.predict_proba(scaler.transform(x)),parent_manifest['temperatures'][f'{user}_{name}'])
        raw_uniform=late_fusion(raw_population,FAMILIES,np.ones(2)/2)
        domains=np.array([trial.split('/')[-2] for trial in trials])
        for key,p in oof.items():saved[f'{user}_oof_{key}']=p
        saved[f'{user}_source_labels']=sy;saved[f'{user}_source_trials']=st
        for domain in sorted(set(domains)):
            idx=np.flatnonzero(domains==domain)
            for shots in (0,1):
                cal_idx=idx[calibration_indices(y[idx],shots,user,domain)]
                ev_idx=idx[~np.isin(idx,cal_idx)]
                cal=target.take(np.flatnonzero(np.isin(target.trials,trials[cal_idx]))) if shots else None
                ev=target.take(np.flatnonzero(np.isin(target.trials,trials[ev_idx])))
                session=pipeline.calibrate_session(cal) if cal is not None else None
                sessions[(user,domain,shots)]=session
                raw,ey,et=pipeline.predict(ev,session)
                np.testing.assert_array_equal(et,trials[ev_idx]);np.testing.assert_array_equal(ey,y[ev_idx])
                p=fuse(raw,t,session['context_weights'] if session is not None else None,shots>0)
                p['unnormalized_source']=raw_uniform[ev_idx]
                for method,probability in p.items():
                    rows.append(dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                        shots_per_class=shots,method=method,feature_bank='F0|F3_Ring_reference',
                        calibration_trials=len(cal_idx),evaluation_trials=len(et),supported=True,
                        **_metrics(ey,probability)))
                    saved[f'{user}_{domain}_{shots}_{method}']=probability
                splits.append(dict(user=user,domain=domain,shots=shots,train=st.tolist(),calibration=trials[cal_idx].tolist(),
                    evaluation=et.tolist(),source_oof=folds))
                descriptors.append(dict(user=user,domain=domain,shots=shots,updates=session['descriptor'] if session is not None else None))
            for shots in (2,5):rows.append(dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                shots_per_class=shots,method='unsupported_budget',feature_bank='F0|F3_Ring_reference',supported=False,
                reason='Only two native trials/class/wearing-domain; no evaluation trial remains'))
    metrics=('macro_f1','accuracy','log_loss','brier','ece')
    for condition in ('ALL','trial_1','trial_2','trial_3','trial_4'):
        for shots in (0,1):
            for method in (*BRANCHES,'session_context','unnormalized_source'):
                selected=[r for r in rows if r['subject']!='ALL' and r['shots_per_class']==shots and r['method']==method and (condition=='ALL' or r['condition']==condition)]
                classes=[json.loads(r['per_class_f1_json']) for r in selected]
                rows.append({**selected[0],'subject':'ALL','condition':condition,
                    **{metric:float(np.mean([r[metric] for r in selected])) for metric in metrics},
                    'per_class_f1_json':json.dumps({h:float(np.mean([r[h] for r in classes])) for h in classes[0]}),
                    'calibration_trials':sum(r['calibration_trials'] for r in selected),'evaluation_trials':sum(r['evaluation_trials'] for r in selected),
                    'aggregation':'mean individual user/domain scores; budgets have different remaining trials'})
    output.mkdir(parents=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'calibration_curve.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (output/'fitted_states.pkl').open('wb') as handle:pickle.dump((models,sessions,temperatures),handle)
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'session_updates.json').write_text(json.dumps(descriptors,indent=2))
    (output/'run_manifest.json').write_text(json.dumps(dict(phase=phase,users=users,rest_label=2,source_temperatures=temperatures,
        source_oof='Each before-wearing repetition evaluates five trials; next repetition provides simulated session calibration; remaining three fit every normalizer/family/scaler/classifier/profile.',
        parent_sha256={name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in ('fitted_states.pkl','run_manifest.json')},
        raw_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        context_rule='prespecified clipped cosine agreement; prototype beta source class count/(source+cal class count)',
        evidence_boundary='Known-user native wearing domains, not calendar days. Explicit8-channel ring200Hz/40sample contract; no actual device result. All branches have source OOF temperatures before fusion.'),indent=2))
    replay(archive,parent,output)


def replay(archive,parent,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['raw_archive_sha256']:raise AssertionError('Native archive changed')
    for name,digest in manifest['parent_sha256'].items():
        if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Raw source baseline changed')
    models,sessions,temps=pickle.loads((output/'fitted_states.pkl').read_bytes())
    before=pickle.dumps((models,sessions));checked=0
    with np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for user,(pipeline,folds) in models.items():
            source=load(archive,user,('training',));sy=saved[f'{user}_source_labels'];st=saved[f'{user}_source_trials']
            oof={f'{branch}_{name}':np.zeros((len(st),5)) for branch in BRANCHES for name in FAMILIES}
            for rep,(fold,session) in folds.items():
                selection=np.array([int(PATH_RE.fullmatch(trial)['rep'])==rep for trial in source.trials])
                raw,y,trials=fold.predict(source.take(np.flatnonzero(selection)),session)
                idx=np.array([np.flatnonzero(st==trial).item() for trial in trials])
                for branch in BRANCHES:
                    for name in FAMILIES:oof[f'{branch}_{name}'][idx]=raw[branch][name]
            for key,p in oof.items():
                np.testing.assert_allclose(p,saved[f'{user}_oof_{key}'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(fit_temperature(p,sy),temps[str(user)][key],atol=1e-12,rtol=0);checked+=1
            for domain in ('trial_1','trial_2','trial_3','trial_4'):
                data=load(archive,user,(domain,));_,y,_,trials=aggregate(np.zeros((len(data.labels),1)),data)
                for shots in (0,1):
                    cal=calibration_indices(y,shots,user,domain);et=trials[~np.isin(np.arange(len(y)),cal)]
                    ev=data.take(np.flatnonzero(np.isin(data.trials,et)));session=sessions[(user,domain,shots)]
                    raw,_,_=pipeline.predict(ev,session)
                    p=fuse(raw,temps[str(user)],session['context_weights'] if session is not None else None,shots>0)
                    for method,probability in p.items():
                        np.testing.assert_allclose(probability,saved[f'{user}_{domain}_{shots}_{method}'],atol=1e-12,rtol=0);checked+=1
    if before!=pickle.dumps((models,sessions)):raise AssertionError('Replay changed source/session profile')
    result=dict(status='ok',prediction_arrays_replayed=checked,native_data_reloaded=True,
        source_temperature_recomputed=True,source_and_session_immutable=True,
        scope='Integrated normalized source/session providers and source OOF; raw source baseline replay retained in parent run')
    (output/'replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def diagnostics(archive,output):
    """Per-family calibrated predictions and paired session-input controls."""
    models,sessions,temperatures=pickle.loads((output/'fitted_states.pkl').read_bytes())
    manifest=json.loads((output/'run_manifest.json').read_text())
    phase=manifest['phase'];family_rows=[];increments=[];saved={}
    for user,(pipeline,_) in models.items():
        for domain in ('trial_1','trial_2','trial_3','trial_4'):
            data=load(archive,user,(domain,));_,y,_,trials=aggregate(np.zeros((len(data.labels),1)),data)
            for shots in (0,1):
                cal=calibration_indices(y,shots,user,domain);et=trials[~np.isin(np.arange(len(y)),cal)]
                ev=data.take(np.flatnonzero(np.isin(data.trials,et)))
                session=sessions[(user,domain,shots)]
                raw,ey,_=pipeline.predict(ev,session)
                p={}
                common=dict(dataset='libemg_electrode_shift',phase=phase,subject=user,condition=domain,
                    calibration_budget=shots,evaluation_trials=len(et),protocol='source OOF temperatures; same held-out wearing trials')
                for branch in BRANCHES:
                    effective=branch if shots or branch in ('source_model','source_long_anchor') else 'source_long_anchor' if 'anchor' in branch else 'source_model'
                    p[branch]={}
                    for name in FAMILIES:
                        probability=temperature_probability(raw[branch][name],temperatures[str(user)][f'{effective}_{name}'])
                        p[branch][name]=probability;saved[f'{user}_{domain}_{shots}_{branch}_{name}']=probability
                        dimension=len(pipeline.models_[name][0].mean_)
                        family_rows.append({**common,'feature_family':name,'method':branch,'feature_dimension':dimension,
                            **_metrics(ey,probability)})
                fused=fuse(raw,temperatures[str(user)],session['context_weights'] if session else None,shots>0)
                for base,added,label in (('source_model','session_model','session_rest_scale_update'),
                        ('session_model','session_context','session_signature_weighting'),
                        ('session_model','session_local_anchor','current_session_prototype')):
                    a,b=_metrics(ey,fused[base]),_metrics(ey,fused[added])
                    increments.append({**common,'core_bank':'F0|F3_Ring_source_normalized','added_family':label,
                        'comparison':f'{base}->{added}','delta_logloss':a['log_loss']-b['log_loss'],
                        'delta_macro_f1':b['macro_f1']-a['macro_f1'],'delta_brier':a['brier']-b['brier'],
                        'scope':'calibration-conditioned input/prototype/fusion control; not concatenated extra-family classifier or MI'})
    metrics=('macro_f1','accuracy','log_loss','brier','ece')
    originals=list(family_rows)
    for condition in ('ALL','trial_1','trial_2','trial_3','trial_4'):
        for shots in (0,1):
            for branch in BRANCHES:
                for name in FAMILIES:
                    subset=[r for r in originals if r['calibration_budget']==shots and r['method']==branch and r['feature_family']==name and (condition=='ALL' or r['condition']==condition)]
                    classes=[json.loads(r['per_class_f1_json']) for r in subset]
                    family_rows.append({**subset[0],'subject':'ALL','condition':condition,
                        **{metric:float(np.mean([r[metric] for r in subset])) for metric in metrics},
                        'per_class_f1_json':json.dumps({h:float(np.mean([r[h] for r in classes])) for h in classes[0]}),
                        'aggregation':'mean individual user/domain scores'})
    original_increments=list(increments)
    for condition in ('ALL','trial_1','trial_2','trial_3','trial_4'):
        for shots in (0,1):
            for label in ('session_rest_scale_update','session_signature_weighting','current_session_prototype'):
                subset=[r for r in original_increments if r['calibration_budget']==shots and r['added_family']==label and (condition=='ALL' or r['condition']==condition)]
                increments.append({**subset[0],'subject':'ALL','condition':condition,
                    **{metric:float(np.mean([r[metric] for r in subset])) for metric in ('delta_logloss','delta_macro_f1','delta_brier')},
                    'aggregation':'mean paired individual user/domain deltas'})
    from .core_incremental_oof import write
    def union_write(path,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    union_write(output/'feature_family_results.csv',family_rows)
    union_write(output/'conditional_incremental.csv',increments)
    np.savez_compressed(output/'calibrated_provider_predictions.npz',**saved)
    (output/'diagnostic_audit.json').write_text(json.dumps(dict(status='ok',family_rows=len(family_rows),
        paired_control_rows=len(increments),provider_arrays=len(saved),classifiers_refit=False,
        scope='Per-family empirical source-OOF probabilities and paired calibration controls, not literal concat Core additions'),indent=2))
    print(json.dumps({'family_rows':len(family_rows),'paired_control_rows':len(increments),'classifiers_refit':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('parent',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=('validation','final'),required=True);parser.add_argument('--replay',action='store_true')
    parser.add_argument('--diagnostics',action='store_true');args=parser.parse_args()
    if args.diagnostics:diagnostics(args.archive,args.output)
    elif args.replay:replay(args.archive,args.parent,args.output)
    else:
        run(args.archive,args.parent,args.output,args.phase)
        diagnostics(args.archive,args.output)
