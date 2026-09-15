"""Known-user session prototype blend controls using trusted frozen native MANUS models."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.semg_manus import load_semg_manus_windows
from .manus_study import GESTURES, _aggregate, _metrics
from .calibration import ReliabilityWeights, SessionSignature, late_fusion
from .force_nested_oof import temperature_probability
from .session_anchor import SessionPrototypeAnchor


def anchor_probability(anchor,x,temperature):
    z=-anchor.transform(x)[:,:len(anchor.classes_)].astype(float)/temperature
    z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)


def run(archive,source,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    if manifest['dataset']!='semg_manus':raise ValueError('Same-user multi-session MANUS only')
    phase=manifest['phase'];splits=json.loads((source/'split_trial_ids.json').read_text())
    states,old_anchors=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps((states,old_anchors))
    ids=tuple(manifest['families']);users=sorted({s['user'] for s in splits})
    train=load_semg_manus_windows(archive,users=users,sessions=(1,),gestures=GESTURES)
    target=load_semg_manus_windows(archive,users=users,sessions=(2 if phase=='validation' else 3,),gestures=GESTURES)
    if set(train.trials)!=set(manifest['source_trials']):raise AssertionError('Long-term source trial mismatch')
    temperatures=json.loads((source/'probability_calibration.json').read_text())['temperatures']
    features={};profiles={};raw={}
    for n,(f,s,m) in states.items():
        a,ay,au,_,_,at=_aggregate(f.transform(train.batch),train)
        b,y,u,_,speed,trials=_aggregate(f.transform(target.batch),target)
        profiles[n]=s.transform(a);features[n]=s.transform(b)
        raw[n]=temperature_probability(m.predict_proba(features[n]),temperatures[n])
        if n=='F9_Quality':qmean=np.clip(b[:,-3],0,1);qmin=np.clip(b[:,-2],0,1)
    population=np.asarray(manifest['population_weights'])
    reliability=ReliabilityWeights(tuple(range(6)),ids,population,n0=manifest['n0'],temperature=manifest['reliability_temperature'])
    rows=[];saved={};inputs={};anchors={};controls=[];groups={};checked=0;error=0.
    with np.load(source/'heldout_predictions.npz',allow_pickle=False) as reference:
        for key,value in (('labels',y),('users',u),('trials',trials)):np.testing.assert_array_equal(reference[key],value)
        for split in splits:
            user=split['user'];shots=split['shots'];cal=np.flatnonzero(np.isin(trials,split['calibration']))
            ev=np.flatnonzero(np.isin(trials,split['evaluation']))
            if set(trials[cal])&set(trials[ev]) or set(trials[cal])&set(at):raise AssertionError('Calibration leakage')
            w=population if not shots else reliability.personal({n:(features[n][cal],y[cal]) for n in ids})
            context=w.copy();quality={n:np.ones(len(ev)) if n in ('F6_IMU','F9_Quality') else qmin[ev] if n in ('F0','F2b_CSP') else qmean[ev] for n in ids}
            modes={mode:{} for mode in ('long_term','local','blended','original_local')}
            print(f'{phase}: known user {user} cal{shots}; long/session prototype controls',flush=True)
            for i,n in enumerate(ids):
                source_x=profiles[n][au==user];source_y=ay[au==user]
                profile=SessionPrototypeAnchor().fit_long_term(source_x,source_y)
                temp=max(float(np.median(profile.anchor_.transform(source_x)[:,:6])),1e-10)
                if shots:
                    signature=SessionSignature().fit_long_term(source_x,source_y)
                    v=signature.from_session_calibration(features[n][cal],y[cal]);context[i]*=float(np.clip((v[6:12].mean()+1)/2,.05,1))
                count=float(profile.counts_.min());alpha=(count+shots)/(count+shots+2)
                for mode in ('long_term','local','blended'):
                    anchor,beta=profile.from_calibration(features[n][cal] if shots else None,y[cal] if shots else None,mode=mode)
                    anchors[(user,shots,n,mode)]=(anchor,temp)
                    p=anchor_probability(anchor,features[n][ev],temp)
                    modes[mode][n]=(1-alpha)*raw[n][ev]+alpha*p if shots or mode!='local' else raw[n][ev]
                    controls.append({'user':user,'shots':shots,'family':n,'mode':mode,'beta':beta.tolist(),'alpha':alpha,
                        'source_scale_fixed':True,'source_trials':at[au==user].tolist()})
                if shots:
                    anchor,t=old_anchors[(user,shots,n)];p=anchor_probability(anchor,features[n][ev],t)
                    a=shots/(shots+2);modes['original_local'][n]=(1-a)*raw[n][ev]+a*p
                else:modes['original_local'][n]=raw[n][ev]
            context/=context.sum()
            variants={mode:late_fusion(p,ids,context,quality) for mode,p in modes.items()}
            variants.update({'without_F7_anchor':late_fusion({n:raw[n][ev] for n in ids},ids,context,quality),
                'blended_without_F8':late_fusion(modes['blended'],ids,w,quality),
                'blended_without_F9_routing':late_fusion(modes['blended'],ids,context),
                'blended_without_F9_routing_and_provider':late_fusion({n:p for n,p in modes['blended'].items() if n!='F9_Quality'},ids,context),
                **{f'blended_without_{n}':late_fusion({k:p for k,p in modes['blended'].items() if k!=n},ids,context,quality) for n in ids}})
            original=reference[f'{user}_{shots}_combined_full']
            np.testing.assert_allclose(variants['original_local'],original,atol=1e-8,rtol=1e-7)
            checked+=1;error=max(error,float(np.abs(original-variants['original_local']).max()))
            prefix=f'{user}_{shots}';saved[f'{prefix}_labels']=y[ev];saved[f'{prefix}_trials']=trials[ev];saved[f'{prefix}_speed']=speed[ev]
            inputs[f'{prefix}_weights']=w;inputs[f'{prefix}_context']=context
            for n in ids:
                inputs[f'{prefix}_raw_{n}']=raw[n][ev];inputs[f'{prefix}_quality_{n}']=quality[n]
                for mode in modes:inputs[f'{prefix}_{mode}_{n}']=modes[mode][n]
            for method,p in variants.items():
                saved[f'{prefix}_{method}']=p
                for value in ('ALL',*sorted(set(speed[ev]))):
                    mask=np.ones(len(ev),bool) if value=='ALL' else speed[ev]==value
                    row={'dataset':'semg_manus','phase':phase,'protocol':'B_session','subject':user,'condition':value,
                        'shots_per_class':shots,'method':method,**_metrics(y[ev][mask],p[mask])}
                    rows.append(row);groups.setdefault((shots,value,method),[]).append(row)
    for group in groups.values():
        result={k:float(np.mean([r[k] for r in group])) for k in ('macro_f1','accuracy','log_loss','brier','ece')}
        pc=[json.loads(r['per_class_f1_json']) for r in group]
        result['per_class_f1_json']=json.dumps({k:float(np.mean([v[k] for v in pc])) for k in pc[0]})
        rows.append({**group[0],'subject':'ALL',**result})
    if pickle.dumps((states,old_anchors))!=before:raise AssertionError('Frozen source state mutated')
    output.mkdir(parents=True)
    with (output/'ablation_full_bank.csv').open('w',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'heldout_predictions.npz',**saved)
    np.savez_compressed(output/'fusion_inputs.npz',**inputs)
    with (output/'session_anchors.pkl').open('wb') as h:pickle.dump(anchors,h)
    (output/'split_trial_ids.json').write_text(json.dumps(splits,indent=2))
    (output/'prototype_controls.json').write_text(json.dumps(controls,indent=2))
    (output/'run_manifest.json').write_text(json.dumps({'dataset':'semg_manus','phase':phase,'source_run':source.name,
        'families':ids,'source_state_sha256':hashlib.sha256((source/'fitted_states.pkl').read_bytes()).hexdigest(),
        'source_predictions_sha256':hashlib.sha256((source/'heldout_predictions.npz').read_bytes()).hexdigest(),
        'beta_rule':'N_long_class/(N_long_class+N_cal_class); source counts and calibration budget only',
        'alpha_rule':'(min source class trial count+shots)/(min source class trial count+shots+2)',
        'classifier_or_family_fit':False,'source_anchor_temperature':'median source prototype distances only',
        'original_units_replayed':checked,'max_original_probability_error':error,'rows':len(rows),
        'limitations':'source coordinate scale fixed; no raw channel normalization branch; cal5 unsupported; source session1 and target speed/session confounded'},indent=2))
    print(json.dumps({'status':'ok','rows':len(rows),'original_units_replayed':checked,'max_original_probability_error':error}))


def replay(output):
    manifest=json.loads((output/'run_manifest.json').read_text());ids=tuple(manifest['families'])
    splits=json.loads((output/'split_trial_ids.json').read_text());checked=set();error=0.
    with np.load(output/'fusion_inputs.npz',allow_pickle=False) as inputs,np.load(output/'heldout_predictions.npz',allow_pickle=False) as saved:
        for split in splits:
            prefix=f"{split['user']}_{split['shots']}"
            np.testing.assert_array_equal(saved[f'{prefix}_trials'],np.sort(split['evaluation']))
            if set(split['calibration'])&set(split['evaluation']):raise AssertionError('Trial leakage')
            context=inputs[f'{prefix}_context'];w=inputs[f'{prefix}_weights']
            q={n:inputs[f'{prefix}_quality_{n}'] for n in ids}
            modes={mode:{n:inputs[f'{prefix}_{mode}_{n}'] for n in ids} for mode in ('long_term','local','blended','original_local')}
            ps={mode:late_fusion(p,ids,context,q) for mode,p in modes.items()}
            ps.update({'without_F7_anchor':late_fusion({n:inputs[f'{prefix}_raw_{n}'] for n in ids},ids,context,q),
                'blended_without_F8':late_fusion(modes['blended'],ids,w,q),
                'blended_without_F9_routing':late_fusion(modes['blended'],ids,context),
                'blended_without_F9_routing_and_provider':late_fusion({n:p for n,p in modes['blended'].items() if n!='F9_Quality'},ids,context),
                **{f'blended_without_{n}':late_fusion({k:p for k,p in modes['blended'].items() if k!=n},ids,context,q) for n in ids}})
            for method,p in ps.items():
                key=f'{prefix}_{method}';np.testing.assert_allclose(p,saved[key],rtol=1e-9,atol=1e-10)
                error=max(error,float(np.abs(p-saved[key]).max()));checked.add(key)
        if checked!={k for k in saved.files if not k.endswith(('_labels','_trials','_speed'))}:
            raise AssertionError('Missing variant replay')
    audit={'status':'ok','prediction_arrays_checked':len(checked),'max_absolute_probability_error':error,
        'classifier_or_family_fit':False,'scope':'all fused variant arrays rebuilt from retained provider probabilities and fixed weights/quality'}
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.output) if a.replay else run(a.archive,a.source,a.output)
