"""Paired synthetic quality stress tests with frozen native force classifiers."""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from emgimu.datasets.libemg_force import load_libemg_force_windows
from .core import FeatureBatch
from .force_full_fusion import IDS, aggregate
from .force_nested_oof import temperature_probability
from .calibration import late_fusion
from .calibration_study import CONDITIONS
from .screening import metrics, SEED
from .core_incremental_oof import write

SCENARIOS=('clean','gaussian_0.25_source_rms','gaussian_0.5_source_rms',
    'line_50Hz_source_rms','synthetic_source_q99_saturation','dropout_ch3',
    'contiguous_half_flatline_ch3','gain2_ch3')


def corrupt(x, rate, scenario, amplitude, bound):
    """Perturb a copy; all scales are frozen source statistics, never test estimates."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown fixed corruption scenario')
    z=np.asarray(x,dtype=float).copy()
    if scenario.startswith('gaussian_'):
        scale=float(scenario.split('_')[1])
        z+=np.random.default_rng(SEED).normal(size=z.shape)*scale*amplitude[None,None,:]
    elif scenario=='line_50Hz_source_rms':
        z+=amplitude[None,None,:]*np.sin(2*np.pi*50*np.arange(z.shape[1])/rate)[None,:,None]
    elif scenario=='synthetic_source_q99_saturation':
        z=np.clip(z,-bound[None,None,:],bound[None,None,:])
    elif scenario=='dropout_ch3':
        z[:,:,2]=0
    elif scenario=='contiguous_half_flatline_ch3':
        z[:,:z.shape[1]//2,2]=z[:,0:1,2]
    elif scenario=='gain2_ch3':
        z[:,:,2]*=2
    return z


def run(raw,source,output):
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((source/'run_manifest.json').read_text())
    phase=manifest['phase']
    if phase not in ('validation','final') or manifest['source_force_only']!='Ramp':
        raise ValueError('Requires frozen source-Ramp validation/final force bank')
    users=(7,8) if phase=='validation' else (9,10)
    train=load_libemg_force_windows(raw,subjects=range(1,7),conditions=('Ramp',))
    if set(train.trials)!=set(manifest['source_trials']):raise AssertionError('Source trial mismatch')
    rms=np.sqrt(np.mean(train.batch.emg.astype(float)**2,axis=1))
    amplitude=np.median(rms,axis=0)
    bound=np.quantile(np.abs(train.batch.emg),.99,axis=(0,1))
    data=load_libemg_force_windows(raw,subjects=users,conditions=CONDITIONS)
    if set(train.trials)&set(data.trials):raise AssertionError('Source/evaluation overlap')
    states,_=pickle.loads((source/'fitted_states.pkl').read_bytes());before=pickle.dumps(states)
    temperature=json.loads((source/'probability_calibration.json').read_text())
    if set(temperature['fit_users'])!=set(range(1,7)) or set(temperature['fit_trials'])!=set(train.trials):
        raise AssertionError('Temperature source leakage')
    rows=[];saved={};clean_checks=0
    for scenario in SCENARIOS:
        altered=replace(data,batch=FeatureBatch(corrupt(data.batch.emg,data.batch.sample_rate_hz,scenario,amplitude,bound),data.batch.sample_rate_hz))
        providers={}
        for name in IDS:
            family,scaler,model=states[name]
            x,y,u,trials=aggregate(family.transform(altered.batch),altered)
            providers[name]=temperature_probability(model.predict_proba(scaler.transform(x)),temperature['temperatures'][name])
            if name=='F9_Quality':
                qmean=np.clip(x[:,-3],0,1);qmin=np.clip(x[:,-2],0,1)
        quality={n:qmin if n in ('F0','F2b_CSP') else qmean for n in IDS}
        weights=np.ones(len(IDS))/len(IDS)
        variants={**providers,'full_uniform':late_fusion(providers,IDS,weights),
            'full_without_quality_provider':late_fusion({n:p for n,p in providers.items() if n!='F9_Quality'},IDS,weights),
            'full_quality_routing':late_fusion(providers,IDS,weights,quality),
            'full_quality_routing_without_quality_provider':late_fusion({n:p for n,p in providers.items() if n!='F9_Quality'},IDS,weights,quality)}
        if scenario=='clean':
            with np.load(source/'heldout_predictions.npz',allow_pickle=False) as reference:
                for key,value in (('labels',y),('users',u),('trials',trials)):np.testing.assert_array_equal(reference[key],value)
                for user in users:
                    mask=u==user
                    for original,new in (('uniform_population','full_uniform'),('baseline_F0','F0')):
                        np.testing.assert_allclose(reference[f'{user}_0_{original}'],variants[new][mask],atol=1e-10,rtol=1e-9)
                        clean_checks+=1
        for user in (*users,'ALL'):
            mask=np.ones(len(y),bool) if user=='ALL' else u==user
            for method,p in variants.items():
                rows.append({'dataset':'libemg_contraction_intensity','phase':phase,'protocol':'paired_synthetic_quality',
                    'subject':user,'condition':scenario,'scenario':scenario,'shots_per_class':0,
                    'feature_family':method if method in IDS else '|'.join(n for n in IDS if 'without_quality_provider' not in method or n!='F9_Quality'),
                    'method':method,'synthetic':scenario!='clean','aggregation':'pooled trials' if user=='ALL' else 'individual user',
                    'quality_mean':float(qmean[mask].mean()),'quality_min':float(qmin[mask].mean()),
                    **metrics(y[mask],p[mask],np.ones(mask.sum()))})
        for name,p in variants.items():saved[f'{scenario}::{name}']=p
        saved[f'{scenario}::quality_mean']=qmean;saved[f'{scenario}::quality_min']=qmin
        print(f'{phase}: {SCENARIOS.index(scenario)+1}/{len(SCENARIOS)} {scenario}',flush=True)
    if pickle.dumps(states)!=before:raise AssertionError('Frozen source state changed')
    output.mkdir(parents=True)
    write(output/'feature_family_results.csv',rows)
    write(output/'quality_corruption_results.csv',rows)
    np.savez_compressed(output/'corruption_predictions.npz',labels=y,users=u,trials=trials,**saved)
    (output/'split_trial_ids.json').write_text(json.dumps({'train':sorted(set(train.trials)),'calibration':[],'evaluation':trials.tolist()},indent=2))
    evidence={'dataset':'libemg_contraction_intensity','phase':phase,'families':list(IDS),'seed':SEED,
        'source_run':source.name,'source_statistics':{'median_window_rms':amplitude.tolist(),'abs_sample_q99':bound.tolist()},
        'scenarios':list(SCENARIOS),'classifier_or_family_fit':False,'target_calibration':False,
        'source_artifact_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('fitted_states.pkl','run_manifest.json','probability_calibration.json','heldout_predictions.npz')},
        'routing':'prespecified exploratory legacy F9 min for F0/CSP, mean for other providers; zero-weight fallback to base prior',
        'scope':'paired synthetic corruption on identical native trials; force and user conditions held fixed across perturbations; not measured hardware noise',
        'saturation':'synthetic clipping at source abs-sample q99, not known ADC bounds',
        'window_limits':'synthetic corruption applied independently within sparse windows; discontinuities at window edges are not a continuous-device simulation'}
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2))
    (output/'replay_audit.json').write_text(json.dumps({'status':'ok','clean_reference_arrays_checked':clean_checks,'source_state_unchanged':True,'rows':len(rows)},indent=2))


def replay(source,output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    for name,expected in manifest['source_artifact_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise AssertionError('Changed source artifact')
    count=0;error=0.
    with np.load(output/'corruption_predictions.npz',allow_pickle=False) as z:
        for scenario in SCENARIOS:
            providers={n:z[f'{scenario}::{n}'] for n in IDS}
            qmean=z[f'{scenario}::quality_mean'];qmin=z[f'{scenario}::quality_min']
            for routed in (False,True):
                for removed in (False,True):
                    active=[n for n in IDS if not removed or n!='F9_Quality']
                    w=np.ones((len(qmean),len(active)))
                    if routed:
                        w*=np.stack([qmin if n in ('F0','F2b_CSP') else qmean for n in active],axis=1)
                    w[w.sum(1)<=1e-10]=1
                    w/=w.sum(1,keepdims=True)
                    p=sum(w[:,i,None]*providers[n] for i,n in enumerate(active))
                    p/=p.sum(1,keepdims=True)
                    method=('full_quality_routing' if routed else 'full_uniform') if not removed else (
                        'full_quality_routing_without_quality_provider' if routed else 'full_without_quality_provider')
                    np.testing.assert_allclose(p,z[f'{scenario}::{method}'],atol=1e-10,rtol=1e-9)
                    error=max(error,float(np.abs(p-z[f'{scenario}::{method}']).max()));count+=1
    audit=json.loads((output/'replay_audit.json').read_text())
    audit.update(fusion_arrays_checked=count,max_absolute_probability_error=error,source_hashes_checked=True)
    (output/'replay_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    a=p.parse_args();replay(a.source,a.output) if a.replay else run(a.raw,a.source,a.output)
