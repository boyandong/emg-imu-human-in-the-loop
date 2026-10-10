"""Precommitted matched refit of exact document window representations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import numpy as np
import scipy
from scipy.special import softmax
import sklearn
from benchmarks.song_real8_study import load_session
from benchmarks.song_real8.song_personal_session_workflow_v1 import aggregate, windows, score
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_stream_v2 import load_gate
from emgimu.feature_bank.document_window_composition_v1 import (
    SCHEMA, GROUPS, CLASSES, CHANNELS, PREPROCESSING, FAMILY_GROUPS,
    fit_document_source, source_trial_features, build_document_workflow, load_document_workflow)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'DOCUMENT_WINDOW_COMPOSITION_V1_PROTOCOL.json'
RESULT = HERE/'DOCUMENT_WINDOW_COMPOSITION_V1_RESULTS.json'
OUT = HERE/'document_window_composition_v1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())
        stream.write('\n')


def prepare():
    parent = json.loads((HERE/'SONG_EXTENDED_WINDOW_V1_PROTOCOL.json').read_text(encoding='utf8'))
    names = list((ROOT/'src/emgimu/feature_bank').glob('*.py'))
    names += [Path(__file__), ROOT/'tests/test_document_window_composition_v1.py',
        HERE/'verify_document_window_composition_v1.py',
        ROOT/'tests/test_extended_window_decision_v1.py', ROOT/'tests/test_joint_bout_workflow_v1.py',
        ROOT/'benchmarks/song_real8_study.py', HERE/'song_personal_session_workflow_v1.py',
        HERE/'song_raw_quality_v1.py']
    artifacts = [HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json', HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json',
        HERE/'song_extended_window_v1/source_bank.pkl', HERE/'song_extended_window_v1/readouts.npz',
        HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json']
    write(PROTOCOL, dict(schema=SCHEMA, source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in names},
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts},
        hdf5_sha256=parent['hdf5_sha256'], source_folder=parent['source_folder'], seed=20261011,
        groups=GROUPS, family_groups=FAMILY_GROUPS, current_shots=[0,1,2,5],
        fit='Refit all seven representations/scalers/balanced logistic C1 classifiers separately inside leave-S01/S02-recording-out source folds and final joined source. F0 Rest-only thresholds. Exact document V3 F1/F2a,c/CES/F4/F5a; uncentered source-only CSP. 281 coordinates. No ring/anatomical F6 fabrication.',
        source_policy='OOF temperatures per provider and softmax(-source OOF logloss) prior. Inherit prior frozen n0/reliability temperature without target search. Pooled source OOF calibration/prior are not an unbiased source performance estimate.',
        split='Exact earlier S03 long20, S04 nested current0/4/8/20 and same124 evaluation trials. Previously inspected same person/day; retrospective, not prospective validation. Source and target identity sets disjoint.',
        composition='Operational F7 affineSPD/F8 calibration routing/raw F9 structural; separate F4d context. Population/uniform/reliability/F7/F8/full; full branch removals; seven fixed-weight provider removals and six family removals (F2 jointly removes F2ac and CSP). Existing seven-provider controls read from frozen probabilities, not rerun.',
        primary='Five-shot new reliability versus frozen old seven reliability: lower logloss/Brier, nonworse macroF1 and all four class recalls. Separate new-full versus frozen old-full and versus new-reliability; no default promotion regardless of retrospective result.',
        scope='Exact eligible document window composition with source-only refit and native matched ablations. Not complete F0-F9: F3a/c geometry and anatomicalF6 unavailable; F5b/c complete-bout system separately tested, Song stable excerpts are not full bouts. Physical faults, cross-user/day, prospective device efficacy and physical timing unproved.',
        default_promoted=False))


def read(families, models, item):
    output = {}
    for g, views in families.items():
        x, ids = source_trial_features(views, item['batch'], item['trial'])
        scaler, model = models[g]
        output[g] = model.predict_proba(scaler.transform(x))
    labels = dict(zip(item['trial'], item['hand']))
    return output, np.array([labels[t] for t in ids]), ids


def run():
    if RESULT.exists() or OUT.exists():
        raise FileExistsError('Completed or partially emitted native study must not be rerun')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name, digest in {**p['source_sha256'], **p['artifact_sha256']}.items():
        if sha(ROOT/name) != digest: raise ValueError('Frozen input changed: '+name)
    prior = json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    previous = json.loads((HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    gate = load_gate(HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    old_bank = pickle.loads((HERE/'song_extended_window_v1/source_bank.pkl').read_bytes())
    old_before = pickle.dumps((old_bank,gate))
    def native(s):
        item = load_session(Path(p['source_folder'])/f'2026-09-18_{s}', s, filter_mode='causal')
        if item['audit']['sha256'] != p['hdf5_sha256'][s]: raise ValueError('Native recording checksum differs')
        item['batch'] = FeatureBatch(item['batch'].emg, 250.)
        return item
    print('1/6 Source-only document refits in two recording folds', flush=True)
    sources = {s:native(s) for s in ('S01','S02')}
    folds = []
    for saved in prior['source_oof']:
        held = saved['session']; train = sources['S02' if held == 'S01' else 'S01']
        families, models, _ = fit_document_source(train['batch'], train['hand'], train['trial'],
            forbidden_trials=sources[held]['trial'], seed=p['seed'])
        assert sorted(set(train['trial'])) == saved['fit_ids']
        raw, y, ids = read(families, models, sources[held])
        assert list(ids) == saved['validation_ids'] and list(y) == saved['labels']
        folds.append(dict(session=held, fit_ids=saved['fit_ids'], ids=ids, labels=y, raw=raw))
    y = np.concatenate([f['labels'] for f in folds]); encoded = np.array([CLASSES.index(c) for c in y])
    temperatures = {g:fit_temperature(np.concatenate([f['raw'][g] for f in folds]),encoded) for g in GROUPS}
    for f in folds:
        f['calibrated'] = {g:temperature_probability(f['raw'][g],temperatures[g]) for g in GROUPS}
    losses = {g:score(y,np.concatenate([f['calibrated'][g] for f in folds]))['log_loss'] for g in GROUPS}
    population = softmax(-np.array([losses[g] for g in GROUPS]))
    joined = dict(batch=FeatureBatch(np.concatenate([s['batch'].emg for s in sources.values()]),250.),
        trial=np.concatenate([s['trial'] for s in sources.values()]),hand=np.concatenate([s['hand'] for s in sources.values()]))
    families,models,metadata = fit_document_source(joined['batch'],joined['hand'],joined['trial'],seed=p['seed'])
    bank = FrozenEmgProviderBankV1(families,models,temperatures,classes=CLASSES,class_names=CLASSES,
        source_trial_ids=np.unique(joined['trial']),source_policy_id=sha(PROTOCOL),sample_rate_hz=250.,
        window_samples=50,channels=8,population=population,n0=old_bank.policy_.n0,
        reliability_temperature=old_bank.policy_.temperature)
    w = build_document_workflow(bank,gate);before=pickle.dumps(w)
    OUT.mkdir();package=OUT/'source_bank.pkl';package.write_bytes(pickle.dumps(bank))
    policy=OUT/'policy.json'
    write(policy,dict(schema=SCHEMA,source_bank_sha256=sha(package),source_bank_id=bank.bank_id_,
        decision_policy_id=w.policy_id,anchor_mix=.5))
    print('2/6 Same long20 calibration and held-out124 native trials',flush=True)
    long,current=native('S03'),native('S04');evaluation=previous['evaluation_ids']
    eb,ei,eo=windows(current,evaluation);labels=dict(zip(current['trial'],current['hand']))
    ey=np.array([labels[t] for t in evaluation]);assert list(ey)==previous['evaluation_labels']
    kwargs=dict(user_id='Song',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
    b,ids,offsets=windows(long,previous['personal_calibration_ids']);ly=dict(zip(long['trial'],long['hand']))
    personal=w.enroll_user(b,ids,{t:ly[t] for t in np.unique(ids)},window_offsets=offsets,
        session_id='S03',forbidden_evaluation_trials=evaluation,**kwargs)
    pp=OUT/'personal_S03.zip';w.save_profile(personal,pp)
    from benchmarks.song_real8.song_raw_quality_v1 import native as raw_native
    raw_item,_,_,raw_windows=raw_native('S04',p)
    assert np.array_equal(raw_item['trial'],current['trial'])
    raw_batch=raw_windows.take(np.flatnonzero(np.isin(current['trial'],evaluation)))
    old_readouts=np.load(HERE/'song_extended_window_v1/readouts.npz',allow_pickle=False)
    cells=[];rows=[];arrays={};profiles={};roundtrips=[]
    readout=bank.predict_providers(eb,ei,window_offsets=eo,user_id='Song')
    for g,q in readout['probabilities'].items(): arrays['provider_'+g]=q
    for shots in p['current_shots']:
        session=None;sp=None;cal=[] if not shots else previous['profiles'][str(shots)]['calibration_ids']
        if shots:
            cb,ci,co=windows(current,cal)
            session=w.calibrate_session(cb,ci,{t:labels[t] for t in cal},window_offsets=co,
                personal=personal,session_id='S04',forbidden_evaluation_trials=evaluation,**kwargs)
            sp=OUT/f'session_S04_{shots}shot.zip';w.save_profile(session,sp)
        profiles[str(shots)]=dict(calibration_ids=cal,path=None if sp is None else sp.relative_to(ROOT).as_posix())
        state_before=pickle.dumps((personal,session))
        common=dict(window_offsets=eo,personal=personal,session=session,session_id='S04',raw_batch=raw_batch,**kwargs)
        definitions=dict(new_reliability=dict(use_anchor=False,use_session_routing=False),
            new_F7=dict(use_session_routing=False),new_F8=dict(use_anchor=False),
            new_full=dict(quality_mode='structural'),full_minus_F7=dict(use_anchor=False,quality_mode='structural'),
            full_minus_F8=dict(use_session_routing=False,quality_mode='structural'),full_minus_F9=dict())
        definitions.update({'minus_provider_'+g:dict(available=tuple(n for n in GROUPS if n!=g),quality_mode='structural') for g in GROUPS})
        definitions.update({'minus_family_'+g:dict(available=tuple(n for n in GROUPS if n not in members),quality_mode='structural') for g,members in FAMILY_GROUPS.items()})
        results={name:w.predict(eb,ei,**common,**options) for name,options in definitions.items()}
        results.update(new_population=bank.predict(eb,ei,window_offsets=eo,user_id='Song'),
            new_uniform=dict(probabilities=np.mean(list(readout['probabilities'].values()),axis=0)),
            old_reliability=dict(probabilities=old_readouts[f'shots{shots}_seven_reliability_probabilities']),
            old_full=dict(probabilities=old_readouts[f'shots{shots}_seven_full_structural_probabilities']))
        assert len(results)==24
        for name,r in results.items():
            q=r['probabilities'];has_personal=name not in ('new_population','new_uniform')
            quality=r.get('quality_decision')
            unknown=np.zeros(len(evaluation),dtype=bool) if quality is None else quality['rejected']
            cells.append(dict(shots=shots,arm=name,long_term_calibration_trials=20 if has_personal else 0,
                current_calibration_trials=4*shots if has_personal else 0,unknown_trials=int(unknown.sum()),**score(ey,q)))
            arrays[f'shots{shots}_{name}']=q
            if name in definitions:
                arrays[f'shots{shots}_{name}_weights']=(np.tile(r['weights'],(len(q),1))
                    if quality is None else quality['provider_weights'])
                arrays[f'shots{shots}_{name}_rejected']=unknown
                for g,prob in r['decision_provider_probabilities'].items():
                    arrays[f'shots{shots}_{name}_provider_{g}']=prob
            for t,c,prob in zip(evaluation,ey,q):
                rows.append(dict(shots=shots,arm=name,trial_id=t,label=c,**{'p_'+g:float(v) for g,v in zip(CLASSES,prob)}))
        restored=load_document_workflow(package,policy,HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
        rp=restored.load_profile(pp,user_id='Song')
        rs=None if sp is None else restored.load_profile(sp,user_id='Song',session_id='S04',personal=rp)
        replay=restored.predict(eb,ei,**dict(common,personal=rp,session=rs),quality_mode='structural')
        error=float(np.max(abs(replay['probabilities']-results['new_full']['probabilities'])))
        assert error<=1e-12 and pickle.dumps((personal,session))==state_before
        roundtrips.append(dict(shots=shots,maximum_probability_error=error))
        print(f'{3+p["current_shots"].index(shots)}/6 Current{shots}shot:24 matched and removal cells complete',flush=True)
    assert pickle.dumps(w)==before and pickle.dumps((old_bank,gate))==old_before
    np.savez_compressed(OUT/'readouts.npz',**arrays)
    with (OUT/'predictions.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lookup={(c['shots'],c['arm']):c for c in cells}
    def guards(a,b):
        a,b=lookup[5,a],lookup[5,b]
        return dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],
            nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],nonworse_all_class_recall=all(b['recall'][c]>=a['recall'][c] for c in CLASSES))
    primary=guards('old_reliability','new_reliability')
    write(RESULT,dict(schema=SCHEMA,protocol_sha256=sha(PROTOCOL),policy_sha256=sha(policy),
        source_bank_sha256=sha(package),decision_policy_id=w.policy_id,source_dimensions=metadata,
        source_oof=folds,source_temperatures=temperatures,source_oof_losses=losses,population=population,
        inherited_n0=bank.policy_.n0,inherited_reliability_temperature=bank.policy_.temperature,
        evaluation_ids=evaluation,evaluation_labels=list(ey),personal_calibration_ids=previous['personal_calibration_ids'],
        reserved_current_ids=previous['reserved_current_ids'],profiles=profiles,cells=cells,roundtrips=roundtrips,
        primary_guards=primary,primary_pass=all(primary.values()),full_vs_old_guards=guards('old_full','new_full'),
        full_vs_reliability_guards=guards('new_reliability','new_full'),source_and_profiles_immutable=True,
        old_source_unchanged=True,default_promoted=False,physical_validation_proven=False,completion_proven=False,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        scope=p['scope'],artifact_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in OUT.iterdir() if v.is_file()}))
    print('Saved96 cells/11904 predictions; primary pass='+str(all(primary.values())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    prepare() if parser.parse_args().prepare else run()
