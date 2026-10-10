"""Source-only CSP addition, context-only F4d and matched seven-provider ablations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from benchmarks.song_real8_study import load_session
from benchmarks.song_real8.song_personal_session_workflow_v1 import aggregate, windows, score, CHANNELS, PREPROCESSING
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v2 import load_gate
from emgimu.feature_bank.personal_session_decision_v1 import PersonalSessionDecisionV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.force_nested_oof import fit_temperature, temperature_probability
from emgimu.feature_bank.extended_window_decision_v1 import SourceCsp250V1, CspQualityGateV1, ExtendedWindowDecisionV1

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'SONG_EXTENDED_WINDOW_V1_PROTOCOL.json'
RESULT = HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json'
OUT = HERE/'song_extended_window_v1'
GROUPS = ('F0', 'F1', 'F2ac', 'F3b', 'F4abc', 'F5window', 'F2b')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent = json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    names = ['src/emgimu/feature_bank/extended_window_decision_v1.py',
        'src/emgimu/feature_bank/document_signal.py', 'src/emgimu/feature_bank/relative_spectrum.py',
        'tests/test_extended_window_decision_v1.py', 'benchmarks/song_real8/song_extended_window_v1.py',
        'benchmarks/song_real8/song_raw_quality_v1.py']
    sources = dict(parent['source_sha256'])
    sources.update({n: sha(ROOT/n) for n in names})
    artifacts = dict(parent['artifact_sha256'])
    for n in ['benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_RESULTS.json',
              'feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json']:
        artifacts[n] = sha(ROOT/n)
    p = dict(schema='song_extended_window_v1', source_folder=parent['source_folder'], hdf5_sha256=parent['hdf5_sha256'],
        source_sha256=sources, artifact_sha256=artifacts, groups=GROUPS, source_seed=20261011,
        current_shots=[0, 1, 2, 5], anchor_mix=.5,
        source_rule='Retain all six original families/scalers/classifiers/temperatures byte-identical. Fit only new CSP in each leave-source-recording-out fold and joined S01/S02. Source OOF CSP temperature; new seven-prior=softmax(-pooled calibrated source OOF logloss). Existing n0/reliability temperature fixed, no new target search.',
        CSP='Uncentered XX^T/(trace+1e-10); four one-vs-rest classes; top2+bottom2, gamma1e-5; log normalized projected variance,16 coordinates. Fold trial provenance and250Hz/50sample/8channel contract are explicit.',
        F4d='Source-frozen F4 bands; log demeaned-Hann rFFT band energy/T plus1e-10. Equal native-trial reference mass. Window-minus-long and current-reference-minus-long are context only, never a classifier or fatigue label.',
        split='Same previously inspected S03 long20, nested S04 current0/4/8/20 and same124 evaluation trials. Reuse exact frozen identities; no evaluation updates. Old six-provider reliability and F7/F8 controls must reproduce previous probabilities.',
        quality='Source-frozen gate references; new policy uses min channel quality for F0/F2ac/CSP, mean for other branches, any-channel structural Unknown. No target threshold fitting.',
        primary='Five-shot seven reliability versus old six reliability on same124 trials: lower logloss/Brier, nonworse macroF1 and all class recalls. Separately preserve full F7/F8 comparison,19 arms/budget, all seven frozen-weight provider removals and branch removals.',
        scope='Seven declared window groups/281coordinates plus context-only F4d. Not every F0-F9 subfamily, full-bout F5, physical ring/F6, unseen-person/day, hardware or deployment efficacy. Retrospective source OOF temperatures/prior reuse is not unbiased source performance. No default promotion.',
        default_promoted=False)
    with PROTOCOL.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(p, stream, indent=2)
        stream.write('\n')


def csp_fit(item, seed):
    family = SourceCsp250V1().fit(item['batch'], item['hand'], source_trial_ids=item['trial'])
    x, y, ids = aggregate(family.transform(item['batch']), item)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=seed).fit(scaler.transform(x), y)
    if model.n_iter_.max() >= 2000:
        raise ValueError('CSP source model did not converge')
    return family, scaler, model


def load_extended(package, policy, gate_package, gate_results):
    config = json.loads(Path(policy).read_text(encoding='utf8'))
    payload = Path(package).read_bytes()
    if hashlib.sha256(payload).hexdigest() != config['source_bank_sha256']:
        raise ValueError('Extended source package checksum differs')
    bank = pickle.loads(payload)
    if not isinstance(bank, FrozenEmgProviderBankV1) or bank.bank_id_ != config['source_bank_id']:
        raise ValueError('Extended source bank identity differs')
    base = PersonalSessionWorkflowV1(bank, channel_ids=CHANNELS, preprocessing_id=PREPROCESSING,
        rest_label='neutral', quality_options={'line_frequency_hz':50, 'pre_highpass_available':False})
    w = ExtendedWindowDecisionV1(base, CspQualityGateV1(load_gate(gate_package, gate_results)), anchor_mix=config['anchor_mix'])
    if w.policy_id != config['decision_policy_id']:
        raise ValueError('Extended decision policy identity differs')
    return w


def run():
    if RESULT.exists() or OUT.exists():
        raise FileExistsError('Existing native experiment must not be rerun/overwritten')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for n, digest in {**p['source_sha256'], **p['artifact_sha256']}.items():
        assert sha(ROOT/n) == digest, n
    prior = json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    previous = json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json').read_text(encoding='utf8'))
    old_base = load_workflow(HERE/'song_personal_session_v1/source_bank.pkl', ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    original_gate = load_gate(HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    old = PersonalSessionDecisionV1(old_base, gate=original_gate)
    historical_before = pickle.dumps((old_base, original_gate))
    def native(session):
        item = load_session(Path(p['source_folder'])/f'2026-09-18_{session}', session, filter_mode='causal')
        assert item['audit']['sha256'] == p['hdf5_sha256'][session]
        item['batch'] = FeatureBatch(item['batch'].emg, 250.)
        return item
    print('1/4 New CSP source-only fits; six historical models retained', flush=True)
    source = {s: native(s) for s in ('S01', 'S02')}
    folds = []
    for saved in prior['source_oof']:
        held = saved['session']
        train = source['S02' if held == 'S01' else 'S01']
        f, scaler, model = csp_fit(train, p['source_seed'])
        x, y, ids = aggregate(f.transform(source[held]['batch']), source[held])
        assert list(ids) == saved['validation_ids'] and list(y) == saved['labels']
        assert list(f.source_trial_ids_) == saved['fit_ids']
        folds.append(dict(session=held, fit_ids=saved['fit_ids'], ids=list(ids), labels=list(y), raw=model.predict_proba(scaler.transform(x))))
    y = np.concatenate([np.array(f['labels']) for f in folds])
    encoded = np.array([old.bank.classes_.index(c) for c in y])
    temperature = fit_temperature(np.concatenate([f['raw'] for f in folds]), encoded)
    for f in folds:
        f['calibrated'] = temperature_probability(f['raw'], temperature)
    losses = {g: score(y, np.concatenate([np.asarray(f['calibrated_probabilities'][g]) for f in prior['source_oof']]))['log_loss'] for g in old.bank.providers_}
    losses['F2b'] = score(y, np.concatenate([f['calibrated'] for f in folds]))['log_loss']
    joined = dict(batch=FeatureBatch(np.concatenate([v['batch'].emg for v in source.values()]),250.),
        trial=np.concatenate([v['trial'] for v in source.values()]), hand=np.concatenate([v['hand'] for v in source.values()]))
    family, scaler, model = csp_fit(joined, p['source_seed'])
    families, models, temperatures = dict(old.bank.families_), dict(old.bank.models_), dict(old.bank.temperatures_)
    families['F2b'], models['F2b'], temperatures['F2b'] = [family], (scaler, model), temperature
    population = softmax(-np.array([losses[g] for g in GROUPS]))
    bank = FrozenEmgProviderBankV1(families, models, temperatures, classes=old.bank.classes_, class_names=old.bank.class_names_,
        source_trial_ids=old.bank.policy_.source_trials, source_policy_id=sha(PROTOCOL), sample_rate_hz=250., window_samples=50, channels=8,
        population=population, n0=old.bank.policy_.n0, reliability_temperature=old.bank.policy_.temperature)
    for g in old.bank.providers_:
        assert pickle.dumps((bank.families_[g], bank.models_[g], bank.temperatures_[g])) == pickle.dumps((old.bank.families_[g], old.bank.models_[g], old.bank.temperatures_[g]))
    base = PersonalSessionWorkflowV1(bank, channel_ids=CHANNELS, preprocessing_id=PREPROCESSING, rest_label='neutral', quality_options=old_base.quality_options)
    w = ExtendedWindowDecisionV1(base, CspQualityGateV1(original_gate))
    before = pickle.dumps(w)
    OUT.mkdir()
    package = OUT/'source_bank.pkl'
    package.write_bytes(pickle.dumps(bank))
    policy = OUT/'policy.json'
    policy.write_text(json.dumps(dict(source_bank_sha256=sha(package), source_bank_id=bank.bank_id_, decision_policy_id=w.policy_id,
        anchor_mix=.5, gate_policy_id=w.gate.policy_id), indent=2)+'\n', encoding='utf8', newline='\n')
    print('2/4 Frozen long/current calibration and same124 evaluation trials', flush=True)
    long, current = native('S03'), native('S04')
    evaluation_ids = previous['evaluation_ids']
    eb, ei, eo = windows(current, evaluation_ids)
    labels = dict(zip(current['trial'], current['hand']))
    ey = np.array([labels[t] for t in evaluation_ids])
    assert list(ey) == previous['evaluation_labels']
    b, ids, offsets = windows(long, previous['personal_calibration_ids'])
    ly = dict(zip(long['trial'], long['hand']))
    kwargs = dict(user_id='Song', observed_channel_ids=CHANNELS, preprocessing_id=PREPROCESSING)
    personal = w.enroll_user(b, ids, {t:ly[t] for t in np.unique(ids)}, window_offsets=offsets, session_id='S03',
        forbidden_evaluation_trials=evaluation_ids, **kwargs)
    pp = OUT/'personal_S03.zip'
    w.save_profile(personal, pp)
    old_personal = old.load_profile(HERE/'song_integrated_decision_v1/personal_S03.zip', user_id='Song')
    # The raw companion uses the existing independently audited native extractor.
    from benchmarks.song_real8.song_raw_quality_v1 import native as raw_native
    raw_item, _, _, raw_windows = raw_native('S04', p)
    assert np.array_equal(raw_item['trial'], current['trial'])
    mask = np.isin(current['trial'], evaluation_ids)
    raw_batch = raw_windows.take(np.flatnonzero(mask))
    cells, rows, arrays, profiles, roundtrips = [], [], {}, {}, []
    for shots in p['current_shots']:
        session = old_session = None
        calibration_ids = [] if not shots else previous['profiles'][str(shots)]['calibration_ids']
        if shots:
            cb, ci, co = windows(current, calibration_ids)
            session = w.calibrate_session(cb, ci, {t:labels[t] for t in calibration_ids}, window_offsets=co,
                personal=personal, session_id='S04', forbidden_evaluation_trials=evaluation_ids, **kwargs)
            sp = OUT/f'session_S04_{shots}shot.zip'
            w.save_profile(session, sp)
            old_session = old.load_profile(HERE/f'song_integrated_decision_v1/session_S04_{shots}shot.zip',
                user_id='Song', session_id='S04', personal=old_personal)
        profiles[str(shots)] = dict(calibration_ids=calibration_ids, path=None if not shots else sp.relative_to(ROOT).as_posix())
        state_before = pickle.dumps((personal, session))
        common = dict(window_offsets=eo, personal=personal, session=session, session_id='S04', **kwargs)
        definitions = {'seven_reliability':dict(use_anchor=False,use_session_routing=False),
            'seven_F8':dict(use_anchor=False),'seven_F7_F8':{},'seven_full_structural':dict(quality_mode='structural'),
            'seven_minus_F7':dict(use_anchor=False,quality_mode='structural'),
            'seven_minus_F8':dict(use_session_routing=False,quality_mode='structural'),'seven_minus_F9':{}}
        definitions.update({'full_minus_'+g:dict(available=tuple(n for n in GROUPS if n != g), quality_mode='structural') for g in GROUPS})
        results = {name:w.predict(eb,ei,raw_batch=raw_batch,**common,**opts) for name,opts in definitions.items()}
        for name, anchor in [('old_six_baseline',False),('old_six_F7_F8',True)]:
            results[name] = old.predict(eb,ei,window_offsets=eo,personal=old_personal,session=old_session,
                use_anchor=anchor,use_session_routing=anchor,session_id='S04',**kwargs)
        native = bank.predict_providers(eb,ei,window_offsets=eo,user_id='Song')
        results.update(seven_population=bank.predict(eb,ei,window_offsets=eo,user_id='Song'),
            seven_uniform=dict(probabilities=np.mean(list(native['probabilities'].values()),axis=0)),
            CSP_only=dict(probabilities=native['probabilities']['F2b']))
        assert len(results) == 19
        for name,r in results.items():
            q = r['probabilities']
            cost = name not in ('seven_population','seven_uniform','CSP_only')
            cells.append(dict(shots=shots, arm=name, long_term_calibration_trials=20 if cost else 0,
                current_calibration_trials=4*shots if cost else 0, **score(ey,q)))
            arrays[f'shots{shots}_{name}_probabilities'] = q
            for t,y0,prob in zip(evaluation_ids,ey,q):
                rows.append(dict(shots=shots,arm=name,trial_id=t,label=y0,**{'p_'+c:float(prob[i]) for i,c in enumerate(bank.classes_)}))
        c = results['seven_F7_F8']['spectral_context']
        arrays[f'shots{shots}_spectral_window_minus_long'] = c['window_minus_long']
        arrays[f'shots{shots}_spectral_trial_minus_long'] = c['trial_minus_long']
        if shots:
            arrays[f'shots{shots}_spectral_session_minus_long'] = c['session_minus_long']
        restored = w.load_profile(pp,user_id='Song')
        current_profile = None if not shots else w.load_profile(sp,user_id='Song',session_id='S04',personal=restored)
        replay = w.predict(eb,ei,window_offsets=eo,personal=restored,session=current_profile,session_id='S04',**kwargs)
        error = float(np.max(abs(replay['probabilities']-results['seven_F7_F8']['probabilities'])))
        assert error <= 1e-12 and pickle.dumps((personal,session)) == state_before
        roundtrips.append(dict(shots=shots,maximum_probability_error=error))
        print(f'3/4 Current{shots}shot:19 paired/control/removal arms complete',flush=True)
    assert pickle.dumps(w) == before and pickle.dumps((old_base, original_gate)) == historical_before
    with (OUT/'predictions.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'readouts.npz',**arrays)
    lookup={(c['shots'],c['arm']):c for c in cells}
    def guards(first,second):
        a,b=lookup[(5,first)],lookup[(5,second)]
        return dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],
            nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],nonworse_all_class_recall=all(b['recall'][c]>=a['recall'][c] for c in bank.classes_))
    primary=guards('old_six_baseline','seven_reliability')
    result=dict(schema=p['schema'],protocol_sha256=sha(PROTOCOL),policy_path=policy.relative_to(ROOT).as_posix(),
        policy_sha256=sha(policy),source_bank_path=package.relative_to(ROOT).as_posix(),source_bank_sha256=sha(package),
        decision_policy_id=w.policy_id,source_dimensions={g:len(bank.models_[g][0].mean_) for g in GROUPS},
        source_oof=folds,source_oof_losses=losses,source_temperature=temperature,source_population=population,
        evaluation_ids=evaluation_ids,evaluation_labels=list(ey),personal_calibration_ids=previous['personal_calibration_ids'],
        reserved_current_ids=previous['reserved_current_ids'],profiles=profiles,cells=cells,roundtrips=roundtrips,
        primary_guards=primary,primary_pass=all(primary.values()),full_anchor_guards=guards('old_six_F7_F8','seven_full_structural'),
        source_and_profiles_immutable=True,six_historical_source_models_unchanged=True,F4d_context_only=True,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=p['scope'],
        artifact_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in OUT.iterdir() if v.is_file()})
    RESULT.write_text(json.dumps(result,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n',encoding='utf8',newline='\n')
    print('4/4 Preserved76 cells/9424 predictions; primary pass='+str(result['primary_pass']),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
