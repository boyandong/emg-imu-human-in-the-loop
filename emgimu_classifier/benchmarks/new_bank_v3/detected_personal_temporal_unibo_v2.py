"""New, frozen continuous personal/session fusion on unchanged native events.

Reuse completed detector and G5 readouts. Do not rerun their experiments or fit
any model. Remove entire current calibration recordings before every budget.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score
from emgimu.signal import polyphase_resample
from emgimu.feature_bank.continuous_calibration_isolation_v1 import (
    numbered_trial_intervals, projected_exclusion_mask, isolate_fixed_pairs,
)
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1, PersonalTemporalBoutsV1
from emgimu.feature_bank.unibo_study import _metrics
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import load, sha
from benchmarks.new_bank_v3.personal_temporal_unibo_v1 import path_and_signature, oracle_read, costs

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'DETECTED_PERSONAL_TEMPORAL_UNIBO_V2_PROTOCOL.json'
RESULT = HERE / 'DETECTED_PERSONAL_TEMPORAL_UNIBO_V2_RESULTS.json'
OUT = HERE / 'detected_personal_temporal_unibo_v2'
CLASSES = ('neutral', 'index_pinch', 'fist', 'open_hand')
MAP = {2: 2, 3: 1, 6: 3}
PARENTS = ['AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json',
           'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json', 'DETECTED_G5_UNIBO_V1_RESULTS.json',
           'PERSONAL_TEMPORAL_UNIBO_V1_PROTOCOL.json', 'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json',
           'F5_PATH_UNIBO_PROTOCOL.json']
SOURCES = ['src/emgimu/feature_bank/continuous_calibration_isolation_v1.py',
           'src/emgimu/feature_bank/personal_temporal_bouts_v1.py',
           'src/emgimu/feature_bank/document_path_v3.py', 'src/emgimu/feature_bank/temporal.py',
           'src/emgimu/feature_bank/core.py', 'src/emgimu/feature_bank/unibo_study.py',
           'src/emgimu/feature_bank/screening.py', 'src/emgimu/signal.py',
           'src/emgimu/datasets/adapters/unibo_inail.py',
           'benchmarks/new_bank_v3/autonomous_continuous_unibo_v1.py',
           'benchmarks/new_bank_v3/personal_temporal_unibo_v1.py',
           'benchmarks/new_bank_v3/detected_personal_temporal_unibo_v2.py',
           'tests/test_continuous_calibration_isolation_v1.py',
           'tests/test_detected_personal_temporal_scoring_v1.py',
           'tests/test_continuous_calibration_precision_v2.py',
           'benchmarks/new_bank_v3/detected_personal_temporal_unibo_v1.py']


def prepare():
    temporal = json.loads((HERE / PARENTS[4]).read_text())
    boundary = json.loads((HERE / PARENTS[0]).read_text())
    source = json.loads((HERE / PARENTS[5]).read_text())
    external = {str(Path(boundary['archive'])): boundary['archive_sha256'],
                str(Path(source['frozen_source']) / 'fitted_states.pkl'):
                    source['frozen_files']['source/fitted_states.pkl']}
    for name, digest in temporal['artifact_sha256'].items():
        if '_Day6_' in name or '_personal.zip' in name or '_config.json' in name:
            assert sha(ROOT / name) == digest
            external[str(ROOT / name)] = digest
    manifest = Path(source['dataset_root']) / 'manifest.json'
    external[str(manifest)] = sha(manifest)
    protocol = dict(schema='detected_personal_temporal_unibo_v2_protocol',
        parent_sha256={name: sha(HERE / name) for name in PARENTS},
        source_sha256={name: sha(ROOT / name) for name in SOURCES}, external_sha256=external,
        current_shots=[0, 1, 2, 5], source_rate_hz=500., target_rate_hz=200.,
        guard_source_samples=100, classes=CLASSES, supported_native_label_map=MAP,
        selection='Reuse all previously saved Day6 detected bounds, one-to-one IoU>=.5 matches and G5 detected/oracle probabilities. No detector replay, classifier fit, profile enrollment, parameter search, threshold change or rematching.',
        isolation='Reuse the exact20 reserved current calibration recording IDs/user from the completed temporal study at EVERY budget including0. Recover entire numbered source recordings using gestureCounter runs and original label, including their Rest edges. Verify each reserved cropped/resampled recording bitwise against its processed NPZ and frozen raw SHA. Guard both edges by100 native samples(.2s), exceeding the pinned resampling impulse support. Exclude any intersecting detected/reference interval and both endpoints of its previously matched pair. Do not recut, rematch or remove eligible unmatched references. All budgets use identical retained axes.',
        preprocessing='Existing personal/session profiles use independently cropped500->200Hz records; evaluation uses unchanged full-recording500->200Hz resampling from the completed boundary/G5 chain. Same resampler/channel semantics, different edge context is disclosed. Guard prevents calibration samples entering retained evaluation through resampling. No claim that cropped and continuous boundary context are identical.',
        inference='Load exact previous long-term20 and nested current0/4/8/20 trial profiles. Estimated native intervals remain estimated and decisions follow interval end. Full32-bin RMS paths, band3 DTW and order1/2 signatures use every interval sample. Fixed eleven-arm .75/.125/.125 fusion; source and profiles immutable. Independent loop paths/signatures/DP/probability oracles check every new prediction.',
        controls='Two modes: all retained detected events (unsupported and unmatched kept), and correct boundaries for the SAME supported matched references only. Oracle comparison does not add missed references or use ground-truth boundaries for detected inference. Preserve base, uniform confidence softening, long/local and branch removal controls.',
        weighting='Within each user, supported matched active classes1/2/3 have equal mass, and matched references within class equal mass; pooled conditional metrics give each user equal mass. Four-class probability loss/Brier retained, active macroF1 explicitly labels1/2/3. No Rest truth examples, so no full-stream or Rest accuracy claim. Unweighted correct/all eligible supported references includes detector misses; report per-class counts/recall and equal-user success separately. Unsupported/unmatched event predictions retained but have no invented gesture truth.',
        primary='Previously inspected Day6 five-shot detected base_full vs base: lower equal-user/class logloss and Brier, nonworse active macroF1 and unweighted supported-reference end-to-end success, at least5/7 user loss wins, and lower loss than fixed uniform-softening control. All six required. All budgets and oracle controls remain descriptive; no tuning on these outcomes.',
        scope='Retrospective public four-channel200Hz continuous diagnostic with original Day1 source-Rest boundaries and personalized Days1-5 G5 history. Calibration and evaluation use disjoint regions of the SAME Day6 continuous files, not independent-day files. No physical8-channel250Hz or current-Rest GUI-detector efficacy, physiological onset, prospective validation or default promotion.',
        versions=dict(numpy=np.__version__, scipy=__import__('scipy').__version__),
        supersedes_precheck_abort='V1 stopped before output-directory creation and before any new prediction: missing float64 cast in reserved-record reconstruction. V2 matches the frozen converter cast; policy/splits/weights unchanged.',
        default_promoted=False)
    with PROTOCOL.open('x', encoding='utf8', newline='\n') as f:
        json.dump(protocol, f, indent=2); f.write('\n')


def conditional_weights(labels):
    y = np.asarray(labels)
    if y.ndim != 1 or set(y) != {1, 2, 3}:
        raise ValueError('All three supported active classes required for matched weighting')
    return np.array([1. / (3 * np.count_nonzero(y == label)) for label in y])


def score(labels, probabilities, weights):
    y = np.asarray(labels)
    q = np.asarray(probabilities)
    w = np.asarray(weights)
    if (not len(y) or q.shape != (len(y), 4) or w.shape != y.shape or not np.isfinite(q).all()
            or np.any(q < 0) or not np.allclose(q.sum(1), 1., rtol=0, atol=1e-12)
            or not np.isfinite(w).all() or np.any(w <= 0) or not set(y) <= {1, 2, 3}):
        raise ValueError('Explicit active truth, valid four-class probabilities and positive weights required')
    result = _metrics(y, q, w)
    result['active_macro_f1'] = float(f1_score(y, q.argmax(1), labels=[1, 2, 3],
        average='macro', sample_weight=w, zero_division=0))
    result['predicted_rest_count'] = int(np.count_nonzero(q.argmax(1) == 0))
    return result


def end_to_end(labels, probabilities, reference_labels):
    """Counts include all eligible supported references, also unmatched ones."""
    y = np.asarray(labels); q = np.asarray(probabilities); refs = np.asarray(reference_labels)
    if q.shape != (len(y), 4) or not set(y) <= {1, 2, 3} or not set(refs) <= {1, 2, 3}:
        raise ValueError('Supported active matched and reference axes required')
    prediction = q.argmax(1)
    per_class = {}
    for label in (1, 2, 3):
        count = int(np.count_nonzero(refs == label)); matched = int(np.count_nonzero(y == label))
        if not count or matched > count:
            raise ValueError('Matched count cannot exceed positive reference count')
        correct = int(np.count_nonzero((y == label) & (prediction == label)))
        per_class[CLASSES[label]] = dict(references=count, matched=matched, misses=count-matched,
            correct=correct, success=correct/count)
    correct = int(np.count_nonzero(prediction == y))
    return dict(supported_references=len(refs), supported_matched=len(y), correct=correct,
        success=correct/len(refs), per_class=per_class)


def run():
    if RESULT.exists() or OUT.exists():
        raise FileExistsError('Experiment already exists; never overwrite/rerun')
    p = json.loads(PROTOCOL.read_text())
    assert p['versions'] == dict(numpy=np.__version__, scipy=__import__('scipy').__version__)
    for names, root in [(p['parent_sha256'], HERE), (p['source_sha256'], ROOT)]:
        for name, digest in names.items():
            assert sha(root / name) == digest, name
    for name, digest in p['external_sha256'].items():
        assert sha(Path(name)) == digest, name
    boundary = json.loads((HERE / PARENTS[1]).read_text())
    bp = json.loads((HERE / PARENTS[0]).read_text())
    prior = json.loads((HERE / PARENTS[4]).read_text())
    g5 = json.loads((HERE / PARENTS[2]).read_text())
    parent = json.loads((HERE / PARENTS[5]).read_text())
    dataset = Path(parent['dataset_root'])
    trials = {f.stem: f for f in (dataset / 'trials').rglob('*.npz')}
    metadata = {}
    for folder in ('frozen_source', 'frozen_final'):
        for r in json.loads((Path(parent[folder]) / 'bout_metadata.json').read_text()):
            metadata[r['trial']] = r['raw_sha256']
    profiles = {r['user']: r for r in prior['blocks'] if r['day'] == 6 and r['shots'] == 0}
    prior_blocks = {(r['user'], r['shots']): r for r in prior['blocks'] if r['day'] == 6}
    old_events = {(r['member'], r['event_index']): r for r in g5['events']}
    arrays = {}; recordings = []; events = []; references = []; raw = {}; oracle_raw = {}; all_reserved = set()
    print('1/3 Recover numbered calibration regions; verify unchanged continuous event axes', flush=True)
    with zipfile.ZipFile(bp['archive']) as archive:
        for recording in boundary['recordings']:
            name = recording['member']; identity = bp['members'][name]; user = f"u{identity['user']:02d}"
            x, _, _, digest = load(archive, name)
            assert digest == recording['member_sha256']
            original = loadmat(io.BytesIO(archive.read(name)), variable_names=('emg', 'label', 'gestureCounter'))
            bounds = numbered_trial_intervals(np.asarray(original['gestureCounter']).ravel(),
                np.asarray(original['label']).ravel(), user=identity['user'], day=6, posture=identity['posture'])
            reserved = sorted(set(profiles[user]['reserved_recordings']) & set(bounds))
            regions = []
            for trial in reserved:
                a, b = bounds[trial]; path = trials[trial]
                assert sha(path) == metadata[trial]
                with np.load(path, allow_pickle=False) as z:
                    expected = polyphase_resample(np.asarray(original['emg'][a:b], dtype=np.float64), 500., 200.).astype(np.float32)
                    np.testing.assert_array_equal(expected, z['emg'])
                    assert str(z['trial_id']) == trial
                regions.append(dict(trial=trial, source_start=a, source_end=b, processed_sha256=sha(path)))
                all_reserved.add(trial)
            mask = projected_exclusion_mask(len(x), [(r['source_start'], r['source_end']) for r in regions],
                source_rate=500., target_rate=200., guard_source_samples=p['guard_source_samples'])
            # Independent finite-support check: unit input in excluded raw regions
            # must have no resampled contribution outside the guarded output mask.
            impulse = np.zeros(len(original['emg']))
            for r in regions:
                impulse[r['source_start']:r['source_end']] = 1.
            assert not np.any(polyphase_resample(impulse, 500., 200.)[~mask])
            refs = recording['reference_bounds']; det = recording['detected_bounds']
            matches = [(m['event_index'], m['reference_index']) for m in recording['matches']]
            isolated = isolate_fixed_pairs(det, [(r[1], r[2]) for r in refs], matches, mask)
            matched = dict(isolated['retained_pairs'])
            record = dict(user=user, member=name, member_sha256=digest, source_samples=len(original['emg']),
                target_samples=len(x), reserved_regions=regions, excluded_target_samples=int(mask.sum()),
                **isolated, onset_errors_s=[], offset_errors_s=[])
            for m in recording['matches']:
                if (m['event_index'], m['reference_index']) in isolated['retained_pairs']:
                    record['onset_errors_s'].append(m['onset_error_s'])
                    record['offset_errors_s'].append(m['offset_error_s'])
            recordings.append(record)
            for j in isolated['retained_references']:
                label, a, b = refs[j]
                references.append(dict(id=f'{name}:reference:{j}', user=user, member=name,
                    reference_index=j, start=a, end=b, native_label=label, label=MAP.get(label)))
            for i in isolated['retained_events']:
                old = old_events[name, i]; a, b = det[i]
                assert (a, b) == (old['start'], old['end']) and old['user'] == user
                j = matched.get(i); label = None if j is None else MAP.get(refs[j][0])
                assert label == old['reference_label']
                event_id = f'{name}:event:{i}'
                events.append(dict(id=event_id, user=user, member=name, event_index=i, start=a, end=b,
                    reference_index=j, reference_label=label, status='unmatched' if j is None else
                    'unsupported_matched' if label is None else 'supported_matched'))
                raw[event_id] = x[a:b].copy()
                if label is not None:
                    assert refs[j][1:] == [old['reference_start'], old['reference_end']]
                    oracle_raw[event_id] = x[refs[j][1]:refs[j][2]].copy()
    assert all_reserved == {i for r in profiles.values() for i in r['reserved_recordings']}
    assert len(all_reserved) == 140
    OUT.mkdir(); blocks = []; aggregate = {}; max_error = 0.
    for user in sorted(profiles):
        selected = [e for e in events if e['user'] == user]
        supported = [e for e in selected if e['reference_label'] is not None]
        user_refs = [r for r in references if r['user'] == user and r['label'] is not None]
        labels = np.array([e['reference_label'] for e in supported]); weights = conditional_weights(labels)
        matched_positions = [i for i, e in enumerate(selected) if e['reference_label'] is not None]
        config = json.loads((HERE / 'personal_temporal_unibo_v1' / (user + '_config.json')).read_text())
        workflow = PersonalTemporalBoutsV1(**config)
        personal = workflow.load_profile(ROOT / profiles[user]['personal_profile'], user_id=user)
        frozen = pickle.dumps((workflow, personal))
        for mode, rows, source_raw, kind in [('detected', selected, raw, 'estimated'),
                ('matched_oracle', supported, oracle_raw, 'complete_cued')]:
            batch = TemporalBoutBatchV1(tuple(source_raw[e['id']] for e in rows),
                tuple(f"{mode}:{e['id']}" for e in rows), tuple(e['member'] for e in rows),
                tuple(e['start'] if mode == 'detected' else old_events[e['member'], e['event_index']]['reference_start']
                      for e in rows), 200., tuple(config['channel_ids']), config['preprocessing_id'], kind)
            q = np.stack([old_events[e['member'], e['event_index']][
                'g5_probability' if mode == 'detected' else 'g5_oracle_probability'] for e in rows])
            paths, signatures = path_and_signature(batch)
            long_dtw, long_sig = oracle_read(paths, signatures, personal, workflow.band)
            for shots in p['current_shots']:
                block = prior_blocks[user, shots]
                assert block['reserved_recordings'] == profiles[user]['reserved_recordings']
                session = None if block['session_profile'] is None else workflow.load_profile(
                    ROOT / block['session_profile'], user_id=user)
                session_bytes = pickle.dumps(session)
                result = workflow.predict(batch, personal=personal, session=session, user_id=user,
                    session_id='Day6', base_probabilities=q, base_trial_ids=batch.trial_ids, base_class_names=CLASSES)
                assert result['certified_full_coverage'] == (mode == 'matched_oracle')
                local_dtw, local_sig = (long_dtw, long_sig) if session is None else oracle_read(
                    paths, signatures, session, workflow.band)
                expected = dict(DTW_long=long_dtw, signature_long=long_sig, DTW_local=local_dtw,
                    DTW_blended=(long_dtw+local_dtw)/2, signature_blended=(long_sig+local_sig)/2, base=q)
                for arm in ('DTW_long', 'DTW_blended', 'signature_blended'):
                    expected['base_'+arm] = .75*q + .25*expected[arm]
                expected['base_full'] = .75*q + .125*(expected['DTW_blended']+expected['signature_blended'])
                expected['base_uniform'] = .75*q + .25/4
                key = f'{user}_{mode}_s{shots}'; scores = {}; successes = {}
                for arm, values in result['arms'].items():
                    error = float(np.max(abs(values-expected[arm]))); max_error = max(max_error, error)
                    assert error <= 1e-10, (key, arm, error)
                    arrays[key+'_'+arm] = values
                    matched_q = values[matched_positions] if mode == 'detected' else values
                    scores[arm] = score(labels, matched_q, weights)
                    if mode == 'detected':
                        successes[arm] = end_to_end(labels, matched_q, [r['label'] for r in user_refs])
                    aggregate.setdefault((mode, shots, arm), []).append((user, labels, matched_q, weights,
                        [r['label'] for r in user_refs]))
                assert pickle.dumps((workflow, personal)) == frozen and pickle.dumps(session) == session_bytes
                blocks.append(dict(key=key, user=user, mode=mode, shots=shots,
                    event_ids=[e['id'] for e in rows], matched_event_ids=[e['id'] for e in supported],
                    matched_positions=matched_positions if mode == 'detected' else list(range(len(supported))),
                    labels=labels.tolist(), weights=weights.tolist(), supported_reference_ids=[r['id'] for r in user_refs],
                    personal_profile=profiles[user]['personal_profile'], session_profile=block['session_profile'],
                    reserved_recordings=block['reserved_recordings'], calibration_ids=block['calibration_ids'],
                    predictive_calibration_cost={arm:dict(long_term=costs(arm, shots)[0], current=costs(arm, shots)[1])
                                                for arm in scores}, scores=scores, end_to_end=successes))
        print(f'2/3 {user}: {len(selected)} retained detections, {len(supported)} supported matches, four budgets verified', flush=True)
    pooled = []
    for (mode, shots, arm), entries in aggregate.items():
        y = np.concatenate([v[1] for v in entries]); q = np.concatenate([v[2] for v in entries])
        weights = np.concatenate([v[3]/len(entries) for v in entries])
        r = dict(mode=mode, shots=shots, arm=arm, **score(y, q, weights))
        if mode == 'detected':
            r['end_to_end'] = end_to_end(y, q, [label for v in entries for label in v[4]])
            r['equal_user_end_to_end_success'] = float(np.mean([end_to_end(v[1], v[2], v[4])['success'] for v in entries]))
        pooled.append(r)
    lookup = {(r['mode'], r['shots'], r['arm']):r for r in pooled}
    base = lookup['detected', 5, 'base']; full = lookup['detected', 5, 'base_full']
    wins = sum(r['scores']['base_full']['log_loss'] < r['scores']['base']['log_loss']
        for r in blocks if r['mode'] == 'detected' and r['shots'] == 5)
    guards = dict(lower_log_loss=full['log_loss'] < base['log_loss'], lower_brier=full['brier'] < base['brier'],
        nonworse_active_macro_f1=full['active_macro_f1'] >= base['active_macro_f1'],
        nonworse_end_to_end_success=full['end_to_end']['success'] >= base['end_to_end']['success'],
        at_least5_of7_user_loss_wins=wins >= 5,
        lower_loss_than_uniform=full['log_loss'] < lookup['detected', 5, 'base_uniform']['log_loss'])
    with (OUT / 'readouts.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    with (OUT / 'intervals.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(dict(recordings=recordings, events=events, references=references), f, indent=2); f.write('\n')
    onsets = [v for r in recordings for v in r['onset_errors_s']]
    offsets = [v for r in recordings for v in r['offset_errors_s']]
    for name, digest in p['external_sha256'].items():
        if name != bp['archive']:
            assert sha(Path(name)) == digest, name
    result = dict(schema='detected_personal_temporal_unibo_v2', protocol_sha256=sha(PROTOCOL),
        artifact_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in OUT.iterdir()}, blocks=blocks, pooled=pooled,
        totals=dict(original_detections=len(g5['events']), retained_detections=len(events),
            retained_references=len(references), retained_matches=len(onsets),
            supported_references=sum(r['label'] is not None for r in references),
            supported_matched=sum(e['reference_label'] is not None for e in events),
            unsupported_matched=sum(e['status'] == 'unsupported_matched' for e in events),
            unmatched_detections=sum(e['status'] == 'unmatched' for e in events), excluded_calibration_recordings=len(all_reserved),
            onset_mae_s=float(np.mean(np.abs(onsets))), offset_mae_s=float(np.mean(np.abs(offsets)))),
        primary_guards=guards, primary_pass=all(guards.values()), primary_user_loss_wins=wins,
        independent_DP_path_signature_fusion_max_error=max_error, source_and_profiles_immutable=True,
        original_boundary_and_G5_experiments_rerun=False, classifier_or_profiles_fitted=False,
        all_budgets_identical_evaluation_axis=True, default_promoted=False, physical_validation_proven=False,
        completion_proven=False, scope=p['scope'])
    with RESULT.open('x', encoding='utf8', newline='\n') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('3/3 '+json.dumps(dict(totals=result['totals'], primary_guards=guards, user_loss_wins=wins)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
