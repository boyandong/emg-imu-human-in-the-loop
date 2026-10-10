"""New registered-Rest detector comparison with frozen classifiers/profiles.

User-level checkpoints are immutable. Resume skips completed users; completed
native experiments and their predictions are never rerun or replaced.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.feature_bank.autonomous_bouts_v1 import DetectedBout
from emgimu.feature_bank.calibration_rest_detector_v1 import fit_calibration_rest_detector
from emgimu.feature_bank.continuous_calibration_isolation_v1 import projected_exclusion_mask
from emgimu.feature_bank.detected_g5_reader_v1 import DetectedG5ReaderV1
from emgimu.feature_bank.personal_temporal_bouts_v1 import PersonalTemporalBoutsV1, TemporalBoutBatchV1
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import load, sha, score as match
from benchmarks.new_bank_v3.personal_temporal_unibo_v1 import path_and_signature, oracle_read, costs
from benchmarks.new_bank_v3.detected_personal_temporal_unibo_v3 import conditional_weights, score, end_to_end

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'CALIBRATION_REST_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
RESULT = HERE/'CALIBRATION_REST_CONTINUOUS_UNIBO_V1_RESULTS.json'
OUT = HERE/'calibration_rest_continuous_unibo_v1'
CLASSES = ('neutral', 'index_pinch', 'fist', 'open_hand')
ARMS = ('DTW_long', 'signature_long', 'DTW_local', 'DTW_blended', 'signature_blended',
        'base', 'base_DTW_long', 'base_DTW_blended', 'base_signature_blended', 'base_full', 'base_uniform')
PARENTS = ['F5_PATH_UNIBO_PROTOCOL.json', 'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json',
    'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json', 'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json',
    'detected_personal_temporal_unibo_v3/intervals.json', 'detected_personal_temporal_unibo_v3/readouts.npz']
SOURCES = ['benchmarks/new_bank_v3/calibration_rest_continuous_unibo_v1.py',
    'src/emgimu/feature_bank/calibration_rest_detector_v1.py', 'tests/test_calibration_rest_detector_v1.py',
    'src/emgimu/feature_bank/autonomous_bouts_v1.py', 'src/emgimu/feature_bank/detected_g5_reader_v1.py',
    'src/emgimu/feature_bank/personal_temporal_bouts_v1.py', 'src/emgimu/feature_bank/document_path_v3.py',
    'src/emgimu/feature_bank/temporal.py', 'src/emgimu/feature_bank/validated_unibo.py',
    'src/emgimu/feature_bank/force_nested_oof.py', 'src/emgimu/datasets/unibo_physiology.py',
    'src/emgimu/feature_bank/core.py', 'src/emgimu/feature_bank/unibo_study.py',
    'src/emgimu/feature_bank/screening.py', 'src/emgimu/datasets/benchmark.py', 'src/emgimu/state.py',
    'src/emgimu/datasets/unibo_baseline.py', 'src/emgimu/signal.py',
    'src/emgimu/feature_bank/continuous_calibration_isolation_v1.py',
    'benchmarks/new_bank_v3/autonomous_continuous_unibo_v1.py',
    'benchmarks/new_bank_v3/personal_temporal_unibo_v1.py',
    'benchmarks/new_bank_v3/detected_personal_temporal_unibo_v3.py',
    'src/emgimu/feature_bank/temporal_bout_live_v1.py']


def read(path): return json.loads(path.read_text(encoding='utf8'))


def prepare():
    source = read(HERE/PARENTS[0]); prior = read(HERE/PARENTS[1]); bp = read(HERE/PARENTS[2])
    external = {bp['archive']:bp['archive_sha256']}
    for side in ('frozen_source', 'frozen_final'):
        for name in ('bout_metadata.json', 'split_trial_ids.json'):
            path = Path(source[side])/name
            if path.exists(): external[str(path)] = sha(path)
    path = Path(source['frozen_source'])/'fitted_states.pkl'; external[str(path)] = sha(path)
    for name, digest in prior['artifact_sha256'].items():
        if '_Day6_' in name or '_personal.zip' in name or '_config.json' in name:
            assert sha(ROOT/name) == digest; external[str(ROOT/name)] = digest
    p = dict(schema='calibration_rest_continuous_unibo_v1_protocol',
        parent_sha256={name:sha(HERE/name) for name in PARENTS},
        source_sha256={name:sha(ROOT/name) for name in SOURCES}, external_sha256=external,
        current_shots=[0, 1, 2, 5], sample_rate_hz=200., chunk_samples=37, match_iou=.5,
        selection='Previously inspected Day6, all7 users/all4 postures, same fixed1152 reference intervals including596 supported references from previous calibration-isolated study. All140 reserved original numbered records and100-source-sample resampling guards remain excluded. This axis is frozen before the new detector fit and is identical for old/source/current candidates. No extra pair-dependent reference removal, recutting or outcome-based matching.',
        detector='Same immutable default AutonomousBoutDetectorV1: trailing25ms RMS and source-prespecified quantile/MAD thresholds, onset.08/release.12/preroll.10/min1/max30s. Zero-shot fits only5 neutral bouts from the20 long-term profile records. Positive shots fit only the nested1/2/5 neutral bouts in the4/8/20 current profile, replacing the long detector just like desktop V5. Concatenate neutral sequences exactly as desktop; no active calibration sample can set a threshold. Freeze thresholds during all queries; no search or user-specific override.',
        classifier='Reuse exact personalized Days1-5 G5 family/scaler/classifier and source Day5 temperature. New detected intervals require new probability reads, not fitting. Use the exact previously saved long/current DTW/signature profiles and eleven fixed .75/.125/.125 arms; no source/profile or fusion update.',
        baseline='Reuse previous Day1 source-Rest detector events/probabilities/metrics at every budget; never replay or refit the old experiment. Compare old full against new full at the SAME calibration budget and supported-reference denominator. Conditional matched subsets differ, so their loss/F1 are descriptive rather than paired same-trial increments.',
        metrics='Report all-native-gesture IoU precision/recall, supported-reference end-to-end correct/596 including every miss, per-user/per-class counts and coverage, onset/offset error, censored candidates and every unsupported/unmatched prediction. Four-class probability loss/Brier and active F1 are conditional on supported matches; equal observed active class within user and equal nonempty user mass. An empty user has no invented conditional metric; all its references remain misses.',
        primary='Fixed five-shot new full vs previous five-shot full: higher unweighted supported-reference success, nonworse all-reference detection recall and precision, at least5/7 user end-to-end success wins, all7 users matching all3 active classes, and nonworse overall fist end-to-end success. All six required, no default promotion even if this retrospective diagnostic passes.',
        isolation='Calibrator requires the exact temporal profile trial/recording axes and separate calibration labels. Whole continuous files are replayed label-free, then any event touching the fixed guarded calibration mask is excluded. The reference axis is already frozen and unchanged. Calibration and evaluation are disjoint regions of the same Day6 files; calibration portions are not necessarily earlier, so no chronological onboarding claim.',
        cost='Detector-only use is5 long neutral bouts at0 current, or1/2/5 current neutral bouts; full temporal lifecycle still requires20 long and0/4/8/20 current trials. G5 common Days1-5 history is additional. Distinguish reserved records, detector-used records and template-used records.',
        scope='Retrospective public4-channel200Hz registered-Rest detector diagnostic. GUI neutral-selection and detector equations match, but its8-channel250Hz highpass-filtered sensor stream differs; no own-device, physical boundary, latency, fatigue, prospective generalization or clinical claim.',
        versions=dict(numpy=np.__version__, scipy=__import__('scipy').__version__), default_promoted=False)
    with PROTOCOL.open('x', encoding='utf8', newline='\n') as f: json.dump(p, f, indent=2); f.write('\n')


def temporal_expected(batch, workflow, personal, session, base):
    paths, signature = path_and_signature(batch)
    long_dtw, long_sig = oracle_read(paths, signature, personal, workflow.band)
    local_dtw, local_sig = (long_dtw, long_sig) if session is None else oracle_read(paths, signature, session, workflow.band)
    expected = dict(DTW_long=long_dtw, signature_long=long_sig, DTW_local=local_dtw,
        DTW_blended=(long_dtw+local_dtw)/2, signature_blended=(long_sig+local_sig)/2, base=base)
    for arm in ('DTW_long', 'DTW_blended', 'signature_blended'):
        expected['base_'+arm] = .75*base+.25*expected[arm]
    expected['base_full'] = .75*base+.125*(expected['DTW_blended']+expected['signature_blended'])
    expected['base_uniform'] = .75*base+.25/4
    return expected


def run(resume=False):
    if RESULT.exists(): raise FileExistsError('Completed native experiment must not be rerun')
    p = read(PROTOCOL)
    assert p['versions'] == dict(numpy=np.__version__, scipy=__import__('scipy').__version__)
    for names, root in [(p['parent_sha256'], HERE), (p['source_sha256'], ROOT)]:
        for name, digest in names.items(): assert sha(root/name) == digest, name
    for name, digest in p['external_sha256'].items(): assert sha(Path(name)) == digest, name
    if OUT.exists():
        if not resume: raise FileExistsError('Use explicit --resume to skip verified completed-user checkpoints')
        assert read(OUT/'run_manifest.json')['protocol_sha256'] == sha(PROTOCOL)
    else:
        if resume: raise FileNotFoundError('No existing run to resume')
        (OUT/'parts').mkdir(parents=True)
        (OUT/'run_manifest.json').write_text(json.dumps(dict(protocol_sha256=sha(PROTOCOL)))+'\n', encoding='utf8', newline='\n')
    source = read(HERE/PARENTS[0]); prior = read(HERE/PARENTS[1]); bp = read(HERE/PARENTS[2])
    baseline = read(HERE/PARENTS[3]); intervals = read(HERE/PARENTS[4])
    blocks_by_user = {(b['user'], b['shots']):b for b in prior['blocks'] if b['day'] == 6}
    original_blocks = {(b['user'], b['shots']):b for b in baseline['blocks'] if b['mode'] == 'detected'}
    metadata = {r['id']:r for side in ('frozen_source', 'frozen_final')
        for r in read(Path(source[side])/'bout_metadata.json')}
    split = {r['user']:r for r in read(Path(source['frozen_source'])/'split_trial_ids.json')}
    files = {f.stem:f for f in (Path(source['dataset_root'])/'trials').rglob('*.npz')}
    states, temperatures = pickle.loads((Path(source['frozen_source'])/'fitted_states.pkl').read_bytes())
    source_before = pickle.dumps((states, temperatures))
    completed = []
    with zipfile.ZipFile(bp['archive']) as archive:
        for user in sorted(split):
            part = OUT/'parts'/user; ready = Path(str(part)+'.ready.json')
            if ready.exists():
                receipt = read(ready); assert receipt['protocol_sha256'] == sha(PROTOCOL)
                for name, digest in receipt['artifacts'].items(): assert sha(OUT/name) == digest
                completed.append(read(Path(str(part)+'.json')))
                print(user+': verified completed checkpoint; skipped inference', flush=True); continue
            assert not Path(str(part)+'.json').exists() and not Path(str(part)+'.npz').exists()
            wanted = split[user]['source_template_candidates']+blocks_by_user[user, 5]['calibration_ids']
            sequences = {}
            for trial in sorted({metadata[i]['trial'] for i in wanted}):
                path = files[trial]; native = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.)
                for i in wanted:
                    row = metadata[i]
                    if row['trial'] == trial:
                        assert sha(path) == row['raw_sha256']
                        assert np.all(native.hand_label[row['start']:row['end']] == row['label'])
                        sequences[i] = native.emg[row['start']:row['end']].copy()
            config = read(HERE/'personal_temporal_unibo_v1'/(user+'_config.json'))
            workflow = PersonalTemporalBoutsV1(**config)
            personal = workflow.load_profile(ROOT/blocks_by_user[user, 0]['personal_profile'], user_id=user)
            def batch_for(ids):
                return TemporalBoutBatchV1(tuple(sequences[i] for i in ids), tuple(ids),
                    tuple(metadata[i]['trial'] for i in ids), tuple(metadata[i]['start'] for i in ids),
                    200., tuple(config['channel_ids']), config['preprocessing_id'], 'complete_cued')
            def labels_for(ids): return {i:CLASSES[metadata[i]['label']] for i in ids}
            long_detector, long_receipt = fit_calibration_rest_detector(workflow, batch_for(personal.trial_ids),
                labels_for(personal.trial_ids), personal, user_id=user, rest_label='neutral')
            source_ids = sorted({r['trial'] for r in metadata.values() if r['user'] == user and r['day'] <= 5})
            family, scaler, model = states[user][1]['g5']
            reader = DetectedG5ReaderV1(family, scaler, model, source_recording_ids=source_ids, temperature=temperatures[user]['G5'])
            selected_recordings = [r for r in intervals['recordings'] if r['user'] == user]
            raw_recordings = {}
            for r in selected_recordings:
                x, _, _, digest = load(archive, r['member']); assert digest == r['member_sha256']
                mask = projected_exclusion_mask(len(x), [(a['source_start'], a['source_end']) for a in r['reserved_regions']],
                    source_rate=500., target_rate=200., guard_source_samples=100)
                raw_recordings[r['member']] = (x, mask)
            refs = [r for r in intervals['references'] if r['user'] == user]
            supported_refs = [r['label'] for r in refs if r['label'] is not None]
            output = dict(user=user, blocks=[], recordings=[]); arrays = {}; maximum_error = 0.
            frozen = pickle.dumps((workflow, personal, long_detector))
            for shots in p['current_shots']:
                b = blocks_by_user[user, shots]
                session = None if not shots else workflow.load_profile(ROOT/b['session_profile'], user_id=user)
                if shots:
                    detector, registration = fit_calibration_rest_detector(workflow, batch_for(session.trial_ids),
                        labels_for(session.trial_ids), session, user_id=user, rest_label='neutral')
                else: detector, registration = long_detector, long_receipt
                immutable = pickle.dumps((detector, session)); events = []; raw_events = []; starts = []; ids = []; recordings = []
                probabilities = []; detected_totals = 0; match_totals = 0; onset = []; offset = []; censored = 0
                for saved in selected_recordings:
                    name = saved['member']; x, mask = raw_recordings[name]; current = copy.deepcopy(detector); detected = []
                    for start in range(0, len(x), p['chunk_samples']):
                        detected += current.feed(x[start:start+p['chunk_samples']], start, trial_id=name)
                    censored_end = current.finish(); censored += int(censored_end); retained = []; excluded_bounds = []
                    for e in detected:
                        if mask[e.start:e.end].any(): excluded_bounds.append([e.start, e.end])
                        else: retained.append(e)
                    member_refs = [r for r in refs if r['member'] == name]
                    pairs = match(retained, [(r['native_label'], r['start'], r['end']) for r in member_refs])
                    by_event = {m['event_index']:m['reference_index'] for m in pairs}
                    detected_totals += len(retained); match_totals += len(pairs)
                    onset.extend(m['onset_error_s'] for m in pairs); offset.extend(m['offset_error_s'] for m in pairs)
                    output['recordings'].append(dict(member=name, shots=shots, detector=registration,
                        excluded_event_bounds=excluded_bounds, detected_bounds=[[e.start, e.end] for e in retained],
                        reference_ids=[r['id'] for r in member_refs], matches=pairs, censored_end=censored_end))
                    for i, e in enumerate(retained):
                        eid = f'{user}:s{shots}:{name}:event:{i}'; j = by_event.get(i)
                        ref = None if j is None else member_refs[j]
                        row = dict(id=eid, member=name, start=e.start, end=e.end,
                            reference_id=None if ref is None else ref['id'],
                            reference_label=None if ref is None else ref['label'],
                            status='unmatched' if ref is None else 'unsupported_matched' if ref['label'] is None else 'supported_matched')
                        q = reader.read(e, recording_id=name)
                        row.update(complete_windows=q['complete_windows'], unrepresented_tail_samples=q['unrepresented_tail_samples'])
                        events.append(row); raw_events.append(e.emg); starts.append(e.start); ids.append(eid); recordings.append(name)
                        probabilities.append(q['probability'])
                if events:
                    query = TemporalBoutBatchV1(tuple(raw_events), tuple(ids), tuple(recordings), tuple(starts),
                        200., tuple(config['channel_ids']), config['preprocessing_id'], 'estimated')
                    base_q = np.stack(probabilities)
                    result = workflow.predict(query, personal=personal, session=session, user_id=user, session_id='Day6',
                        base_probabilities=base_q, base_trial_ids=query.trial_ids, base_class_names=CLASSES)
                    expected = temporal_expected(query, workflow, personal, session, base_q)
                    arm_values = result['arms']
                    for arm, q in arm_values.items():
                        error = float(np.max(abs(q-expected[arm]))); maximum_error = max(maximum_error, error)
                        assert error <= 1e-10
                else: arm_values = {arm:np.empty((0, 4)) for arm in ARMS}
                positions = [i for i, e in enumerate(events) if e['reference_label'] is not None]
                y = np.array([events[i]['reference_label'] for i in positions], dtype=int)
                weights = conditional_weights(y) if len(y) else np.empty(0)
                scores = {}; outcomes = {}; key = f'{user}_s{shots}'
                for arm, q in arm_values.items():
                    arrays[key+'_'+arm] = q; matched_q = q[positions]
                    scores[arm] = score(y, matched_q, weights) if len(y) else None
                    outcomes[arm] = end_to_end(y, matched_q, supported_refs)
                baseline_outcome = original_blocks[user, shots]['end_to_end']['base_full']
                original_recordings = [r for r in intervals['recordings'] if r['user'] == user]
                old_detections = sum(len(r['retained_events']) for r in original_recordings)
                old_matches = sum(len(r['retained_pairs']) for r in original_recordings)
                output['blocks'].append(dict(key=key, user=user, shots=shots, detector_registration=registration,
                    personal_profile=b['personal_profile'], session_profile=b['session_profile'],
                    template_calibration_ids=b['calibration_ids'], reserved_recordings=b['reserved_recordings'],
                    reference_ids=[r['id'] for r in refs], supported_reference_labels=supported_refs,
                    events=events, matched_positions=positions, labels=y.tolist(), weights=weights.tolist(),
                    detection=dict(detections=detected_totals, matches=match_totals, references=len(refs),
                        precision=match_totals/detected_totals if detected_totals else 0., recall=match_totals/len(refs),
                        onset_mae_s=float(np.mean(np.abs(onset))) if onset else None,
                        offset_mae_s=float(np.mean(np.abs(offset))) if offset else None, censored_candidates=censored),
                    scores=scores, end_to_end=outcomes, baseline_end_to_end=baseline_outcome,
                    baseline_detection=dict(detections=old_detections, matches=old_matches, references=len(refs)),
                    temporal_calibration_cost={arm:dict(long_term=costs(arm, shots)[0], current=costs(arm, shots)[1]) for arm in ARMS},
                    detector_calibration_cost=dict(long_term_neutral=0 if shots else 5, current_neutral=shots)))
                assert pickle.dumps((detector, session)) == immutable and pickle.dumps((workflow, personal, long_detector)) == frozen
                print(f'{user} {shots}-shot: {len(events)} detections, {len(y)}/{len(supported_refs)} supported matches', flush=True)
            assert pickle.dumps((states, temperatures)) == source_before
            output['independent_DP_path_fusion_max_error'] = maximum_error
            with Path(str(part)+'.npz').open('xb') as f: np.savez_compressed(f, **arrays)
            with Path(str(part)+'.json').open('x', encoding='utf8', newline='\n') as f: json.dump(output, f, indent=2); f.write('\n')
            with ready.open('x', encoding='utf8', newline='\n') as f:
                json.dump(dict(protocol_sha256=sha(PROTOCOL), artifacts={
                    ('parts/'+user+suffix):sha(Path(str(part)+suffix)) for suffix in ('.json', '.npz')}), f, indent=2); f.write('\n')
            completed.append(output)
    # Aggregate saved immutable checkpoints without replay or fitting.
    blocks = [b for part in completed for b in part['blocks']]; pooled = []
    for shots in p['current_shots']:
        subset = [b for b in blocks if b['shots'] == shots]
        for arm in ARMS:
            ys = []; qs = []; ws = []; correct = 0; total = 0; per_class = {}
            for b in subset:
                values = np.load(OUT/'parts'/(b['user']+'.npz'), allow_pickle=False)
                q = values[b['key']+'_'+arm][b['matched_positions']]; values.close()
                if b['labels']: ys.append(np.array(b['labels'])); qs.append(q); ws.append(np.array(b['weights']))
                outcome = b['end_to_end'][arm]; correct += outcome['correct']; total += outcome['supported_references']
                for c, counts in outcome['per_class'].items():
                    acc = per_class.setdefault(c, dict(references=0, matched=0, misses=0, correct=0))
                    for key in acc: acc[key] += counts[key]
            for counts in per_class.values(): counts['success'] = counts['correct']/counts['references']
            metrics = score(np.concatenate(ys), np.concatenate(qs), np.concatenate([w/len(ys) for w in ws])) if ys else None
            pooled.append(dict(shots=shots, arm=arm, conditional_scores=metrics, conditional_users=len(ys),
                end_to_end=dict(correct=correct, supported_references=total, success=correct/total, per_class=per_class)))
    selected = [b for b in blocks if b['shots'] == 5]
    full = next(r for r in pooled if r['shots'] == 5 and r['arm'] == 'base_full')['end_to_end']
    old_full = next(r for r in baseline['pooled'] if r['mode'] == 'detected' and r['shots'] == 5 and r['arm'] == 'base_full')['end_to_end']
    n_det = sum(b['detection']['detections'] for b in selected); n_match = sum(b['detection']['matches'] for b in selected)
    n_ref = sum(b['detection']['references'] for b in selected); old_det = baseline['totals']['retained_detections']
    old_match = baseline['totals']['retained_matches']; wins = sum(b['end_to_end']['base_full']['success'] > b['baseline_end_to_end']['success'] for b in selected)
    guards = dict(higher_supported_reference_success=full['success'] > old_full['success'],
        nonworse_detection_recall=n_match/n_ref >= old_match/n_ref,
        nonworse_detection_precision=(n_match/n_det if n_det else 0.) >= old_match/old_det,
        at_least5of7_user_success_wins=wins >= 5,
        all7_users_all3_matched_classes=all(set(b['labels']) == {1, 2, 3} for b in selected),
        nonworse_fist_success=full['per_class']['fist']['success'] >= old_full['per_class']['fist']['success'])
    for name, digest in p['external_sha256'].items():
        if name != bp['archive']: assert sha(Path(name)) == digest
    artifacts = {f.relative_to(ROOT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file()}
    result = dict(schema='calibration_rest_continuous_unibo_v1', protocol_sha256=sha(PROTOCOL), artifact_sha256=artifacts,
        blocks=blocks, pooled=pooled, primary_guards=guards, primary_pass=all(guards.values()), primary_user_success_wins=wins,
        primary_detection=dict(detections=n_det, matches=n_match, references=n_ref,
            precision=n_match/n_det if n_det else 0., recall=n_match/n_ref),
        maximum_independent_DP_path_fusion_error=max(part['independent_DP_path_fusion_max_error'] for part in completed),
        source_classifier_and_profiles_immutable=True, source_G5_refitted=False, completed_prior_experiments_rerun=False,
        default_promoted=False, physical_validation_proven=False, completion_proven=False, scope=p['scope'])
    with RESULT.open('x', encoding='utf8', newline='\n') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(primary_guards=guards, user_success_wins=wins, success=full['success'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run(args.resume)
