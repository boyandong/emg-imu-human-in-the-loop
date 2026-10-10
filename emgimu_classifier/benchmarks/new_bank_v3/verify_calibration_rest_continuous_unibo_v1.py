"""Independent vector-energy/state-machine and algebraic G5 verification.

No detector feed, model fit, classifier predict or production scoring calls.
"""
import hashlib
import json
import math
from pathlib import Path
import pickle
import zipfile
import numpy as np
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import sha, load
from benchmarks.new_bank_v3.verify_detected_personal_temporal_unibo_v3 import metrics, success, check_metrics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'feature_bank/CALIBRATION_REST_CONTINUOUS_UNIBO_ACCEPTANCE_V1.json'
ARMS = ('DTW_long', 'signature_long', 'DTW_local', 'DTW_blended', 'signature_blended',
        'base', 'base_DTW_long', 'base_DTW_blended', 'base_signature_blended', 'base_full', 'base_uniform')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def thresholds(rest, rate):
    width = max(1, round(rate*.025))
    energy = np.mean(rest.astype(float)**2, axis=1)
    activity = np.sqrt(np.convolve(energy, np.ones(width)/width, mode='valid'))
    median = np.median(activity); deviation = np.median(abs(activity-median))
    scale = max(1.4826*deviation, median*.05, 1e-10)
    off = max(np.quantile(activity, .95), median+3*scale)
    on = max(np.quantile(activity, .995), median+6*scale, off*1.2)
    return float(on), float(off), width


def independent_intervals(samples, on, off, rate=200.):
    """Vector trailing energy followed by an independently written FSM."""
    width = max(1, round(rate*.025)); n = len(samples)
    energy = np.mean(np.asarray(samples, float)**2, axis=1)
    activity = np.sqrt(np.convolve(energy, np.ones(width), mode='full')[:n]/np.minimum(np.arange(n)+1, width))
    onset = math.ceil(.08*rate); release = math.ceil(.12*rate); preroll = math.ceil(.10*rate)
    minimum = math.ceil(rate); maximum = math.ceil(30*rate)
    armed = False; start = None; high = 0; low = 0; first = None; result = []
    for i, value in enumerate(activity):
        ready = i >= width-1
        low = low+1 if ready and value <= off else 0
        if start is None:
            if low >= release: armed = True
            if armed and ready and value >= on:
                if high == 0: first = max(0, i-preroll)
                high += 1
                if high >= onset: start = first; low = 0
            else: high = 0
        else:
            if i-start+1 > maximum:
                start = None; armed = False; high = low = 0
            elif low >= release:
                if i-start+1 >= minimum: result.append([int(start), i+1])
                start = None; armed = True; high = 0
    return result, start is not None or high > 0


def independent_matches(events, refs):
    candidates = []
    for i, (a, b) in enumerate(events):
        for j, r in enumerate(refs):
            c, d = r['start'], r['end']; intersection = max(0, min(b, d)-max(a, c))
            iou = intersection/(max(b, d)-min(a, c))
            if iou >= .5: candidates.append((-iou, i, j))
    used_events = set(); used_refs = set(); result = []
    for negative_iou, i, j in sorted(candidates):
        if i in used_events or j in used_refs: continue
        used_events.add(i); used_refs.add(j); a, b = events[i]; r = refs[j]
        result.append(dict(iou=-negative_iou, event_index=i, reference_index=j, label=r['native_label'],
            onset_error_s=(a-r['start'])/200., offset_error_s=(b-r['end'])/200.))
    return result


def independent_g5(samples, scaler, model, temperature):
    # Match the documented float32 window coordinates/mean, with independent
    # direct waveform statistics, scaler arithmetic and multinomial logits.
    x = np.asarray(samples[:len(samples)//40*40], np.float32).reshape(-1, 40, 4).astype(float)
    early = np.sqrt(np.sum(x[:, :20]**2, axis=1)/20)
    late = np.sqrt(np.sum(x[:, 20:]**2, axis=1)/20)
    time = np.arange(40, dtype=float)/39-.5
    slope = np.sum(x*time[None, :, None], axis=1)/np.sum(time*time)
    features = np.column_stack((early, late, early-late, np.log((early+1e-12)/(late+1e-12)), slope)).astype(np.float32)
    z = features.mean(0, keepdims=True); z -= scaler.mean_; z /= scaler.scale_
    logits = z@model.coef_.T+model.intercept_; logits -= logits.max(1, keepdims=True)
    raw = np.exp(logits); raw /= raw.sum(1, keepdims=True)
    calibrated = np.log(np.maximum(raw, 1e-15))/temperature; calibrated -= calibrated.max(1, keepdims=True)
    q = np.exp(calibrated); q /= q.sum(1, keepdims=True)
    return q[0]


def verify():
    protocol_path = HERE/'CALIBRATION_REST_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
    result_path = HERE/'CALIBRATION_REST_CONTINUOUS_UNIBO_V1_RESULTS.json'
    p, result = read(protocol_path), read(result_path)
    assert sha(protocol_path) == result['protocol_sha256']
    for key, root in [('parent_sha256', HERE), ('source_sha256', ROOT)]:
        for name, digest in p[key].items(): assert sha(root/name) == digest, name
    for name, digest in p['external_sha256'].items(): assert sha(Path(name)) == digest, name
    for name, digest in result['artifact_sha256'].items(): assert sha(ROOT/name) == digest, name
    source = read(HERE/'F5_PATH_UNIBO_PROTOCOL.json'); bp = read(HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json')
    prior = read(HERE/'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json')
    previous = read(HERE/'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json')
    intervals = read(HERE/'detected_personal_temporal_unibo_v3/intervals.json')
    prior_blocks = {(b['user'], b['shots']):b for b in prior['blocks'] if b['day'] == 6}
    old_blocks = {(b['user'], b['shots']):b for b in previous['blocks'] if b['mode'] == 'detected'}
    metadata = {r['id']:r for side in ('frozen_source', 'frozen_final')
        for r in read(Path(source[side])/'bout_metadata.json')}
    files = {f.stem:f for f in (Path(source['dataset_root'])/'trials').rglob('*.npz')}
    states, temperatures = pickle.loads((Path(source['frozen_source'])/'fitted_states.pkl').read_bytes())
    calibration = {}; used_calibration = 0; max_g5_error = 0.; threshold_error = 0.; checked_events = 0
    aggregate = {}; records_checked = 0
    with zipfile.ZipFile(bp['archive']) as archive:
        for user in sorted(states):
            part = read(HERE/'calibration_rest_continuous_unibo_v1/parts'/(user+'.json'))
            ready = read(HERE/'calibration_rest_continuous_unibo_v1/parts'/(user+'.ready.json'))
            assert ready['protocol_sha256'] == sha(protocol_path)
            for name, digest in ready['artifacts'].items(): assert sha(HERE/'calibration_rest_continuous_unibo_v1'/name) == digest
            assert part['blocks'] == [b for b in result['blocks'] if b['user'] == user]
            refs = [r for r in intervals['references'] if r['user'] == user]
            raw = {}
            for recording in [r for r in intervals['recordings'] if r['user'] == user]:
                x, _, _, digest = load(archive, recording['member']); assert digest == recording['member_sha256']
                # Independent integer source-clock projection, not production mask.
                mask = np.zeros(len(x), bool)
                for r in recording['reserved_regions']:
                    a = max(0, math.ceil((r['source_start']-100)*2/5))
                    b = min(len(x), math.ceil((r['source_end']+100)*2/5)); mask[a:b] = True
                raw[recording['member']] = (x, mask)
            with np.load(HERE/'calibration_rest_continuous_unibo_v1/parts'/(user+'.npz'), allow_pickle=False) as arrays:
                assert len(arrays.files) == 44
                for block in part['blocks']:
                    shots = block['shots']; old = prior_blocks[user, shots]
                    assert block['reference_ids'] == [r['id'] for r in refs]
                    assert block['supported_reference_labels'] == [r['label'] for r in refs if r['label'] is not None]
                    for key in ('personal_profile', 'session_profile', 'reserved_recordings'):
                        assert block[key] == old[key]
                    assert block['template_calibration_ids'] == old['calibration_ids']
                    path = ROOT/(old['session_profile'] if shots else old['personal_profile'])
                    with zipfile.ZipFile(path) as z:
                        manifest = json.loads(z.read('manifest.json')); payload = z.read('profile.pkl')
                        assert hashlib.sha256(payload).hexdigest() == manifest['payload_sha256']
                        profile = pickle.loads(payload)
                    neutral_ids = [i for i in profile.trial_ids if metadata[i]['label'] == 0]
                    rest = []
                    for i in neutral_ids:
                        if i not in calibration:
                            r = metadata[i]; path = files[r['trial']]; assert sha(path) == r['raw_sha256']
                            with np.load(path, allow_pickle=False) as z:
                                assert np.all(z['hand_label'][r['start']:r['end']] == 0)
                                calibration[i] = z['emg'][r['start']:r['end']].copy()
                        rest.append(calibration[i])
                    rest = np.concatenate(rest); on, off, width = thresholds(rest, 200.)
                    registration = block['detector_registration']
                    assert registration['temporal_profile_id'] == profile.profile_id
                    assert registration['neutral_trial_ids'] == neutral_ids and len(neutral_ids) == (shots if shots else 5)
                    assert registration['neutral_samples'] == len(rest) and registration['smoothing_samples'] == width
                    assert registration['neutral_recording_ids'] == [metadata[i]['trial'] for i in neutral_ids]
                    assert registration['neutral_only'] and not registration['current_queries_used']
                    threshold_error = max(threshold_error, abs(on-registration['on']), abs(off-registration['off']))
                    np.testing.assert_allclose([registration['on'], registration['off']], [on, off], rtol=0, atol=1e-10)
                    records = [r for r in part['recordings'] if r['shots'] == shots]; assert len(records) == 4
                    by_reference = {}; expected_events = []; det_count = matches_count = 0; censored_count = 0; onset = []; offset = []
                    for record in records:
                        records_checked += 1; name = record['member']; x, mask = raw[name]
                        whole, censored = independent_intervals(x, on, off)
                        retained = [bounds for bounds in whole if not mask[bounds[0]:bounds[1]].any()]
                        excluded = [bounds for bounds in whole if mask[bounds[0]:bounds[1]].any()]
                        assert retained == record['detected_bounds'] and excluded == record['excluded_event_bounds']
                        assert censored == record['censored_end']; censored_count += int(censored)
                        member_refs = [r for r in refs if r['member'] == name]
                        assert record['reference_ids'] == [r['id'] for r in member_refs]
                        pairs = independent_matches(retained, member_refs); assert pairs == record['matches']
                        by_event = {m['event_index']:m['reference_index'] for m in pairs}
                        det_count += len(retained); matches_count += len(pairs)
                        onset.extend(m['onset_error_s'] for m in pairs); offset.extend(m['offset_error_s'] for m in pairs)
                        for i, bounds in enumerate(retained):
                            ref = None if i not in by_event else member_refs[by_event[i]]
                            expected_events.append((name, bounds, ref))
                    assert len(expected_events) == len(block['events'])
                    source_family, scaler, model = states[user][1]['g5']
                    assert source_family.group == 'G5' and source_family.transformer_.groups == ('G5',)
                    np.testing.assert_array_equal(model.classes_, [0, 1, 2, 3])
                    base = arrays[block['key']+'_base']; checked_events += len(expected_events)
                    for index, ((name, (a, b), ref), event) in enumerate(zip(expected_events, block['events'])):
                        assert (event['member'], event['start'], event['end']) == (name, a, b)
                        assert event['reference_id'] == (None if ref is None else ref['id'])
                        assert event['reference_label'] == (None if ref is None else ref['label'])
                        assert event['complete_windows'] == (b-a)//40 and event['unrepresented_tail_samples'] == (b-a)%40
                        q = independent_g5(raw[name][0][a:b], scaler, model, temperatures[user]['G5'])
                        difference = float(np.max(abs(q-base[index]))); max_g5_error = max(max_g5_error, difference)
                        assert difference < 1e-6, (user, shots, index, difference)
                    positions = [i for i, e in enumerate(block['events']) if e['reference_label'] is not None]
                    y = np.array([block['events'][i]['reference_label'] for i in positions], int)
                    assert positions == block['matched_positions'] and y.tolist() == block['labels']
                    w = np.array([1/(len(set(y))*list(y).count(label)) for label in y]) if len(y) else np.empty(0)
                    np.testing.assert_array_equal(w, block['weights'])
                    values = {arm:arrays[block['key']+'_'+arm] for arm in ARMS}
                    expected = dict(base_uniform=.75*base+.25/4,
                        base_full=.75*base+.125*values['DTW_blended']+.125*values['signature_blended'])
                    for arm in ('DTW_long', 'DTW_blended', 'signature_blended'):
                        expected['base_'+arm] = .75*base+.25*values[arm]
                    for arm, q in expected.items(): np.testing.assert_allclose(q, values[arm], rtol=0, atol=1e-14)
                    for arm, q in values.items():
                        assert q.shape == (len(block['events']), 4) and np.isfinite(q).all() and np.all(q >= 0)
                        np.testing.assert_allclose(q.sum(1), 1., rtol=0, atol=1e-12)
                        if len(y): check_metrics(block['scores'][arm], y, q[positions], w)
                        else: assert block['scores'][arm] is None
                        assert block['end_to_end'][arm] == success(y, q[positions], block['supported_reference_labels'])
                        aggregate.setdefault((shots, arm), []).append((y, q[positions], w, block['supported_reference_labels']))
                        if arm in ('base', 'base_uniform'): cost = (0, 0)
                        elif arm in ('DTW_long', 'signature_long', 'base_DTW_long'): cost = (20, 0)
                        elif arm == 'DTW_local' and shots: cost = (0, 4*shots)
                        else: cost = (20, 4*shots)
                        assert block['temporal_calibration_cost'][arm] == dict(long_term=cost[0], current=cost[1])
                    assert block['detector_calibration_cost'] == dict(long_term_neutral=0 if shots else 5, current_neutral=shots)
                    assert block['baseline_end_to_end'] == old_blocks[user, shots]['end_to_end']['base_full']
                    d = block['detection']
                    assert (d['detections'], d['matches'], d['references'], d['censored_candidates']) == (det_count, matches_count, len(refs), censored_count)
                    assert d['precision'] == (matches_count/det_count if det_count else 0.) and d['recall'] == matches_count/len(refs)
                    assert d['onset_mae_s'] == (float(np.mean(np.abs(onset))) if onset else None)
                    assert d['offset_mae_s'] == (float(np.mean(np.abs(offset))) if offset else None)
    assert len(result['blocks']) == 28 and records_checked == 112 and len(result['pooled']) == 44
    for row in result['pooled']:
        entries = aggregate[row['shots'], row['arm']]; present = [e for e in entries if len(e[0])]
        assert row['conditional_users'] == len(present)
        if present:
            check_metrics(row['conditional_scores'], np.concatenate([e[0] for e in present]),
                np.concatenate([e[1] for e in present]), np.concatenate([e[2]/len(present) for e in present]))
        outcomes = [success(e[0], e[1], e[3]) for e in entries]
        correct = sum(r['correct'] for r in outcomes); total = sum(r['supported_references'] for r in outcomes)
        assert row['end_to_end']['correct'] == correct and row['end_to_end']['supported_references'] == total == 596
        assert row['end_to_end']['success'] == correct/total
        for c in ('index_pinch', 'fist', 'open_hand'):
            counts = row['end_to_end']['per_class'][c]
            for field in ('references', 'matched', 'misses', 'correct'):
                assert counts[field] == sum(r['per_class'][c][field] for r in outcomes)
            assert counts['success'] == counts['correct']/counts['references']
    selected = [b for b in result['blocks'] if b['shots'] == 5]
    full = next(r for r in result['pooled'] if r['shots'] == 5 and r['arm'] == 'base_full')['end_to_end']
    baseline = next(r for r in previous['pooled'] if r['mode'] == 'detected' and r['shots'] == 5 and r['arm'] == 'base_full')['end_to_end']
    wins = sum(b['end_to_end']['base_full']['success'] > b['baseline_end_to_end']['success'] for b in selected)
    det = sum(b['detection']['detections'] for b in selected); matched = sum(b['detection']['matches'] for b in selected)
    guards = dict(higher_supported_reference_success=full['success'] > baseline['success'],
        nonworse_detection_recall=matched/1152 >= previous['totals']['retained_matches']/1152,
        nonworse_detection_precision=matched/det >= previous['totals']['retained_matches']/previous['totals']['retained_detections'],
        at_least5of7_user_success_wins=wins >= 5, all7_users_all3_matched_classes=all(set(b['labels']) == {1, 2, 3} for b in selected),
        nonworse_fist_success=full['per_class']['fist']['success'] >= baseline['per_class']['fist']['success'])
    assert guards == result['primary_guards'] and all(guards.values()) == result['primary_pass']
    assert wins == result['primary_user_success_wins'] and not result['default_promoted']
    assert not result['source_G5_refitted'] and not result['completed_prior_experiments_rerun']
    return dict(schema='calibration_rest_continuous_unibo_acceptance_v1', verifier_sha256=sha(Path(__file__)),
        protocol_sha256=sha(protocol_path), result_sha256=sha(result_path), blocks=28, arm_cells=308,
        recordings=112, checked_detected_intervals=checked_events, retained_probability_values=checked_events*4*11,
        independent_threshold_max_error=threshold_error, independent_direct_G5_max_probability_error=max_g5_error,
        independent_vector_energy_FSM_boundaries_exact=True, independent_matching_and_fixed_reference_axes=True,
        metrics_and_misses_independently_reconstructed=True,
        native_inrun_DP_path_fusion_max_error=result['maximum_independent_DP_path_fusion_error'],
        primary_pass=result['primary_pass'], primary_guards=guards, user_success_wins=wins,
        supported_references=596, old_correct=baseline['correct'], registered_rest_correct=full['correct'],
        default_promoted=False, physical_validation_proven=False, completion_proven=False, scope=p['scope'])


if __name__ == '__main__':
    actual = verify()
    OUT.write_text(json.dumps(actual, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps({k:actual[k] for k in ('blocks', 'checked_detected_intervals', 'primary_pass', 'registered_rest_correct')}))
