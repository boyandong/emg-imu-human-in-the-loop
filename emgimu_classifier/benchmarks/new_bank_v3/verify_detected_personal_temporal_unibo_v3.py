"""Read-only independent source-clock, split, probability and metric checks."""
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from scipy.io import loadmat
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import sha

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ACCEPTANCE = ROOT/'feature_bank/DETECTED_PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V3.json'
CLASSES = ('neutral', 'index_pinch', 'fist', 'open_hand')
ARMS = ('DTW_long', 'signature_long', 'DTW_local', 'DTW_blended', 'signature_blended',
        'base', 'base_DTW_long', 'base_DTW_blended', 'base_signature_blended', 'base_full', 'base_uniform')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def metrics(y, q, w):
    w = np.asarray(w)/np.sum(w); y = np.asarray(y); predicted = q.argmax(1)
    f1 = []
    for label in range(4):
        tp = w[(y == label) & (predicted == label)].sum()
        denominator = w[y == label].sum()+w[predicted == label].sum()
        f1.append(2*tp/denominator if denominator else 0.)
    return dict(accuracy=float(w[y == predicted].sum()), macro_f1=float(np.mean(f1)),
        active_macro_f1=float(np.mean(f1[1:])),
        log_loss=float(-np.dot(w, np.log(np.maximum(q[np.arange(len(y)), y], np.finfo(float).eps)))),
        brier=float(np.dot(w, np.mean((q-np.eye(4)[y])**2, axis=1))),
        observed_active_classes=sorted(map(int, set(y))), all_active_classes_observed=set(y) == {1, 2, 3},
        predicted_rest_count=int(np.count_nonzero(predicted == 0)))


def check_metrics(saved, y, q, w):
    actual = metrics(y, q, w)
    for key, value in actual.items():
        if isinstance(value, float):
            np.testing.assert_allclose(saved[key], value, rtol=0, atol=1e-12)
        else:
            assert saved[key] == value, key


def success(y, q, refs):
    counts = {}
    guessed = q.argmax(1); correct = 0
    for label in (1, 2, 3):
        total = sum(v == label for v in refs); matched = int(np.count_nonzero(y == label))
        right = int(np.count_nonzero((y == label) & (guessed == label))); correct += right
        counts[CLASSES[label]] = dict(references=total, matched=matched, misses=total-matched,
            correct=right, success=right/total)
    return dict(supported_references=len(refs), supported_matched=len(y), correct=correct,
        success=correct/len(refs), per_class=counts)


def verify():
    protocol_path = HERE/'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_PROTOCOL.json'
    result_path = HERE/'DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json'
    p, result = read(protocol_path), read(result_path)
    assert sha(protocol_path) == result['protocol_sha256']
    for name, digest in p['parent_sha256'].items(): assert sha(HERE/name) == digest, name
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for name, digest in p['external_sha256'].items(): assert sha(Path(name)) == digest, name
    for name, digest in result['artifact_sha256'].items(): assert sha(ROOT/name) == digest, name
    boundary = read(HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json')
    bp = read(HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json')
    prior = read(HERE/'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json')
    prior_blocks = {(b['user'], b['shots']): b for b in prior['blocks'] if b['day'] == 6}
    old = {(e['member'], e['event_index']): e for e in read(HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json')['events']}
    intervals = read(HERE/'detected_personal_temporal_unibo_v3/intervals.json')
    saved_recordings = {r['member']: r for r in intervals['recordings']}
    expected_event_ids = set(); expected_ref_ids = set(); recovered = set()
    onset = []; offset = []
    with zipfile.ZipFile(bp['archive']) as archive:
        for source in boundary['recordings']:
            name = source['member']; saved = saved_recordings[name]; ident = bp['members'][name]
            user = f"u{ident['user']:02d}"; reserved = set(prior_blocks[user, 0]['reserved_recordings'])
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == source['member_sha256'] == saved['member_sha256']
            mat = loadmat(io.BytesIO(payload), variable_names=('label', 'gestureCounter'))
            counter = mat['gestureCounter'].ravel(); labels = mat['label'].ravel(); regions = []
            # Independent scalar grouping, rather than production edge arrays.
            for value, positions in itertools.groupby(range(len(counter)), key=lambda i:counter[i]):
                positions = list(positions)
                if not value: continue
                a, b = positions[0], positions[-1]+1
                gestures = sorted(set(map(int, labels[a:b]))-{1}); assert len(gestures) == 1
                trial = f"unibo-{user}-d06-p{ident['posture']:02d}-g{gestures[0]:02d}-r{int(value):02d}"
                if trial in reserved:
                    assert trial not in recovered; recovered.add(trial); regions.append((trial, a, b))
            assert sorted((r['trial'], r['source_start'], r['source_end']) for r in saved['reserved_regions']) == sorted(regions)
            n = (len(counter)*2+4)//5; assert n == saved['target_samples'] and len(counter) == saved['source_samples']
            excluded = set()
            for _, a, b in regions:
                first = max(0, math.ceil((a-p['guard_source_samples'])*2/5))
                stop = min(n, math.ceil((b+p['guard_source_samples'])*2/5))
                excluded.update(range(first, stop))
            assert len(excluded) == saved['excluded_target_samples']
            events = source['detected_bounds']; refs = source['reference_bounds']
            bad_events = {i for i, (a, b) in enumerate(events) if any(t in excluded for t in range(a, b))}
            bad_refs = {j for j, (_, a, b) in enumerate(refs) if any(t in excluded for t in range(a, b))}
            assert sorted(bad_events) == saved['direct_excluded_events']
            assert sorted(bad_refs) == saved['direct_excluded_references']
            for match in source['matches']:
                i, j = match['event_index'], match['reference_index']
                if i in bad_events or j in bad_refs:
                    bad_events.add(i); bad_refs.add(j)
            assert sorted(bad_events) == saved['excluded_events'] and sorted(bad_refs) == saved['excluded_references']
            kept_events = sorted(set(range(len(events)))-bad_events)
            kept_refs = sorted(set(range(len(refs)))-bad_refs)
            assert kept_events == saved['retained_events'] and kept_refs == saved['retained_references']
            pairs = [[m['event_index'], m['reference_index']] for m in source['matches']
                     if m['event_index'] not in bad_events]
            assert pairs == saved['retained_pairs']
            matched = dict(pairs)
            for m in source['matches']:
                if m['event_index'] not in bad_events:
                    onset.append(m['onset_error_s']); offset.append(m['offset_error_s'])
            for i in kept_events:
                eid = f'{name}:event:{i}'; expected_event_ids.add(eid)
                row = next(e for e in intervals['events'] if e['id'] == eid)
                assert [row['start'], row['end']] == events[i] and row['reference_index'] == matched.get(i)
                native_label = None if i not in matched else refs[matched[i]][0]
                label = {2:2, 3:1, 6:3}.get(native_label)
                assert row['reference_label'] == label == old[name, i]['reference_label']
                assert row['status'] == ('unmatched' if i not in matched else 'unsupported_matched' if label is None else 'supported_matched')
            for j in kept_refs:
                rid = f'{name}:reference:{j}'; expected_ref_ids.add(rid)
                row = next(r for r in intervals['references'] if r['id'] == rid)
                assert [row['native_label'], row['start'], row['end']] == refs[j]
                assert row['label'] == {2:2, 3:1, 6:3}.get(refs[j][0])
    assert recovered == {r for b in prior_blocks.values() for r in b['reserved_recordings']}
    assert len(recovered) == 140 and len(saved_recordings) == 28
    assert expected_event_ids == {e['id'] for e in intervals['events']}
    assert expected_ref_ids == {r['id'] for r in intervals['references']}
    assert len(expected_event_ids) == len(intervals['events']) and len(expected_ref_ids) == len(intervals['references'])
    totals = dict(original_detections=len(old), retained_detections=len(expected_event_ids),
        retained_references=len(expected_ref_ids), retained_matches=len(onset),
        supported_references=sum(r['label'] is not None for r in intervals['references']),
        supported_matched=sum(e['reference_label'] is not None for e in intervals['events']),
        unsupported_matched=sum(e['status'] == 'unsupported_matched' for e in intervals['events']),
        unmatched_detections=sum(e['status'] == 'unmatched' for e in intervals['events']),
        excluded_calibration_recordings=140, onset_mae_s=float(np.mean(np.abs(onset))),
        offset_mae_s=float(np.mean(np.abs(offset))))
    assert totals == result['totals']
    groups = {}; pooled = {}; predictions = 0; error = 0.; event_by_id = {e['id']:e for e in intervals['events']}
    refs_by_id = {r['id']:r for r in intervals['references']}
    with np.load(HERE/'detected_personal_temporal_unibo_v3/readouts.npz', allow_pickle=False) as arrays:
        assert len(arrays.files) == 7*2*4*11 and len(result['blocks']) == 7*2*4
        for b in result['blocks']:
            user, mode, shots = b['user'], b['mode'], b['shots']
            group = groups.setdefault((user, mode), []); group.append(b)
            assert shots in (0, 1, 2, 5) and b['key'] == f'{user}_{mode}_s{shots}'
            prior_block = prior_blocks[user, shots]
            for field in ('reserved_recordings', 'calibration_ids', 'personal_profile', 'session_profile'):
                assert b[field] == prior_block[field]
            wanted_events = [e for e in intervals['events'] if e['user'] == user and
                (mode == 'detected' or e['reference_label'] is not None)]
            assert b['event_ids'] == [e['id'] for e in wanted_events]
            supported = [e for e in wanted_events if e['reference_label'] is not None]
            assert b['matched_event_ids'] == [e['id'] for e in supported]
            positions = [i for i, e in enumerate(wanted_events) if e['reference_label'] is not None]
            assert b['matched_positions'] == positions
            y = np.array([e['reference_label'] for e in supported]); assert list(y) == b['labels']
            w = np.array([1/(len(set(y))*list(y).count(label)) for label in y])
            np.testing.assert_array_equal(w, b['weights'])
            refs = [r for r in intervals['references'] if r['user'] == user and r['label'] is not None]
            assert b['supported_reference_ids'] == [r['id'] for r in refs]
            q = {arm:arrays[b['key']+'_'+arm] for arm in ARMS}
            predictions += sum(v.size for v in q.values())
            baseline = np.array([old[e['member'], e['event_index']][
                'g5_probability' if mode == 'detected' else 'g5_oracle_probability'] for e in wanted_events])
            np.testing.assert_array_equal(q['base'], baseline)
            if not shots:
                np.testing.assert_array_equal(q['DTW_local'], q['DTW_long'])
                np.testing.assert_array_equal(q['DTW_blended'], q['DTW_long'])
                np.testing.assert_array_equal(q['signature_blended'], q['signature_long'])
            else:
                np.testing.assert_allclose(q['DTW_blended'], (q['DTW_long']+q['DTW_local'])/2, rtol=0, atol=1e-14)
            expected = dict(base_full=.75*baseline+.125*q['DTW_blended']+.125*q['signature_blended'],
                base_uniform=.75*baseline+.25/4)
            for arm in ('DTW_long', 'DTW_blended', 'signature_blended'):
                expected['base_'+arm] = .75*baseline+.25*q[arm]
            for arm, values in expected.items():
                difference = float(np.max(abs(q[arm]-values))); error = max(error, difference); assert difference <= 1e-14
            for arm, values in q.items():
                assert values.shape == (len(wanted_events), 4) and np.isfinite(values).all() and np.all(values >= 0)
                np.testing.assert_allclose(values.sum(1), 1., rtol=0, atol=1e-12)
                matched_q = values[positions]; check_metrics(b['scores'][arm], y, matched_q, w)
                if mode == 'detected': assert b['end_to_end'][arm] == success(y, matched_q, [r['label'] for r in refs])
                else: assert not b['end_to_end']
                if arm in ('base', 'base_uniform'): cost = (0, 0)
                elif arm in ('DTW_long', 'signature_long', 'base_DTW_long'): cost = (20, 0)
                elif arm == 'DTW_local' and shots: cost = (0, 4*shots)
                else: cost = (20, 4*shots)
                assert b['predictive_calibration_cost'][arm] == dict(long_term=cost[0], current=cost[1])
                pooled.setdefault((mode, shots, arm), []).append((y, matched_q, w, [r['label'] for r in refs]))
        for group in groups.values():
            assert sorted(b['shots'] for b in group) == [0, 1, 2, 5]
            for b in group[1:]:
                for field in ('event_ids', 'matched_event_ids', 'labels', 'weights', 'supported_reference_ids'):
                    assert b[field] == group[0][field]
    assert len(result['pooled']) == 88
    for r in result['pooled']:
        entries = pooled[r['mode'], r['shots'], r['arm']]
        y = np.concatenate([v[0] for v in entries]); q = np.concatenate([v[1] for v in entries])
        w = np.concatenate([v[2]/7 for v in entries]); check_metrics(r, y, q, w)
        if r['mode'] == 'detected':
            assert r['end_to_end'] == success(y, q, [label for v in entries for label in v[3]])
            assert abs(r['equal_user_end_to_end_success']-np.mean([success(*v[:2], v[3])['success'] for v in entries])) < 1e-14
    lookup = {(r['mode'], r['shots'], r['arm']):r for r in result['pooled']}
    base, full, uniform = [lookup['detected', 5, arm] for arm in ('base', 'base_full', 'base_uniform')]
    primary = [b for b in result['blocks'] if b['mode'] == 'detected' and b['shots'] == 5]
    wins = sum(b['scores']['base_full']['log_loss'] < b['scores']['base']['log_loss'] for b in primary)
    guards = dict(lower_log_loss=full['log_loss'] < base['log_loss'], lower_brier=full['brier'] < base['brier'],
        nonworse_active_macro_f1=full['active_macro_f1'] >= base['active_macro_f1'],
        nonworse_end_to_end_success=full['end_to_end']['success'] >= base['end_to_end']['success'],
        at_least5_of7_user_loss_wins=wins >= 5, lower_loss_than_uniform=full['log_loss'] < uniform['log_loss'],
        all7_users_have_all3_matched_active_classes=all(set(b['labels']) == {1, 2, 3} for b in primary))
    assert guards == result['primary_guards'] and wins == result['primary_user_loss_wins']
    assert result['primary_pass'] == all(guards.values()) and result['primary_eligible'] == guards['all7_users_have_all3_matched_active_classes']
    assert not result['default_promoted'] and not result['physical_validation_proven']
    assert result['independent_DP_path_signature_fusion_max_error'] < 1e-10
    return dict(schema='detected_personal_temporal_unibo_acceptance_v3', verifier_sha256=sha(Path(__file__)),
        protocol_sha256=sha(protocol_path), result_sha256=sha(result_path), recordings=28,
        excluded_calibration_recordings=140, blocks=56, arm_cells=616, pooled_cells=88,
        retained_predictions=predictions, totals=totals, independent_source_clock_projection_and_pair_exclusion=True,
        every_metric_probability_cost_and_budget_axis_checked=True, frozen_G5_probabilities_exact=True,
        independent_inrun_DP_path_signature_probability_max_error=result['independent_DP_path_signature_fusion_max_error'],
        independent_fusion_max_error=error, primary_guards=guards, primary_eligible=result['primary_eligible'],
        primary_pass=result['primary_pass'], user_loss_wins=wins, default_promoted=False,
        physical_validation_proven=False, completion_proven=False, scope=p['scope'])


if __name__ == '__main__':
    actual = verify()
    ACCEPTANCE.write_text(json.dumps(actual, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps({k:actual[k] for k in ('blocks', 'arm_cells', 'retained_predictions', 'primary_eligible', 'primary_pass')}))
