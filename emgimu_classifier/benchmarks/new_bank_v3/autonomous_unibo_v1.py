"""Frozen source-Rest boundary experiment; no gesture labels enter streaming."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1
from emgimu.feature_bank.unibo_sequence_temporal import contiguous_bouts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'AUTONOMOUS_UNIBO_V1_PROTOCOL.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent = HERE/'F5_PATH_UNIBO_PROTOCOL.json'
    p = json.loads(parent.read_text())
    source = Path(p['frozen_source'])
    splits = json.loads((source/'split_trial_ids.json').read_text())
    metadata = json.loads((source/'bout_metadata.json').read_text())
    ids = {i for s in splits for i in s['source_template_candidates']}
    selected = [r for r in metadata if r['id'] in ids or r['day'] == 6]
    by_user = {}
    for user in sorted({r['user'] for r in selected}):
        by_user[user] = {
            'source_trials': sorted({r['trial'] for r in selected if r['user'] == user and r['id'] in ids}),
            'evaluation_trials': sorted({r['trial'] for r in selected if r['user'] == user and r['day'] == 6}),
        }
        assert not set(by_user[user]['source_trials']) & set(by_user[user]['evaluation_trials'])
    value = {
        'version': 'autonomous_unibo_v1', 'dataset_root': p['dataset_root'],
        'parent_protocol_sha256': sha(parent),
        'source_split_sha256': sha(source/'split_trial_ids.json'),
        'source_metadata_sha256': sha(source/'bout_metadata.json'),
        'by_user': by_user,
        'policy': {'onset_s': .08, 'release_s': .12, 'preroll_s': .10,
                   'min_duration_s': 1., 'max_duration_s': 30.},
        'match_iou': .5, 'chunk_samples': 37,
        'reference': 'Native non-neutral label segments >=1 second, solely for scoring. Labels represent protocol/cue intervals, not measured physiological onset.',
        'scope': 'Retrospective public Day6 validation on already inspected recordings. New frozen detector; no heldout tuning, no classifier, no live/device or biological-completeness claim.',
    }
    if PROTOCOL.exists():
        raise FileExistsError('Do not replace a frozen protocol')
    PROTOCOL.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    print('Prepared protocol for', len(by_user), 'users', flush=True)


def run():
    p = json.loads(PROTOCOL.read_text())
    files = {path.stem: path for path in (Path(p['dataset_root'])/'trials').rglob('*.npz')}
    if not files:
        raise PermissionError('Native cache missing or inaccessible')
    rows = []; states = []; totals = {'references': 0, 'detections': 0, 'matched': 0, 'censored_trials': 0}
    for user, split in p['by_user'].items():
        rest = []; raw_hashes = {}
        for trial_id in split['source_trials']:
            path = files[trial_id]; raw_hashes[trial_id] = sha(path)
            trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.)
            if not trial.benchmark_eligible:
                raise ValueError('Ineligible source trial')
            # Long contiguous native Rest runs; reject very short transition pieces.
            for start, end, label in contiguous_bouts(trial.hand_label):
                if label == 0:
                    rest.append(trial.emg[start:end])
        if not rest:
            raise ValueError('No eligible source Rest')
        detector = AutonomousBoutDetectorV1(**p['policy']).fit_rest(
            np.concatenate(rest), 200., source_trial_ids=split['source_trials'])
        frozen = pickle.dumps((detector.on_, detector.off_, detector.counts_, detector.source_trial_ids_))
        states.append({'user': user, 'on': detector.on_, 'off': detector.off_,
                       'source_rest_samples': sum(len(x) for x in rest), 'source_raw_sha256': raw_hashes})
        print('Replaying', user, len(split['evaluation_trials']), 'Day6 recordings', flush=True)
        for trial_id in split['evaluation_trials']:
            path = files[trial_id]
            trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.)
            if not trial.benchmark_eligible:
                raise ValueError('Ineligible evaluation trial')
            detector.reset(); events = []
            for start in range(0, len(trial.emg), p['chunk_samples']):
                events += detector.feed(trial.emg[start:start+p['chunk_samples']], start, trial_id=trial_id)
            censored = detector.finish()
            assert frozen == pickle.dumps((detector.on_, detector.off_, detector.counts_, detector.source_trial_ids_))
            refs = [(label, a, b) for a, b, label in contiguous_bouts(trial.hand_label) if label != 0]
            candidates = []
            for i, event in enumerate(events):
                for j, (_, a, b) in enumerate(refs):
                    intersection = max(0, min(event.end, b)-max(event.start, a))
                    union = max(event.end, b)-min(event.start, a)
                    iou = intersection/union
                    if iou >= p['match_iou']:
                        candidates.append((iou, i, j))
            used_events = set(); used_refs = set(); matches = []
            for iou, i, j in sorted(candidates, key=lambda v: (-v[0], v[1], v[2])):
                if i not in used_events and j not in used_refs:
                    used_events.add(i); used_refs.add(j)
                    label, a, b = refs[j]; e = events[i]
                    matches.append({'iou': iou, 'event_index': i, 'reference_index': j,
                                    'label': int(label), 'onset_error_s': (e.start-a)/200.,
                                    'offset_error_s': (e.end-b)/200.})
            rows.append({'user': user, 'trial': trial_id, 'raw_sha256': sha(path),
                         'reference_bounds': [[int(v) for v in r] for r in refs],
                         'detected_bounds': [[e.start, e.end] for e in events],
                         'matches': matches, 'censored_end': censored})
            totals['references'] += len(refs); totals['detections'] += len(events)
            totals['matched'] += len(matches); totals['censored_trials'] += int(censored)
    matched = [m for r in rows for m in r['matches']]
    totals.update(precision=totals['matched']/max(totals['detections'], 1),
                  recall=totals['matched']/max(totals['references'], 1),
                  onset_mae_s=float(np.mean([abs(m['onset_error_s']) for m in matched])) if matched else None,
                  offset_mae_s=float(np.mean([abs(m['offset_error_s']) for m in matched])) if matched else None)
    result = {'protocol_sha256': sha(PROTOCOL), 'source_states': states, 'recordings': rows,
              'totals': totals, 'source_hashes': {path: sha(ROOT/path) for path in
                  ('src/emgimu/feature_bank/autonomous_bouts_v1.py', 'benchmarks/new_bank_v3/autonomous_unibo_v1.py')},
              'scope': p['scope'], 'default_promoted': False}
    (HERE/'AUTONOMOUS_UNIBO_V1_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
