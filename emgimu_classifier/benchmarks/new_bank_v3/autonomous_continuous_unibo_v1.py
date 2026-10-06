"""Continuous archive replay, independent of repetition-trial cropping."""
import argparse
import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from emgimu.signal import polyphase_resample
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
ARCHIVE = Path('D:/emg-imu-benchmarks/data/raw/unibo_inail/unibo-inail-main.zip')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    with zipfile.ZipFile(ARCHIVE) as z:
        members = {}
        for name in z.namelist():
            m = re.search(r'user_(\d+)_day_(\d+)_posture_(\d+)\.mat$', name)
            if m:
                user, day, posture = map(int, m.groups())
                if day in (1, 6):
                    members[name] = {'user': user, 'day': day, 'posture': posture}
    assert len(members) == 56
    p = {'version': 'autonomous_continuous_unibo_v1', 'archive': str(ARCHIVE),
         'archive_sha256': sha(ARCHIVE), 'members': members,
         'source_day': 1, 'evaluation_day': 6, 'sample_rate_hz': 200.,
         'source_rest_rule': 'Both original label and relabel=1, contiguous runs >=1 second; discard 125ms at each edge. All four source postures, same user, separate Day1 files.',
         'policy': {'onset_s': .08, 'release_s': .12, 'preroll_s': .10,
                    'min_duration_s': 1., 'max_duration_s': 30.},
         'chunk_samples': 37, 'match_iou': .5,
         'reference': 'Every non-Rest relabel=2..6 contiguous interval >=1s. Counter-zero regions preserved. Protocol labels are scoring references, not physiological onset measurements.',
         'scope': 'Retrospective Day6 continuous public recording screen. No classifier, heldout tuning, real 8-channel device or biological completeness claim; no default promotion.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse to replace frozen protocol')
    PROTOCOL.write_text(json.dumps(p, indent=2)+'\n', encoding='utf8')
    print('Frozen 28 source and 28 evaluation continuous recordings', flush=True)


def load(z, name):
    raw = z.read(name)
    d = loadmat(io.BytesIO(raw), variable_names=('emg', 'label', 'relabel'))
    x = np.asarray(d['emg'], dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('Invalid native EMG')
    x = polyphase_resample(x, 500., 200.)
    positions = np.minimum((np.arange(len(x))*500/200).astype(int), len(d['label'])-1)
    y = np.asarray(d['relabel']).ravel()[positions].astype(int)
    original = np.asarray(d['label']).ravel()[positions].astype(int)
    assert set(y) <= set(range(1, 7))
    return x, y, original, hashlib.sha256(raw).hexdigest()


def runs(y):
    edges = np.r_[0, np.flatnonzero(np.diff(y) != 0)+1, len(y)]
    return [(int(y[a]), int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b-a >= 200]


def score(events, refs):
    candidates = []
    for i, e in enumerate(events):
        for j, (_, a, b) in enumerate(refs):
            intersection = max(0, min(e.end, b)-max(e.start, a))
            union = max(e.end, b)-min(e.start, a)
            if intersection/union >= .5:
                candidates.append((intersection/union, i, j))
    used_i = set(); used_j = set(); matches = []
    for iou, i, j in sorted(candidates, key=lambda v: (-v[0], v[1], v[2])):
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        label, a, b = refs[j]; e = events[i]
        matches.append({'iou': iou, 'event_index': i, 'reference_index': j, 'label': label,
                        'onset_error_s': (e.start-a)/200., 'offset_error_s': (e.end-b)/200.})
    return matches


def run():
    p = json.loads(PROTOCOL.read_text())
    archive = Path(p['archive'])
    if sha(archive) != p['archive_sha256']:
        raise ValueError('Archive hash differs')
    states = []; rows = []
    with zipfile.ZipFile(archive) as z:
        for user in range(1, 8):
            source_names = [n for n, v in p['members'].items() if v['user'] == user and v['day'] == 1]
            target_names = [n for n, v in p['members'].items() if v['user'] == user and v['day'] == 6]
            rest = []; hashes = {}
            print(f'{user}/7: reading source continuous Rest', flush=True)
            for name in source_names:
                x, y, original, digest = load(z, name); hashes[name] = digest
                mask = ((y == 1) & (original == 1)).astype(int)
                for label, a, b in runs(mask):
                    if label == 1:
                        rest.append(x[a+25:b-25])
            if not rest:
                raise ValueError('No eligible source Rest')
            model = AutonomousBoutDetectorV1(**p['policy']).fit_rest(
                np.concatenate(rest), 200., source_trial_ids=source_names)
            frozen = pickle.dumps((model.on_, model.off_, model.counts_, model.source_trial_ids_))
            states.append({'user': user, 'on': model.on_, 'off': model.off_,
                           'source_rest_samples': sum(len(r) for r in rest), 'source_member_sha256': hashes})
            print(f'{user}/7: replaying four uninterrupted Day6 recordings', flush=True)
            for name in target_names:
                x, y, original, digest = load(z, name)
                model.reset(); events = []
                for start in range(0, len(x), p['chunk_samples']):
                    events += model.feed(x[start:start+p['chunk_samples']], start, trial_id=name)
                censored = model.finish()
                assert frozen == pickle.dumps((model.on_, model.off_, model.counts_, model.source_trial_ids_))
                refs = [r for r in runs(y) if r[0] != 1]
                rows.append({'user': user, 'member': name, 'member_sha256': digest,
                             'reference_bounds': refs, 'detected_bounds': [[e.start, e.end] for e in events],
                             'matches': score(events, refs), 'censored_end': censored})
    matches = [m for r in rows for m in r['matches']]
    n_ref = sum(len(r['reference_bounds']) for r in rows)
    n_det = sum(len(r['detected_bounds']) for r in rows)
    totals = {'references': n_ref, 'detections': n_det, 'matched': len(matches),
              'precision': len(matches)/max(n_det, 1), 'recall': len(matches)/max(n_ref, 1),
              'onset_mae_s': float(np.mean([abs(m['onset_error_s']) for m in matches])) if matches else None,
              'offset_mae_s': float(np.mean([abs(m['offset_error_s']) for m in matches])) if matches else None}
    result = {'protocol_sha256': sha(PROTOCOL), 'totals': totals, 'source_states': states, 'recordings': rows,
              'source_hashes': {name: sha(ROOT/name) for name in
                  ('src/emgimu/feature_bank/autonomous_bouts_v1.py', 'benchmarks/new_bank_v3/autonomous_continuous_unibo_v1.py')},
              'scope': p['scope'], 'default_promoted': False}
    (HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
