"""Frozen continuous detected boundaries into existing source-only DTW templates."""
import argparse
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import load, sha
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.feature_bank.autonomous_bouts_v1 import DetectedBout
from emgimu.feature_bank.document_path_v3 import DocumentTemporalTemplatesV3, document_envelope_path
from emgimu.feature_bank.detected_template_reader_v1 import DetectedTemplateReaderV1
from emgimu.feature_bank.unibo_sequence_temporal import envelope_path, complete_paths

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'DETECTED_DTW_UNIBO_V1_PROTOCOL.json'
BOUNDS = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json'
BOUND_PROTOCOL = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
TEMPLATE_PROTOCOL = HERE/'F5_PATH_UNIBO_PROTOCOL.json'


def prepare():
    p = {'version': 'detected-dtw-unibo-v1', 'boundary_result_sha256': sha(BOUNDS),
         'boundary_protocol_sha256': sha(BOUND_PROTOCOL), 'template_protocol_sha256': sha(TEMPLATE_PROTOCOL),
         'templates': 'Same frozen twenty source candidates/user, five/class, source Days1-5. New document DTW medoids; no candidate reselection or target fitting.',
         'readout': 'Every saved detected native interval into the explicit estimated-boundary full-RMS-path reader. All four source classes including Rest; no forced active class, probability confidence or target label input.',
         'scoring': 'Score only one-to-one IoU>=.5 matched power-grip/two-finger-pinch/open references (native2/3/6 -> canonical2/1/3). Other native active gestures remain unsupported and are reported separately. Misses count in end-to-end reference success.',
         'oracle_control': 'Same matched reference intervals and same source DTW templates; whole interval RMS path, native labels used solely for evaluation/control boundaries.',
         'scope': 'Retrospective continuous Day6 public four-channel offline chain diagnostic. Previous boundaries/sessions inspected. No live eight-channel accuracy, calibrated fusion or default promotion.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse protocol replacement')
    PROTOCOL.write_text(json.dumps(p, indent=2)+'\n', encoding='utf8')
    print('Frozen complete detected-path DTW chain diagnostic', flush=True)


def run():
    p = json.loads(PROTOCOL.read_text())
    for path, key in [(BOUNDS, 'boundary_result_sha256'), (BOUND_PROTOCOL, 'boundary_protocol_sha256'),
                      (TEMPLATE_PROTOCOL, 'template_protocol_sha256')]:
        if sha(path) != p[key]:
            raise ValueError('Frozen parent changed')
    bounds = json.loads(BOUNDS.read_text()); bp = json.loads(BOUND_PROTOCOL.read_text())
    tp = json.loads(TEMPLATE_PROTOCOL.read_text()); source = Path(tp['frozen_source'])
    for filename in ['split_trial_ids.json', 'bout_metadata.json']:
        if sha(source/filename) != tp['frozen_files']['source/'+filename]:
            raise ValueError('Frozen template candidate metadata changed')
    splits = json.loads((source/'split_trial_ids.json').read_text())
    metadata = {r['id']: r for r in json.loads((source/'bout_metadata.json').read_text())}
    files = {f.stem: f for f in (Path(tp['dataset_root'])/'trials').rglob('*.npz')}
    if not files:
        raise PermissionError('Native source cache inaccessible')
    archive = Path(bp['archive'])
    if sha(archive) != bp['archive_sha256']:
        raise ValueError('Continuous archive changed')
    mapping = {2: 2, 3: 1, 6: 3}; rows = []; selections = []
    totals = {'supported_references': 0, 'matched_supported': 0, 'detected_correct': 0,
              'oracle_correct_on_matched': 0, 'unsupported_matched': 0, 'unmatched_detections': 0}
    with zipfile.ZipFile(archive) as z:
        for split in splits:
            user = split['user']; candidates = []
            for bout_id in split['source_template_candidates']:
                r = metadata[bout_id]; path = files[r['trial']]
                if r['day'] > 5 or sha(path) != r['raw_sha256']:
                    raise ValueError('Source candidate identity changed')
                trial = load_benchmark_trial(path, expected_channels=4, expected_rate_hz=200.)
                if not np.all(trial.hand_label[r['start']:r['end']] == r['label']):
                    raise ValueError('Candidate labels differ')
                candidates.append({**r, 'path': envelope_path(trial.emg[r['start']:r['end']])})
            model = DocumentTemporalTemplatesV3().fit(complete_paths(candidates),
                [r['label'] for r in candidates], trial_ids=[r['id'] for r in candidates])
            recording_ids = sorted({r['trial'] for r in candidates})
            reader = DetectedTemplateReaderV1(model, native_sample_rate_hz=200., source_recording_ids=recording_ids)
            frozen = pickle.dumps(model)
            selections.append({'user': user, 'candidate_ids': [r['id'] for r in candidates],
                               'medoid_ids': list(model.medoid_trial_ids_), 'source_recordings': recording_ids})
            print('Replaying full detected DTW paths for', user, flush=True)
            for record in bounds['recordings']:
                if record['user'] != int(user[1:]):
                    continue
                name = record['member']; x, y, original, digest = load(z, name)
                if digest != record['member_sha256']:
                    raise ValueError('Native continuous member changed')
                refs = record['reference_bounds']
                totals['supported_references'] += sum(r[0] in mapping for r in refs)
                matched = {m['event_index']: m for m in record['matches']}
                for index, (a, b) in enumerate(record['detected_bounds']):
                    result = reader.read(DetectedBout(a, b, x[a:b], 200.), recording_id=name)
                    row = {'user': user, 'member': name, 'event_index': index, 'start': a, 'end': b,
                           'prediction': int(result['nearest_class']), 'distances': result['distances'].tolist(),
                           'reference_label': None, 'oracle_prediction': None}
                    if index not in matched:
                        totals['unmatched_detections'] += 1
                    else:
                        m = matched[index]; label, start, end = refs[m['reference_index']]
                        if label not in mapping:
                            totals['unsupported_matched'] += 1
                        else:
                            truth = mapping[label]
                            path = document_envelope_path(envelope_path(x[start:end]))
                            oracle = model._distances([path])[0]
                            row.update(reference_label=truth, reference_start=start, reference_end=end,
                                       oracle_prediction=int(model.classes_[np.argmin(oracle)]), oracle_distances=oracle.tolist())
                            totals['matched_supported'] += 1
                            totals['detected_correct'] += int(row['prediction'] == truth)
                            totals['oracle_correct_on_matched'] += int(row['oracle_prediction'] == truth)
                    rows.append(row)
            assert pickle.dumps(model) == frozen
    totals['matched_detection_accuracy'] = totals['detected_correct']/max(totals['matched_supported'], 1)
    totals['matched_oracle_accuracy'] = totals['oracle_correct_on_matched']/max(totals['matched_supported'], 1)
    totals['end_to_end_reference_success'] = totals['detected_correct']/max(totals['supported_references'], 1)
    evidence = ['benchmarks/new_bank_v3/detected_dtw_unibo_v1.py',
                'benchmarks/new_bank_v3/autonomous_continuous_unibo_v1.py',
                'src/emgimu/feature_bank/document_path_v3.py', 'src/emgimu/feature_bank/detected_template_reader_v1.py']
    result = {'protocol_sha256': sha(PROTOCOL), 'source_hashes': {n: sha(ROOT/n) for n in evidence},
              'source_selections': selections, 'totals': totals, 'events': rows, 'scope': p['scope'],
              'source_state_immutable': True, 'default_promoted': False}
    (HERE/'DETECTED_DTW_UNIBO_V1_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
