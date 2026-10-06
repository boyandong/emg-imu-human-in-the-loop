"""Read-only native G5 classifier replay on the fixed detected/DTW intervals."""
import argparse
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import load, sha
from emgimu.feature_bank.autonomous_bouts_v1 import DetectedBout
from emgimu.feature_bank.detected_g5_reader_v1 import DetectedG5ReaderV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'DETECTED_G5_UNIBO_V1_PROTOCOL.json'
CHAIN = HERE/'DETECTED_DTW_UNIBO_V1_RESULTS.json'
BOUND_PROTOCOL = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
SOURCE_PROTOCOL = HERE/'F5_PATH_UNIBO_PROTOCOL.json'


def prepare():
    source = json.loads(SOURCE_PROTOCOL.read_text())
    p = {'version': 'detected-g5-unibo-v1', 'dtw_chain_sha256': sha(CHAIN),
         'boundary_protocol_sha256': sha(BOUND_PROTOCOL), 'source_protocol_sha256': sha(SOURCE_PROTOCOL),
         'source_root': source['frozen_source'],
         'source_state_sha256': source['frozen_files']['source/fitted_states.pkl'],
         'source_metadata_sha256': source['frozen_files']['source/bout_metadata.json'],
         'selection': 'Same1257 detected intervals and same750 supported matched references from the locked DTW chain, no new boundary/classifier selection.',
         'model': 'Existing source-only G5 feature/scaler/classifier per user, final source Days1-5 state, source Day5 temperature. No fitting or parameter change.',
         'aggregation': 'Mean of all complete nonoverlapping40-sample/200ms windows within the native detected interval; remainder reported, no padding. Match existing source training exactly.',
         'controls': 'Same-reference G5 oracle intervals and the saved detected/oracle DTW predictions. Retain every unsupported and unmatched event separately.',
         'scope': 'Previously inspected public four-channel/200Hz Day6 continuous diagnostic. No new training, live8-channel deployment, threshold selection, or default promotion.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse protocol replacement')
    PROTOCOL.write_text(json.dumps(p, indent=2)+'\n', encoding='utf8')
    print('Frozen existing G5 classifier replay with identical DTW intervals', flush=True)


def run():
    p = json.loads(PROTOCOL.read_text())
    for path, key in [(CHAIN, 'dtw_chain_sha256'), (BOUND_PROTOCOL, 'boundary_protocol_sha256'),
                      (SOURCE_PROTOCOL, 'source_protocol_sha256')]:
        if sha(path) != p[key]:
            raise ValueError('Frozen parent changed')
    source = Path(p['source_root']); state_path = source/'fitted_states.pkl'; meta_path = source/'bout_metadata.json'
    if sha(state_path) != p['source_state_sha256'] or sha(meta_path) != p['source_metadata_sha256']:
        raise ValueError('Frozen source state or metadata changed')
    states, temperatures = pickle.loads(state_path.read_bytes()); frozen = pickle.dumps((states, temperatures))
    metadata = json.loads(meta_path.read_text()); chain = json.loads(CHAIN.read_text())
    bp = json.loads(BOUND_PROTOCOL.read_text()); archive = Path(bp['archive'])
    if sha(archive) != bp['archive_sha256']:
        raise ValueError('Native archive changed')
    rows = []; source_records = []
    with zipfile.ZipFile(archive) as z:
        for user in sorted(states):
            source_ids = sorted({r['trial'] for r in metadata if r['user'] == user and r['day'] <= 5})
            family, scaler, classifier = states[user][1]['g5']
            reader = DetectedG5ReaderV1(family, scaler, classifier, source_recording_ids=source_ids,
                                       temperature=temperatures[user]['G5'])
            source_records.append({'user': user, 'source_recordings': source_ids,
                                   'source_temperature': temperatures[user]['G5']})
            print('Read-only G5 replay for', user, flush=True)
            selected = [r for r in chain['events'] if r['user'] == user]
            for name in sorted({r['member'] for r in selected}):
                x, y, original, digest = load(z, name)
                for event in [r for r in selected if r['member'] == name]:
                    a, b = event['start'], event['end']
                    observed = reader.read(DetectedBout(a, b, x[a:b], 200.), recording_id=name)
                    row = {**event, 'g5_prediction': observed['prediction'],
                           'g5_probability': observed['probability'].tolist(),
                           'complete_windows': observed['complete_windows'],
                           'unrepresented_tail_samples': observed['unrepresented_tail_samples'],
                           'g5_oracle_prediction': None}
                    if event['reference_label'] is not None:
                        start, end = event['reference_start'], event['reference_end']
                        oracle = reader.read(DetectedBout(start, end, x[start:end], 200.), recording_id=name)
                        row.update(g5_oracle_prediction=oracle['prediction'], g5_oracle_probability=oracle['probability'].tolist())
                    rows.append(row)
    assert frozen == pickle.dumps((states, temperatures))
    supported = [r for r in rows if r['reference_label'] is not None]
    totals = {**chain['totals'], 'g5_detected_correct': sum(r['g5_prediction'] == r['reference_label'] for r in supported),
              'g5_oracle_correct': sum(r['g5_oracle_prediction'] == r['reference_label'] for r in supported)}
    totals.update(g5_matched_detection_accuracy=totals['g5_detected_correct']/len(supported),
                  g5_matched_oracle_accuracy=totals['g5_oracle_correct']/len(supported),
                  g5_end_to_end_reference_success=totals['g5_detected_correct']/totals['supported_references'])
    evidence = ['benchmarks/new_bank_v3/detected_g5_unibo_v1.py', 'src/emgimu/feature_bank/detected_g5_reader_v1.py',
                'src/emgimu/feature_bank/validated_unibo.py', 'src/emgimu/feature_bank/force_nested_oof.py']
    r = {'protocol_sha256': sha(PROTOCOL), 'source_hashes': {n: sha(ROOT/n) for n in evidence},
         'source_selections': source_records, 'source_state_immutable': True, 'classifier_refitted': False,
         'events': rows, 'totals': totals, 'scope': p['scope'], 'default_promoted': False}
    (HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json').write_text(json.dumps(r, indent=2)+'\n', encoding='utf8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
