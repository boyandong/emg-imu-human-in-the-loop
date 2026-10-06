"""Native chunk-to-prediction replay against frozen boundary and G5 results."""
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from benchmarks.new_bank_v3.autonomous_continuous_unibo_v1 import load, runs, sha
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1
from emgimu.feature_bank.autonomous_g5_stream_v1 import AutonomousG5StreamV1
from emgimu.feature_bank.detected_g5_reader_v1 import DetectedG5ReaderV1

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run():
    bp_path = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
    br_path = HERE/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json'
    gp_path = HERE/'DETECTED_G5_UNIBO_V1_PROTOCOL.json'
    gr_path = HERE/'DETECTED_G5_UNIBO_V1_RESULTS.json'
    bp, br, gp, gr = [json.loads(p.read_text()) for p in (bp_path, br_path, gp_path, gr_path)]
    if sha(bp_path) != br['protocol_sha256'] or sha(gp_path) != gr['protocol_sha256']:
        raise ValueError('Frozen parent protocol changed')
    state_path = Path(gp['source_root'])/'fitted_states.pkl'
    if sha(state_path) != gp['source_state_sha256'] or sha(Path(bp['archive'])) != bp['archive_sha256']:
        raise ValueError('Frozen native archive or classifier changed')
    states, temperatures = pickle.loads(state_path.read_bytes())
    frozen = pickle.dumps((states, temperatures)); records = []
    reference = {r['member']: r for r in br['recordings']}
    with zipfile.ZipFile(bp['archive']) as archive:
        for user in range(1,8):
            key = f'u{user:02d}'
            names = [n for n,v in bp['members'].items() if v['user'] == user and v['day'] == 1]
            rest = []
            for name in names:
                x, y, original, digest = load(archive, name)
                for label, a, b in runs(((y == 1) & (original == 1)).astype(int)):
                    if label == 1:
                        rest.append(x[a+25:b-25])
            detector = AutonomousBoutDetectorV1(**bp['policy']).fit_rest(np.concatenate(rest), 200., source_trial_ids=names)
            previous = next(r for r in br['source_states'] if r['user'] == user)
            if (detector.on_, detector.off_) != (previous['on'], previous['off']):
                raise ValueError('Source detector thresholds differ')
            family, scaler, classifier = states[key][1]['g5']
            source_ids = next(r['source_recordings'] for r in gr['source_selections'] if r['user'] == key)
            reader = DetectedG5ReaderV1(family, scaler, classifier, source_recording_ids=source_ids,
                                       temperature=temperatures[key]['G5'])
            for name,v in bp['members'].items():
                if v['user'] != user or v['day'] != 6:
                    continue
                x, y, original, digest = load(archive, name)
                if digest != reference[name]['member_sha256']:
                    raise ValueError('Native recording changed')
                stream = AutonomousG5StreamV1(detector, reader); emitted = []
                for start in range(0,len(x),bp['chunk_samples']):
                    emitted += stream.feed(x[start:start+bp['chunk_samples']], start, recording_id=name)
                censored = stream.finish()
                old = sorted([e for e in gr['events'] if e['member'] == name], key=lambda e:e['event_index'])
                if len(emitted) != len(old) or censored != reference[name]['censored_end']:
                    raise ValueError('Stream detector output differs')
                errors = []
                for a,b in zip(emitted, old):
                    if (a['start'], a['end'], a['prediction']) != (b['start'], b['end'], b['g5_prediction']):
                        raise ValueError('Stream boundary or classification differs')
                    errors.append(float(np.max(np.abs(a['probability']-b['g5_probability']))))
                    if not 0 <= a['delivery_delay_samples'] < bp['chunk_samples']:
                        raise ValueError('Noncausal chunk delivery')
                if max(errors,default=0.) > 1e-12:
                    raise ValueError('Stream probability changed')
                records.append({'member':name, 'events':len(emitted), 'max_probability_error':max(errors,default=0.),
                                'censored_end':censored, 'max_chunk_delay_samples':max((e['delivery_delay_samples'] for e in emitted),default=0)})
            print(f'{user}/7 streaming replay verified', flush=True)
    if frozen != pickle.dumps((states,temperatures)):
        raise ValueError('Source predictor state mutated')
    paths = [bp_path, br_path, gp_path, gr_path, Path(__file__),
             ROOT/'src/emgimu/feature_bank/autonomous_g5_stream_v1.py']
    result = {'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}, 'recordings':records,
              'source_state_immutable':True, 'classifier_refitted':False, 'chunk_samples':bp['chunk_samples'],
              'scope':'Native four-channel/200Hz continuous public replay. Chunk-to-release-confirmed prediction parity; no hardware throughput, onset-time classification or eight-channel transfer claim.',
              'default_promoted':False, 'completion_proven':False}
    (HERE/'AUTONOMOUS_G5_STREAM_V1_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    run()
