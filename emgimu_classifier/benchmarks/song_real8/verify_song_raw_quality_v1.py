"""No-fit native F9 verification with independent observations, fusion and decoding.

The recordings have been inspected previously. Full-stream checks establish
numerical integration, including calibration intervals, never new efficacy.
"""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import h5py
import numpy as np
from benchmarks.song_real8.song_raw_quality_v1 import native, SCENARIOS, KNOWN_FAULTS
from benchmarks.song_real8_study import _filter_emg
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v2 import PersonalSessionStreamV2, load_gate

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json'
RESULT = HERE/'SONG_RAW_QUALITY_V1_RESULTS.json'
OUT = ROOT/'feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json'
SOURCE = HERE/'song_personal_session_v1/source_bank.pkl'
ACCEPTANCE = ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(x, gate):
    """Recompute all masked metrics without calling the quality observer."""
    x = np.asarray(x, float)
    zero = (abs(x) < .5).mean(1)
    current = np.zeros((len(x), 8), int)
    longest = current.copy()
    for edge in np.moveaxis(abs(np.diff(x, axis=1)) < .5, 1, 0):
        current = np.where(edge, current + 1, 0)
        longest = np.maximum(longest, current)
    lo, hi = gate.observer.adc_range
    tolerance = (hi-lo)*1e-6
    clip = ((x <= lo+tolerance) | (x >= hi-tolerance)).mean(1)
    rms = np.sqrt((x*x).mean(1))
    z = (rms-gate.observer.reference_median_)/(1.4826*gate.observer.reference_mad_+1e-10)
    frequency = np.fft.rfftfreq(50, 1/250.)
    power = abs(np.fft.rfft((x-x.mean(1, keepdims=True))*np.hanning(50)[None, :, None], axis=1))**2
    line = abs(frequency-50) <= 1
    neighbors = ((frequency >= 44) & (frequency <= 47)) | ((frequency >= 53) & (frequency <= 56))
    line_ratio = power[:, line].mean(1)/(power[:, neighbors].mean(1)+1e-10)
    low_ratio = power[:, frequency <= 10].sum(1)/(power[:, (frequency >= 20) & (frequency <= 118.75)].sum(1)+1e-10)
    # The frozen observer exports float32 before mask arithmetic.
    return {k: np.asarray(v, np.float32).astype(float) for k, v in
            dict(zero_fraction=zero, longest_flatline_ratio=longest/50.,
                 clip_fraction=clip, amplitude_z=z, line_noise_ratio=line_ratio,
                 low_frequency_ratio=low_ratio).items()}


def channel_masks(x, ids, gate):
    metrics = observations(x, gate)
    soft = np.ones((len(x), 8))
    for name, threshold in gate.thresholds.items():
        value = abs(metrics[name]) if name == 'amplitude_z' else metrics[name]
        soft *= 1-np.clip(value/threshold-1, 0, 1)
    bad = ((metrics['zero_fraction'] >= .5) | (metrics['longest_flatline_ratio'] >= .5)
           | (metrics['clip_fraction'] >= .1))
    trials = tuple(np.unique(ids))
    return trials, np.stack([soft[ids == t].min(0) for t in trials]), np.stack([bad[ids == t].any(0) for t in trials])


def fuse(providers, weights, channels, bad, mode):
    masks = np.ones((len(channels), len(providers)))
    if mode == 'structural':
        masks[:] = (~bad.any(1))[:, None]
    elif mode == 'soft':
        masks = np.column_stack([channels.min(1) if name in ('F0', 'F2ac') else channels.mean(1) for name in providers])
    effective = masks*np.asarray(weights)[None, :]
    rejected = effective.sum(1) <= 1e-10
    effective[rejected] = weights
    effective /= effective.sum(1, keepdims=True)
    q = np.einsum('ng,ngc->nc', effective, np.stack(list(providers.values()), axis=1))
    q /= q.sum(1, keepdims=True)
    return q, rejected


def altered_recording(raw, scenario, starts, source_raw, amplitude, adc_range):
    x = raw.astype(float).copy()
    if scenario == 'dropout_ch3': x[:, 2] = 0
    elif scenario == 'flat_ch3': x[:, 2] = np.median(source_raw[:, :, 2])
    elif scenario == 'transport_rail_ch3': x[:, 2] = adc_range[1]
    elif scenario == 'flat26_ch3':
        for start in starts: x[start:start+26, 2] = x[start, 2]
    elif scenario in ('line50_source_rms', 'low5_source_rms'):
        hz = 50 if scenario.startswith('line') else 5
        x += amplitude[None, :]*np.sin(2*np.pi*hz*np.arange(len(x))/250.)[:, None]
    elif scenario == 'gain2_ch3': x[:, 2] *= 2
    return x


def decode(q, rejected, classes, state):
    candidate, active, count = state
    labels = []
    for probability, reject in zip(q, rejected):
        if reject: candidate, active, count = None, None, 0
        else:
            label = classes[int(probability.argmax())]
            count = count+1 if label == candidate else 1
            candidate = label
            if count >= 2: active = label
        labels.append(active)
    return labels, (candidate, active, count)


def run():
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    r = json.loads(RESULT.read_text(encoding='utf8'))
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for name, digest in {**p['source_sha256'], **p['artifact_sha256']}.items():
        assert sha(REPO/name) == digest, name
    for name, digest in r['artifact_sha256'].items(): assert sha(ROOT/name) == digest, name
    gate = load_gate(ROOT/r['gate_path'], RESULT)
    w = load_workflow(SOURCE, ACCEPTANCE)
    personal = w.load_profile(HERE/'song_personal_session_v1/personal_S03.zip', user_id='Song')
    session = w.load_profile(HERE/'song_personal_session_v1/session_S04_5shot.zip', user_id='Song', session_id='S04', personal=personal)
    before = pickle.dumps((gate, w.bank, personal, session))
    source = [native(s, p) for s in p['source_sessions']]
    sx = np.concatenate([v[3].emg for v in source]).astype(float)
    si = np.concatenate([v[0]['trial'] for v in source])
    assert tuple(np.unique(si)) == gate.source_trials and len(sx) == r['source_windows'] == 849
    rms = np.sqrt((sx*sx).mean(1));median = np.median(rms, axis=0)
    np.testing.assert_array_equal(gate.observer.reference_median_, median)
    np.testing.assert_array_equal(gate.observer.reference_mad_, np.median(abs(rms-median), axis=0))
    centered = sx-sx.mean(1, keepdims=True)
    covariance = np.stack([v.T@v/49 for v in centered]).mean(0)
    np.testing.assert_allclose(gate.observer.reference_covariance_, covariance, rtol=2e-14, atol=1e-10)
    metrics = observations(sx, gate)
    for name, floor in dict(zero_fraction=.1, longest_flatline_ratio=.1, amplitude_z=6., clip_fraction=.01,
                            line_noise_ratio=10., low_frequency_ratio=10.).items():
        values = abs(metrics[name]) if name == 'amplitude_z' else metrics[name]
        np.testing.assert_array_equal(gate.thresholds[name], np.maximum(np.quantile(values, .995, axis=0), floor))
    amplitude = np.median(rms, axis=0)
    np.testing.assert_array_equal(amplitude, r['source_amplitude'])
    item, raw, starts, _ = native('S04', p)
    mask = np.isin(item['trial'], r['evaluation_ids']);starts = starts[mask];ids = item['trial'][mask]
    offsets = np.zeros(len(ids), int)
    for t in np.unique(ids): offsets[ids == t] = np.arange(np.sum(ids == t))
    with (HERE/'song_raw_quality_v1/predictions.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    arrays = np.load(HERE/'song_raw_quality_v1/readouts.npz', allow_pickle=False)
    assert len(rows) == 2976 and len(r['cells']) == 24
    classes = list(arrays['class_names']);maximum = 0.;unmodified = {}
    for scenario in SCENARIOS:
        altered = altered_recording(raw, scenario, starts, sx, amplitude, p['adc_range'])
        filtered = _filter_emg(altered, 'causal')
        eb = FeatureBatch(np.stack([filtered[s:s+50] for s in starts]), 250.)
        rb = np.stack([altered[s:s+50] for s in starts])
        trial_axis, soft, bad = channel_masks(rb, ids, gate)
        assert list(trial_axis) == r['evaluation_ids']
        providers = w.bank.predict_providers(eb, ids, window_offsets=offsets, user_id='Song')['probabilities']
        for mode in p['modes']:
            q, rejected = fuse(providers, session.fusion_state.weights, soft, bad, mode)
            records = [v for v in rows if (v['scenario'], v['mode']) == (scenario, mode)]
            expected = np.array([[float(v['p_'+c]) for c in classes] for v in records])
            assert [v['trial_id'] for v in records] == list(trial_axis)
            assert [v['label'] for v in records] == r['evaluation_labels']
            maximum = max(maximum, float(abs(q-expected).max()))
            np.testing.assert_allclose(q, expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(q, arrays[scenario+'_'+mode+'_probabilities'], rtol=0, atol=1e-12)
            channel = soft if mode == 'soft' else (~bad).astype(float) if mode == 'structural' else np.ones_like(soft)
            np.testing.assert_allclose(channel, arrays[scenario+'_'+mode+'_channel_quality'], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(rejected, [v['rejected'] == 'True' for v in records])
            labels = np.asarray(classes)[q.argmax(1)];labels = labels.astype(object);labels[rejected] = 'Unknown'
            assert list(labels) == [v['predicted_label'] for v in records]
            if scenario == 'unmodified': unmodified[mode] = labels == np.asarray(r['evaluation_labels'])
        print(scenario+': independent raw masks and all3 fusion modes verified', flush=True)
    with (HERE/'song_personal_session_v1/predictions.csv').open(encoding='utf8', newline='') as f:
        old = [v for v in csv.DictReader(f) if v['arm'] == 'session' and int(v['shots']) == 5]
    assert [v['trial_id'] for v in old] == r['evaluation_ids']
    oldq = np.array([[float(v['p_'+c]) for c in classes] for v in old])
    np.testing.assert_allclose(oldq, arrays['unmodified_off_probabilities'], rtol=0, atol=1e-12)
    records = []
    fullraw_path = Path(p['source_folder'])/'2026-09-18_S04/session.h5'
    with h5py.File(fullraw_path) as f:
        fullraw = f['streams/emg/raw'][:];indices = f['streams/emg/sample_index'][:]
    assert np.array_equal(indices, np.arange(len(fullraw)))
    filtered = _filter_emg(fullraw, 'causal')
    for arm in ('population', 'session_5shot'):
        for mode in ('off', 'structural'):
            service = PersonalSessionStreamV2(w, gate, user_id='Song', session_id='S04', channel_ids=w.channels)
            if arm == 'session_5shot':
                service.command(dict(op='profiles', personal=str(HERE/'song_personal_session_v1/personal_S03.zip'),
                                     session=str(HERE/'song_personal_session_v1/session_S04_5shot.zip')))
            weights = w.bank.policy_.population if service.session is None else service.session.fusion_state.weights
            saved = pickle.dumps((service.personal, service.session))
            service.command(dict(op='quality', mode=mode));service.command(dict(op='recognize'))
            state = (None, None, 0);count = rejects = 0;error = 0.
            for start in range(0, len(fullraw), 4096):
                output = service.ingest(fullraw[start:start+4096], indices[start:start+4096])
                if 'probabilities' not in output: continue
                ends = np.asarray(output['output_sample_indices'])
                vids = np.array([f'verification:{v:09}' for v in ends])
                xb = FeatureBatch(np.stack([filtered[e-49:e+1] for e in ends]), 250.)
                providers = w.bank.predict_providers(xb, vids, window_offsets=np.zeros(len(ends), int), user_id='Song')['probabilities']
                _, soft, bad = channel_masks(np.stack([fullraw[e-49:e+1] for e in ends]), vids, gate)
                q, rejected = fuse(providers, weights, soft, bad, mode)
                error = max(error, float(abs(q-output['probabilities']).max()))
                np.testing.assert_allclose(q, output['probabilities'], rtol=0, atol=1e-12)
                np.testing.assert_array_equal(rejected, output['quality_rejected'])
                confirmed, state = decode(q, rejected, classes, state)
                assert confirmed == output['confirmed_labels']
                count += len(ends);rejects += int(rejected.sum())
            assert saved == pickle.dumps((service.personal, service.session))
            records.append(dict(arm=arm, mode=mode, windows=count, rejected_windows=rejects,
                                maximum_probability_error=error, confirmation_exact=True,
                                source_and_profiles_immutable=True))
            print(f'{arm}/{mode}: {count} complete-recording emissions verified', flush=True)
    assert pickle.dumps((gate, w.bank, personal, session)) == before
    paths = [Path(__file__), ROOT/'tests/test_song_raw_quality_v1_delivery.py', PROTOCOL, RESULT]
    acceptance = dict(schema='song_raw_quality_acceptance_v1', protocol_sha256=sha(PROTOCOL),
        source_sha256={v.relative_to(ROOT).as_posix(): sha(v) for v in paths},
        artifact_sha256=r['artifact_sha256'], checked_cells=24, checked_trial_probabilities=2976,
        source_trials=285, source_windows=849, maximum_probability_error=maximum,
        source_observation_and_quantile_oracles=True, baseline_probability_parity=True,
        unmodified_soft_correct_to_wrong=int(np.sum(unmodified['off'] & ~unmodified['soft'])),
        unmodified_soft_wrong_to_correct=int(np.sum(~unmodified['off'] & unmodified['soft'])),
        full_stream_records=records, source_and_profiles_immutable=True,
        default_promoted=False, physical_validation_proven=False, completion_proven=False,
        scope='Raw pre-software-highpass synthetic fault guards and independent no-fit fusion oracles. Full S04 streams include calibration data and verify numerical integration only. Unmodified fault truth is unknown; physical false-positive rate is unavailable. Soft routing harms are retained. Not full F0-F9 or hardware efficacy.')
    OUT.write_text(json.dumps(acceptance, indent=2)+'\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    run()
