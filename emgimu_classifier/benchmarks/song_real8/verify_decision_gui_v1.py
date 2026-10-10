"""Precommitted no-fit native stream verification for operational F7/F8/raw F9.

Complete recordings include calibration intervals. This establishes execution
equivalence and synthetic rejection, never new classification/device efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import h5py
import numpy as np
from benchmarks.song_real8_study import _filter_emg
from benchmarks.song_real8.verify_integrated_decision_v1 import covariance_oracle, probability
from benchmarks.song_real8.verify_song_raw_quality_v1 import channel_masks, fuse, decode
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_decision_cli_v1 import load_decision_workflow
from emgimu.feature_bank.personal_session_stream_v3 import PersonalSessionStreamV3

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'SONG_DECISION_GUI_V1_PROTOCOL.json'
OUT = ROOT/'feature_bank/SONG_DECISION_GUI_V1_ACCEPTANCE.json'
EMISSIONS = HERE/'song_decision_gui_v1/emissions.npz'
FILES = [
    'emgimu_classifier/src/emgimu/feature_bank/personal_session_stream_v3.py',
    'emgimu_classifier/tests/test_personal_session_stream_v3.py',
    'emgimu_classifier/benchmarks/song_real8/verify_decision_gui_v1.py',
    'collection/emg_meta/emg_meta/emgforce/inference/personal_session_worker_v3.py',
    'collection/emg_meta/emg_meta/emgforce/ui/realtime_inference_page_v3.py',
    'collection/emg_meta/emg_meta/emgforce/ui/main_window_v2.py',
    'collection/emg_meta/emg_meta/emgforce/ui/__init__.py',
    'collection/emg_meta/emg_meta/tests/test_decision_page_v3.py',
]
ARMS = [
    dict(name='population', personal=False, session=False, anchor=False, routing=False, anchor_mode='blended', quality='off'),
    dict(name='session_baseline', personal=True, session=True, anchor=False, routing=False, anchor_mode='blended', quality='off'),
    dict(name='personal_anchor', personal=True, session=False, anchor=True, routing=False, anchor_mode='long_term', quality='off'),
    dict(name='session_router', personal=True, session=True, anchor=False, routing=True, anchor_mode='blended', quality='off'),
    dict(name='session_anchor_long', personal=True, session=True, anchor=True, routing=False, anchor_mode='long_term', quality='off'),
    dict(name='session_anchor_local', personal=True, session=True, anchor=True, routing=False, anchor_mode='local', quality='off'),
    dict(name='full_off', personal=True, session=True, anchor=True, routing=True, anchor_mode='blended', quality='off'),
    dict(name='full_structural', personal=True, session=True, anchor=True, routing=True, anchor_mode='blended', quality='structural'),
    dict(name='full_soft', personal=True, session=True, anchor=True, routing=True, anchor_mode='blended', quality='soft'),
    dict(name='full_dropout_structural', personal=True, session=True, anchor=True, routing=True, anchor_mode='blended', quality='structural', dropout=True),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    raw = json.loads((HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json').read_text(encoding='utf8'))
    decision = json.loads((HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    sources = {**raw['source_sha256'], **{f'emgimu_classifier/{k}': v for k, v in decision['source_sha256'].items()}}
    sources.update({name: sha(REPO/name) for name in FILES})
    artifacts = [
        'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl',
        'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
        'benchmarks/song_real8/song_raw_quality_v1/source_gate.pkl',
        'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json',
        'benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_RESULTS.json',
        'feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json',
        'benchmarks/song_real8/song_integrated_decision_v1/decision_policy.json',
        'benchmarks/song_real8/song_integrated_decision_v1/personal_S03.zip',
        'benchmarks/song_real8/song_integrated_decision_v1/session_S04_5shot.zip',
        'benchmarks/song_real8/verify_integrated_decision_v1.py',
        'benchmarks/song_real8/verify_song_raw_quality_v1.py',
        'benchmarks/song_real8_study.py',
    ]
    protocol = dict(schema='song_decision_gui_v1_protocol', source_sha256=sources,
        artifact_sha256={name: sha(ROOT/name) for name in artifacts}, arms=ARMS,
        native_path=str(Path(raw['source_folder'])/'2026-09-18_S04/session.h5'),
        native_sha256=raw['hdf5_sha256']['S04'], chunk_pattern=[4096, 777, 137, 2048],
        sample_rate_hz=250, window_samples=50, hop_samples=10, consecutive_frames=2,
        tolerance=1e-12, default_promoted=False,
        scope='Previously inspected single-user/day full S04 including calibration intervals. No fitting, labels, future context, selection, accuracy or device claims. Immutable pre-existing profiles and policy; injected channel3 zero is synthetic. Initial branches remain off.')
    with PROTOCOL.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(protocol, stream, indent=2)
        stream.write('\n')


def oracle(w, personal, session, arm, batch, raw, ids):
    """No service/decision predict or runtime anchor/gate implementation calls."""
    offsets = np.zeros(len(ids), int)
    readout = w.bank.predict_providers(batch, ids, window_offsets=offsets, user_id='Song')
    weights = np.array(w.bank.policy_.population if personal is None else
        personal.base.fusion_state.fusion_state.weights if session is None else session.base.fusion_state.weights)
    providers = dict(readout['probabilities'])
    if personal is not None and arm['anchor']:
        mode = 'long_term' if session is None else arm['anchor_mode']
        matrices = covariance_oracle(batch, ids)
        for name in providers:
            if name == 'F2ac':
                prototypes = personal.spd_prototypes if mode == 'long_term' else session.spd_local if mode == 'local' else session.spd_blended
                q, _ = probability(matrices, prototypes, spd=True)
            else:
                anchor = personal.base.anchors[name] if mode == 'long_term' else session.base.anchors[name][mode]
                axis = [list(anchor.classes_).index(c) for c in w.bank.classes_]
                q, _ = probability(readout['source_standardized_features'][name], anchor.prototypes_[axis])
            providers[name] = (1-w.anchor_mix)*providers[name]+w.anchor_mix*q
    if session is not None and arm['routing']:
        # The immutable calibration-only routing risks/prototypes are separately
        # reconstructed by the bound integrated-decision acceptance.
        weights *= np.maximum(np.exp(-np.clip(session.routing['risks'], 0, 5)), .05)
        weights /= weights.sum()
    axis, channels, bad = channel_masks(raw.emg, ids, w.gate)
    assert tuple(axis) == tuple(readout['trial_ids'])
    q, rejected = fuse(providers, weights, channels, bad, arm['quality'])
    return q, rejected, weights


def run():
    if OUT.exists() or EMISSIONS.exists():
        raise FileExistsError('Native verification already recorded; do not silently rerun/overwrite')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name, digest in protocol['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    for name, digest in protocol['artifact_sha256'].items():
        assert sha(ROOT/name) == digest, name
    native = Path(protocol['native_path'])
    assert sha(native) == protocol['native_sha256']
    with h5py.File(native) as f:
        original = f['streams/emg/raw'][:]
        indices = f['streams/emg/sample_index'][:]
    assert np.array_equal(indices, np.arange(len(original)))
    w = load_decision_workflow(HERE/'song_personal_session_v1/source_bank.pkl',
        ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
        HERE/'song_integrated_decision_v1/decision_policy.json',
        HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json',
        HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    pp = HERE/'song_integrated_decision_v1/personal_S03.zip'
    sp = HERE/'song_integrated_decision_v1/session_S04_5shot.zip'
    personal = w.load_profile(pp, user_id='Song')
    session = w.load_profile(sp, user_id='Song', session_id='S04', personal=personal)
    before = pickle.dumps((w, personal, session))
    records, emissions = [], {}
    for arm in protocol['arms']:
        started = time.monotonic()
        service = PersonalSessionStreamV3(w, user_id='Song', session_id='S04', channel_ids=w.channels)
        if arm['personal']:
            service.command(dict(op='profiles', personal=str(pp), session=str(sp) if arm['session'] else None))
        service.command(dict(op='decision', use_anchor=arm['anchor'], use_session_routing=arm['routing'], anchor_mode=arm['anchor_mode']))
        service.command(dict(op='quality', mode=arm['quality']))
        service.command(dict(op='recognize'))
        raw = original.copy()
        if arm.get('dropout'):
            raw[:, 2] = 0
        filtered = _filter_emg(raw, 'causal')
        q_values, rejected_values, labels, emitted = [], [], [], []
        maximum_error = 0.
        state = (None, None, 0)
        start, chunk = 0, 0
        while start < len(raw):
            stop = min(start+protocol['chunk_pattern'][chunk % len(protocol['chunk_pattern'])], len(raw))
            result = service.ingest(raw[start:stop], indices[start:stop])
            start, chunk = stop, chunk+1
            if 'probabilities' not in result:
                continue
            ends = np.asarray(result['output_sample_indices'])
            ids = np.array([f'oracle:{end:09}' for end in ends])
            batch = FeatureBatch(np.stack([filtered[end-49:end+1] for end in ends]), 250.)
            raw_batch = FeatureBatch(np.stack([raw[end-49:end+1] for end in ends]), 250.)
            expected, rejected, weights = oracle(w, personal if arm['personal'] else None,
                session if arm['session'] else None, arm, batch, raw_batch, ids)
            error = float(np.max(abs(expected-result['probabilities'])))
            maximum_error = max(maximum_error, error)
            assert error <= protocol['tolerance'], (arm['name'], error)
            np.testing.assert_array_equal(rejected, result['quality_rejected'])
            np.testing.assert_allclose(result['weights'], weights, rtol=0, atol=protocol['tolerance'])
            reference, state = decode(expected, rejected, w.bank.class_names_, state)
            assert reference == result['confirmed_labels']
            emitted.extend(ends)
            q_values.extend(result['probabilities'])
            rejected_values.extend(rejected)
            labels.extend(-1 if label is None else list(w.bank.class_names_).index(label) for label in reference)
        expected_ends = np.arange(49, len(raw), 10)
        np.testing.assert_array_equal(emitted, expected_ends)
        assert pickle.dumps((w, personal, session)) == before
        assert service.info()['mode'] == 'recognizing'
        emissions[arm['name']+'_probabilities'] = np.asarray(q_values)
        emissions[arm['name']+'_rejected'] = np.asarray(rejected_values)
        emissions[arm['name']+'_confirmed'] = np.asarray(labels, np.int8)
        count = int(np.sum(rejected_values))
        if arm.get('dropout'):
            assert count == len(emitted) and set(labels) == {-1}
        records.append(dict(arm=arm['name'], windows=len(emitted), rejected=count,
            maximum_probability_error=maximum_error, confirmations_exact=True,
            source_and_profiles_immutable=True, elapsed_seconds=time.monotonic()-started))
        print(f"{arm['name']}: {len(emitted)} windows verified, {count} rejected, error {maximum_error:.3g}", flush=True)
    np.testing.assert_array_equal(emissions['full_off_probabilities'], emissions['full_structural_probabilities'])
    emissions['output_sample_indices'] = expected_ends
    EMISSIONS.parent.mkdir(exist_ok=True)
    with EMISSIONS.open('xb') as stream:
        np.savez_compressed(stream, **emissions)
    output = dict(schema='song_decision_gui_v1_acceptance', protocol_sha256=sha(PROTOCOL),
        source_sha256=protocol['source_sha256'], artifact_sha256=protocol['artifact_sha256'],
        emissions_path=EMISSIONS.relative_to(ROOT).as_posix(), emissions_sha256=sha(EMISSIONS),
        records=records, verified_windows=sum(r['windows'] for r in records),
        independent_filter_and_geometry_probabilities=True, chronological_confirmation_exact=True,
        source_and_profiles_immutable=True, unmodified_structural_off_probability_identity=True,
        actual_qt_subprocess_lifecycle_test='collection/emg_meta/emg_meta/tests/test_decision_page_v3.py',
        default_promoted=False, physical_validation_proven=False, completion_proven=False,
        scope=protocol['scope'])
    OUT.write_text(json.dumps(output, indent=2)+'\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()
