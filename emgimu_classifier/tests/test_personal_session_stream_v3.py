"""Decision-stream parity, raw gating and persistent guided lifecycle."""
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import sosfilt
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_decision_cli_v1 import load_decision_workflow
from emgimu.feature_bank.personal_session_stream_v3 import PersonalSessionStreamV3
from emgimu.feature_bank.personal_session_stream_v1 import PersonalSessionStreamV1

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/song_real8'


def workflow():
    return load_decision_workflow(HERE/'song_personal_session_v1/source_bank.pkl',
        ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
        HERE/'song_integrated_decision_v1/decision_policy.json',
        HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json',
        HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')


def cal(w, prefix, session, personal=None):
    rng = np.random.default_rng(130+len(prefix))
    ids = np.array([f'{prefix}:{c}' for c in w.bank.classes_ for _ in range(2)])
    x = rng.normal(size=(len(ids), 50, 8)).astype(np.float32)
    for i, c in enumerate(w.bank.classes_):
        x[ids == f'{prefix}:{c}'] *= (i+1)*(1+np.arange(8)*.2)
    kwargs = dict(window_offsets=np.tile([0, 1], 4), user_id='fixture',
        session_id=session, observed_channel_ids=w.channels, preprocessing_id=w.preprocessing_id)
    labels = {f'{prefix}:{c}': c for c in w.bank.classes_}
    if personal is None:
        return w.enroll_user(FeatureBatch(x, 250.), ids, labels, **kwargs)
    return w.calibrate_session(FeatureBatch(x, 250.), ids, labels, personal=personal, **kwargs)


def service(w):
    return PersonalSessionStreamV3(w, user_id='fixture', session_id='current', channel_ids=w.channels)


def test_irregular_stream_matches_decision_and_old_baseline(tmp_path):
    w = workflow()
    personal = cal(w, 'long', 'long')
    current = cal(w, 'current', 'current', personal)
    pp, sp = tmp_path/'p.zip', tmp_path/'s.zip'
    w.save_profile(personal, pp)
    w.save_profile(current, sp)
    raw = np.random.default_rng(45).integers(-1000, 1000, size=(357, 8))
    for anchor, routing in ((False, False), (False, True), (True, True)):
        s = service(w)
        s.command(dict(op='profiles', personal=str(pp), session=str(sp)))
        assert s.info()['personal_trials'] == s.info()['session_trials'] == 4
        s.command(dict(op='decision', use_anchor=anchor, use_session_routing=routing))
        before = pickle.dumps((w, personal, current))
        filtered = raw.astype(float)
        for sos in s.filters:
            filtered = sosfilt(sos, filtered, axis=0)
        filtered = filtered.astype(np.float32)
        s.command(dict(op='recognize'))
        actual, ends, confirmed = [], [], []
        start = 0
        for size in (1, 7, 41, 2, 59, 3, 91, 153):
            stop = min(start+size, len(raw))
            r = s.ingest(raw[start:stop], np.arange(start, stop))
            if 'probabilities' in r:
                actual.extend(r['probabilities'])
                ends.extend(r['output_sample_indices'])
                confirmed.extend(r['confirmed_labels'])
            start = stop
        assert start == len(raw) and ends == list(range(49, len(raw), 10))
        ids = np.array([f'oracle:{i:06}' for i in ends])
        windows = FeatureBatch(np.stack([filtered[i-49:i+1] for i in ends]), 250.)
        expected = w.predict(windows, ids, window_offsets=np.zeros(len(ids), int),
            personal=personal, session=current, use_anchor=anchor, use_session_routing=routing,
            user_id='fixture', session_id='current', observed_channel_ids=w.channels,
            preprocessing_id=w.preprocessing_id)
        np.testing.assert_allclose(actual, expected['probabilities'], rtol=0, atol=1e-12)
        assert r['anchor_enabled'] == anchor and r['session_routing_enabled'] == routing
        label = active = None
        count = 0
        reference = []
        for p in expected['probabilities']:
            next_label = w.bank.class_names_[int(p.argmax())]
            count = count+1 if next_label == label else 1
            label = next_label
            if count >= 2:
                active = label
            reference.append(active)
        assert confirmed == reference
        if not anchor and not routing:
            old = PersonalSessionStreamV1(w.base, user_id='fixture', session_id='current', channel_ids=w.channels)
            old.personal, old.session = personal.base, current.base
            old.command(dict(op='recognize'))
            result = old.ingest(raw, np.arange(len(raw)))
            np.testing.assert_allclose(actual, result['probabilities'], rtol=0, atol=1e-12)
        assert pickle.dumps((w, personal, current)) == before


def test_raw_replay_uses_mixed_providers_and_resets_stale_state(tmp_path):
    w = workflow()
    p = cal(w, 'long', 'long')
    s = service(w)
    pp = tmp_path/'p.zip'
    w.save_profile(p, pp)
    s.command(dict(op='profiles', personal=str(pp)))
    s.command(dict(op='decision', use_anchor=True, use_session_routing=True))
    s.command(dict(op='quality', mode='structural'))
    raw = np.random.default_rng(43).integers(-1000, 1000, size=(4, 50, 8))
    raw[2:, :, 2] = 0
    filtered = raw.astype(np.float32)
    ids, offsets = np.array(['held:3', 'held:1', 'held:4', 'held:2']), np.zeros(4, int)
    fp, rp = tmp_path/'f.npz', tmp_path/'r.npz'
    for path, values in ((fp, filtered), (rp, raw)):
        np.savez(path, emg=values, sample_rate_hz=np.array(250.), trial_ids=ids, window_offsets=offsets)
    result = s.command(dict(op='replay', path=str(fp), raw_path=str(rp)))['prediction']
    expected = w.predict(FeatureBatch(filtered, 250.), ids, window_offsets=offsets,
        personal=p, user_id='fixture', session_id='current', observed_channel_ids=w.channels,
        preprocessing_id=w.preprocessing_id, quality_mode='structural', raw_batch=FeatureBatch(raw, 250.))
    np.testing.assert_allclose(result['probabilities'], expected['probabilities'], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result['quality_decision']['rejected'], expected['quality_decision']['rejected'])
    assert result['anchor_enabled'] and not result['session_routing_enabled']
    s.command(dict(op='recognize'))
    r = s.ingest(raw.reshape(-1, 8), np.arange(200))
    assert r['quality_rejected'][-1] and r['confirmed_labels'][-1] is None
    state = (s.personal.profile_id, s.workflow.use_anchor)
    r = s.ingest(raw[0], np.arange(201, 251))
    assert r['reset_reason'] and s.mode == 'idle' and s.quality_active is None
    assert (s.personal.profile_id, s.workflow.use_anchor) == state
    s.command(dict(op='decision', use_anchor=False, use_session_routing=True))
    assert s.info()['mode'] == 'idle' and not s.info()['anchor_enabled']


def test_guided_profiles_roundtrip_contracts_and_no_overwrite(tmp_path):
    w = workflow()
    s = service(w)
    s.command(dict(op='begin', kind='personal', shots=1, settle_samples=0, hold_samples=50))
    with pytest.raises(ValueError, match='Cancel/save'):
        s.command(dict(op='decision', use_anchor=True, use_session_routing=True))
    for i in range(4):
        s.command(dict(op='trial'))
        s.ingest(np.random.default_rng(i).integers(-1000, 1000, size=(50, 8)), np.arange(i*50, (i+1)*50))
    pp = tmp_path/'guided.zip'
    saved = s.command(dict(op='save', path=str(pp)))
    assert saved['personal_trials'] == 4 and saved['contract_id'] == w.policy_id
    audit = json.loads(Path(saved['calibration_audit_path']).read_text(encoding='utf8'))
    assert audit['contract_id'] == w.policy_id and Path(saved['calibration_raw_windows_path']).is_file()
    new = PersonalSessionStreamV3(w, user_id='fixture', session_id='next', channel_ids=w.channels)
    new.command(dict(op='profiles', personal=str(pp)))
    new.command(dict(op='begin', kind='session', shots=1, settle_samples=0, hold_samples=50))
    for i in range(4):
        new.command(dict(op='trial'))
        new.ingest(np.random.default_rng(i+9).integers(-1000, 1000, size=(50, 8)), np.arange(i*50, (i+1)*50))
    sp = tmp_path/'session.zip'
    result = new.command(dict(op='save', path=str(sp)))
    identity = result['session_profile_id']
    new.command(dict(op='profiles', personal=str(pp), session=str(sp)))
    assert new.info()['session_profile_id'] == identity
    with pytest.raises(ValueError, match='identity'):
        new.command(dict(op='profiles', personal=str(pp), session=str(pp)))
    assert new.info()['session_profile_id'] == identity
    with pytest.raises(FileExistsError):
        w.save_profile(new.personal.decision, pp)
    with pytest.raises(ValueError, match='Boolean'):
        new.command(dict(op='decision', use_anchor='yes', use_session_routing=True))
    with pytest.raises(ValueError, match='Evaluation labels'):
        new.command(dict(op='info', evaluation_labels=['fist']))


def test_native_verifier_independent_oracle_on_synthetic_windows():
    from benchmarks.song_real8.verify_decision_gui_v1 import oracle
    w = workflow()
    p = cal(w, 'long', 'long')
    s = cal(w, 'current', 'current', p)
    raw = np.random.default_rng(92).normal(size=(9, 50, 8))
    raw[5:, :, 2] = 0
    b = FeatureBatch(raw.astype(np.float32), 250.)
    ids = np.array([f'held:{i:04}' for i in range(len(raw))])
    for personal, session, anchor, routing, quality in (
            (None, None, False, False, 'off'), (p, None, True, False, 'off'),
            (p, s, False, True, 'off'), (p, s, True, True, 'structural'),
            (p, s, True, True, 'soft')):
        arm = dict(anchor=anchor, routing=routing, anchor_mode='blended', quality=quality)
        q, rejected, weights = oracle(w, personal, session, arm, b, FeatureBatch(raw, 250.), ids)
        actual = w.predict(b, ids, window_offsets=np.zeros(len(ids), int), personal=personal, session=session,
            use_anchor=anchor, use_session_routing=routing, quality_mode=quality, raw_batch=FeatureBatch(raw, 250.),
            user_id='fixture', session_id='current', observed_channel_ids=w.channels, preprocessing_id=w.preprocessing_id)
        np.testing.assert_allclose(q, actual['probabilities'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(weights, actual['weights'], rtol=0, atol=1e-12)
        if quality != 'off':
            np.testing.assert_array_equal(rejected, actual['quality_decision']['rejected'])
