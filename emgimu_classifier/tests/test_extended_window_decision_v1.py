"""Source-only CSP, direct spectral oracle, trial-mass profiles and gate routing."""
import pickle
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import eigvalsh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v2 import load_gate
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1
from emgimu.feature_bank.extended_window_decision_v1 import SourceCsp250V1, CspQualityGateV1, ExtendedWindowDecisionV1

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/song_real8'


@pytest.fixture
def workflow():
    old = load_workflow(HERE/'song_personal_session_v1/source_bank.pkl', ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    ids = np.array(old.bank.policy_.source_trials)
    labels = np.array([old.bank.classes_[i % 4] for i in range(len(ids))])
    raw = np.random.default_rng(74).normal(size=(len(ids), 50, 8)).astype(np.float32)
    b = FeatureBatch(raw, 250.)
    csp = SourceCsp250V1().fit(b, labels, source_trial_ids=ids)
    x = csp.transform(b)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(x), labels)
    families, models, temperatures = dict(old.bank.families_), dict(old.bank.models_), dict(old.bank.temperatures_)
    families['F2b'], models['F2b'], temperatures['F2b'] = [csp], (scaler, model), 1.
    bank = FrozenEmgProviderBankV1(families, models, temperatures, classes=old.bank.classes_, class_names=old.bank.class_names_,
        source_trial_ids=ids, source_policy_id='fixture', sample_rate_hz=250., window_samples=50, channels=8,
        population=np.ones(7)/7, n0=4., reliability_temperature=.5)
    base = PersonalSessionWorkflowV1(bank, channel_ids=old.channels, preprocessing_id=old.preprocessing_id,
        rest_label=old.rest_label, quality_options=old.quality_options)
    gate = CspQualityGateV1(load_gate(HERE/'song_raw_quality_v1/source_gate.pkl', HERE/'SONG_RAW_QUALITY_V1_RESULTS.json'))
    return ExtendedWindowDecisionV1(base, gate)


def calibration(w, prefix):
    ids, labels, offsets, values = [], {}, [], []
    rng = np.random.default_rng(len(prefix)+55)
    for c in w.bank.classes_:
        for trial in range(2):
            name = f'{prefix}:{c}:{trial}'
            labels[name] = c
            for offset in range(1+trial*2):
                ids.append(name)
                offsets.append(offset)
                values.append(rng.normal(size=(50, 8))*(1+trial*2))
    return FeatureBatch(np.asarray(values, np.float32), 250.), np.array(ids), np.array(offsets), labels


def kwargs(w, session):
    return dict(user_id='fixture', session_id=session, observed_channel_ids=w.channels, preprocessing_id=w.preprocessing_id)


def test_csp_equations_and_source_sensor_provenance():
    x = np.random.default_rng(91).normal(size=(28, 50, 8))+np.arange(8)[None, None, :]
    y = np.arange(28) % 4
    b = FeatureBatch(x, 250.)
    f = SourceCsp250V1().fit(b, y, source_trial_ids=[f't{i//2}' for i in range(len(y))])
    covariance = np.stack([v.T@v/(np.trace(v.T@v)+1e-10) for v in x])
    for i, c in enumerate(f.classes_):
        positive = covariance[y == c].mean(0)
        total = positive+covariance[y != c].mean(0)+1e-5*np.eye(8)
        vectors, values = f.filters_[:, i*4:i*4+4], f.eigenvalues_[i]
        np.testing.assert_allclose(positive@vectors, (total@vectors)*values, rtol=1e-10, atol=1e-10)
        eigen = eigvalsh(positive, total)
        np.testing.assert_allclose(values, np.r_[eigen[-2:], eigen[:2]], rtol=1e-10, atol=1e-10)
    variance = np.stack([np.var(v@f.filters_, axis=0) for v in x])
    expected = np.log(variance/(variance.sum(1, keepdims=True)+1e-10)+1e-10)
    np.testing.assert_allclose(f.transform(b), expected, rtol=1e-6, atol=2e-7)
    before = pickle.dumps(f)
    with pytest.raises(ValueError, match='contract'):
        f.transform(FeatureBatch(x[:, :40], 200.))
    f.transform(FeatureBatch(x*3, 250.))
    assert pickle.dumps(f) == before
    with pytest.raises(ValueError, match='provenance'):
        SourceCsp250V1().fit(b, y, source_trial_ids=['']*len(y))


def test_context_formula_trial_balancing_and_profile_roundtrip(workflow, tmp_path):
    w = workflow
    b, ids, offsets, labels = calibration(w, 'long')
    p = w.enroll_user(b, ids, labels, window_offsets=offsets, **kwargs(w, 'long'))
    frequency = np.fft.rfftfreq(50, 1/250.)
    harmonics = np.exp(-2j*np.pi*np.outer(np.arange(26), np.arange(50))/50)
    tapered = (b.emg.astype(float)-b.emg.astype(float).mean(1, keepdims=True))*np.hanning(50)[None, :, None]
    power = np.abs(np.einsum('kt,ntc->nkc', harmonics, tapered))**2/50
    columns = []
    for i, (low, high) in enumerate(w.bands):
        mask = (frequency >= low) & (frequency <= high if i == len(w.bands)-1 else frequency < high)
        columns.append(np.log(power[:, mask].sum(1)+1e-10))
    reference = np.concatenate(columns, 1).astype(np.float32)
    np.testing.assert_allclose(w.log_bands(b), reference, rtol=0, atol=2e-6)
    equal = np.stack([reference[ids == t].astype(float).mean(0) for t in np.unique(ids)]).mean(0)
    np.testing.assert_allclose(p.spectral_reference, equal, rtol=0, atol=2e-6)
    assert not np.allclose(equal, reference.mean(0))
    cb, ci, co, cy = calibration(w, 'current')
    s = w.calibrate_session(cb, ci, cy, window_offsets=co, personal=p, **kwargs(w, 'current'))
    eb, ei, eo, _ = calibration(w, 'held')
    before = pickle.dumps((w, p, s))
    actual = w.predict(eb, ei, window_offsets=eo, personal=p, session=s, **kwargs(w, 'current'))
    old = w.decision.predict(eb, ei, window_offsets=eo, personal=p.decision, session=s.decision, **kwargs(w, 'current'))
    np.testing.assert_array_equal(actual['probabilities'], old['probabilities'])
    context = actual['spectral_context']
    np.testing.assert_allclose(context['window_minus_long'], w.log_bands(eb)-p.spectral_reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(context['session_minus_long'], s.spectral_reference-p.spectral_reference)
    assert 'F4d' not in w.bank.providers_ and context['window_minus_long'].shape == (len(ei), 32)
    pp, sp = tmp_path/'personal.zip', tmp_path/'session.zip'
    w.save_profile(p, pp)
    w.save_profile(s, sp)
    restored = w.load_profile(pp, user_id='fixture')
    current = w.load_profile(sp, user_id='fixture', session_id='current', personal=restored)
    replay = w.predict(eb, ei, window_offsets=eo, personal=restored, session=current, **kwargs(w, 'current'))
    np.testing.assert_array_equal(actual['probabilities'], replay['probabilities'])
    with pytest.raises(FileExistsError):
        w.save_profile(p, pp)
    with pytest.raises(ValueError, match='identity'):
        w.load_profile(pp, user_id='another')
    assert pickle.dumps((w, p, s)) == before


def test_csp_soft_mask_uses_channel_min_and_source_contract_is_immutable(workflow):
    w = workflow
    gate = w.gate
    x = np.random.default_rng(22).integers(-1000, 1000, size=(3, 50, 8))
    x[1, :, 2] = 0
    ids = np.array(['held0', 'held1', 'held2'])
    observation = gate.observe(FeatureBatch(x, 250.), ids, observed_channel_ids=w.channels)
    q = np.tile([.7, .1, .1, .1], (3, 1))
    names = w.bank.providers_
    result = gate.decide({n: q for n in names}, np.ones(7)/7, FeatureBatch(x, 250.), ids,
        observed_channel_ids=w.channels, class_names=w.bank.class_names_, mode='soft',
        provider_trial_ids={n: tuple(ids) for n in names})
    channels = observation['trial_soft_channel_quality']
    masks = np.column_stack([channels.min(1) if n in ('F0', 'F2ac', 'F2b') else channels.mean(1) for n in names])
    weights = masks/7
    weights[weights.sum(1) <= 1e-10] = np.ones(7)/7
    weights /= weights.sum(1, keepdims=True)
    np.testing.assert_array_equal(result['provider_weights'], weights)
    assert result['provider_weights'][1, names.index('F2b')] == 0
    with pytest.raises(ValueError, match='trial axis'):
        gate.decide({n: q for n in names}, np.ones(7)/7, FeatureBatch(x, 250.), ids,
            observed_channel_ids=w.channels, class_names=w.bank.class_names_, mode='soft',
            provider_trial_ids={n: tuple(ids[::-1]) for n in names})
