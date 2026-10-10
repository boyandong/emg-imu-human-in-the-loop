"""Exact document source composition, identity and downstream integration."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_window_composition_v1 import (
    GROUPS, CLASSES, TYPES, SCHEMA, fit_document_source, build_document_workflow,
    load_document_workflow, source_trial_features)
from emgimu.feature_bank.frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from emgimu.feature_bank.personal_session_stream_v2 import load_gate
from emgimu.feature_bank.joint_bout_workflow_v1 import JointBoutWorkflowV1
from test_extended_window_decision_v1 import calibration, kwargs
from test_joint_bout_workflow_v1 import native_fixture

ROOT = Path(__file__).resolve().parents[1]
SONG = ROOT/'benchmarks/song_real8'


@pytest.fixture(scope='module')
def source():
    gate = load_gate(SONG/'song_raw_quality_v1/source_gate.pkl', SONG/'SONG_RAW_QUALITY_V1_RESULTS.json')
    ids = np.array(gate.source_trials)
    labels = np.array([CLASSES[i % 4] for i in range(len(ids))])
    x = np.random.default_rng(750).normal(size=(len(ids), 50, 8)).astype(np.float32)
    batch = FeatureBatch(x, 250.)
    families, models, metadata = fit_document_source(batch, labels, ids)
    bank = FrozenEmgProviderBankV1(families, models, {g: 1. for g in GROUPS},
        classes=CLASSES, class_names=CLASSES, source_trial_ids=tuple(np.unique(ids)),
        source_policy_id='document_fixture', sample_rate_hz=250., window_samples=50,
        channels=8, population=np.ones(7)/7, n0=4., reliability_temperature=.5)
    return build_document_workflow(bank, gate), batch, ids, labels, metadata


def test_fitted_document_types_dimensions_and_exact_trial_probability_oracle(source):
    w, batch, ids, _, metadata = source
    assert [metadata[g]['dimension'] for g in GROUPS] == [48, 8, 72, 8, 72, 57, 16]
    for g, types in zip(GROUPS, TYPES):
        assert tuple(type(f) for f in w.bank.families_[g]) == types
    # Query uses new independent identities, never the source trial identities.
    qids = np.array(['query:'+t for t in ids])
    before = pickle.dumps(w)
    readout = w.bank.predict_providers(batch, qids, window_offsets=np.zeros(len(ids), dtype=int), user_id='fixture')
    for g in GROUPS:
        x, axis = source_trial_features(w.bank.families_[g], batch, qids)
        scaler, model = w.bank.models_[g]
        logits = scaler.transform(x) @ model.coef_.T + model.intercept_
        logits -= logits.max(1, keepdims=True)
        direct = np.exp(logits); direct /= direct.sum(1, keepdims=True)
        assert axis == readout['trial_ids']
        np.testing.assert_allclose(direct, readout['probabilities'][g], rtol=0, atol=1e-14)
    assert pickle.dumps(w) == before


@pytest.mark.parametrize('failure', ['source_overlap', 'mixed_trial_labels', 'missing_class', 'rate', 'samples', 'channels', 'empty_id'])
def test_source_fit_rejects_invalid_provenance_before_training(source, failure):
    _, b, ids, y, _ = source
    ids, y = ids.copy(), y.copy(); forbidden = ()
    if failure == 'source_overlap': forbidden = (ids[0],)
    elif failure == 'mixed_trial_labels': ids[1] = ids[0]
    elif failure == 'missing_class': y[y == CLASSES[0]] = CLASSES[1]
    elif failure == 'rate': b = FeatureBatch(b.emg, 200.)
    elif failure == 'samples': b = FeatureBatch(b.emg[:, :-1], 250.)
    elif failure == 'channels': b = FeatureBatch(b.emg[:, :, :-1], 250.)
    else: ids[0] = ''
    with pytest.raises(ValueError): fit_document_source(b, y, ids, forbidden_trials=forbidden)


def test_personal_session_persistence_and_all_provider_omissions(source, tmp_path):
    w = source[0]
    b, ids, offsets, labels = calibration(w, 'doc_long')
    p = w.enroll_user(b, ids, labels, window_offsets=offsets, **kwargs(w, 'long'))
    b, ids, offsets, labels = calibration(w, 'doc_current')
    s = w.calibrate_session(b, ids, labels, window_offsets=offsets, personal=p, **kwargs(w, 'current'))
    b, ids, offsets, _ = calibration(w, 'doc_query')
    before = pickle.dumps((w, p, s))
    common = dict(personal=p, session=s, window_offsets=offsets, **kwargs(w, 'current'))
    q = w.predict(b, ids, **common)
    assert q['document_composition_schema'] == SCHEMA
    pp, sp = tmp_path/'p.zip', tmp_path/'s.zip'
    w.save_profile(p, pp); w.save_profile(s, sp)
    rp = w.load_profile(pp, user_id='fixture')
    rs = w.load_profile(sp, user_id='fixture', session_id='current', personal=rp)
    replay = w.predict(b, ids, **dict(common, personal=rp, session=rs))
    np.testing.assert_array_equal(q['probabilities'], replay['probabilities'])
    for removed in GROUPS:
        active = tuple(g for g in GROUPS if g != removed)
        r = w.predict(b, ids, **common, available=active)
        direct = sum(weight*r['decision_provider_probabilities'][g] for weight, g in zip(r['weights'], active))
        np.testing.assert_allclose(r['probabilities'], direct, rtol=0, atol=1e-14)
    assert pickle.dumps((w, p, s)) == before


def test_document_package_reload_rejects_bytes_and_legacy_feature_identity(source, tmp_path):
    w = source[0]; package, policy = tmp_path/'source.pkl', tmp_path/'policy.json'
    payload = pickle.dumps(w.bank); package.write_bytes(payload)
    policy.write_text(json.dumps(dict(schema=SCHEMA, source_bank_sha256=hashlib.sha256(payload).hexdigest(),
        source_bank_id=w.bank.bank_id_, decision_policy_id=w.policy_id)), encoding='utf8')
    args = (package, policy, SONG/'song_raw_quality_v1/source_gate.pkl', SONG/'SONG_RAW_QUALITY_V1_RESULTS.json')
    restored = load_document_workflow(*args)
    assert restored.policy_id == w.policy_id
    package.write_bytes(payload+b'corrupted')
    with pytest.raises(ValueError, match='checksum'): load_document_workflow(*args)
    bad = copy.deepcopy(w.bank)
    bad.families_['F1'][0].document_source_trials_ = ('foreign',)
    with pytest.raises(ValueError, match='provenance'): build_document_workflow(bad, w.gate)


def test_exact_document_composition_is_usable_by_shared_complete_bout_workflow(source, tmp_path):
    w = JointBoutWorkflowV1(source[0])
    b, y, raw = native_fixture(w, 'document_long', shots=1)
    p = w.enroll(b, y, user_id='fixture', session_id='long', raw_batch=raw)
    b, y, raw = native_fixture(w, 'document_current', shots=1)
    s = w.enroll(b, y, user_id='fixture', session_id='current', personal=p, raw_batch=raw)
    query, _, raw = native_fixture(w, 'document_query', shots=1)
    before = pickle.dumps((w, p, s))
    r = w.predict(query, personal=p, session=s, user_id='fixture', session_id='current', raw_batch=raw)
    assert r['probabilities'].shape == (4, 4)
    np.testing.assert_allclose(r['probabilities'].sum(1), 1., rtol=0, atol=1e-14)
    assert p.calibration_cost['unique_native_calibration_trials'] == 4
    path = tmp_path/'joint.zip'; w.save_profile(p, path)
    assert w.load_profile(path, user_id='fixture').profile_id == p.profile_id
    assert pickle.dumps((w, p, s)) == before
