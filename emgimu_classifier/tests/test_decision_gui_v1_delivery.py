"""Hash-bound full-stream outputs; numerical integration stays distinct from efficacy."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_operational_gui_native_emissions_protocol_and_no_promotion():
    protocol_path = ROOT/'benchmarks/song_real8/SONG_DECISION_GUI_V1_PROTOCOL.json'
    evidence_path = ROOT/'feature_bank/SONG_DECISION_GUI_V1_ACCEPTANCE.json'
    p = json.loads(protocol_path.read_text(encoding='utf8'))
    e = json.loads(evidence_path.read_text(encoding='utf8'))
    assert e['protocol_sha256'] == sha(protocol_path)
    assert e['source_sha256'] == p['source_sha256']
    assert e['artifact_sha256'] == p['artifact_sha256']
    for name, digest in e['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    for name, digest in e['artifact_sha256'].items():
        assert sha(ROOT/name) == digest, name
    emissions = ROOT/e['emissions_path']
    assert sha(emissions) == e['emissions_sha256']
    assert len(e['records']) == len(p['arms']) == 10
    assert e['verified_windows'] == 297900
    assert e['independent_filter_and_geometry_probabilities']
    assert e['chronological_confirmation_exact'] and e['source_and_profiles_immutable']
    assert not e['default_promoted'] and not e['physical_validation_proven'] and not e['completion_proven']
    assert 'calibration intervals' in e['scope']
    with np.load(emissions, allow_pickle=False) as saved:
        ends = saved['output_sample_indices']
        assert len(ends) == 29790 and ends[0] == 49 and np.all(np.diff(ends) == 10)
        for row, arm in zip(e['records'], p['arms']):
            name = row['arm']
            assert name == arm['name'] and row['windows'] == len(ends)
            assert row['maximum_probability_error'] <= p['tolerance']
            assert row['confirmations_exact'] and row['source_and_profiles_immutable']
            q, rejected, labels = (saved[name+suffix] for suffix in ('_probabilities', '_rejected', '_confirmed'))
            assert q.shape == (len(ends), 4) and np.isfinite(q).all()
            np.testing.assert_allclose(q.sum(1), 1, rtol=0, atol=1e-12)
            assert row['rejected'] == int(rejected.sum())
            candidate = active = -1
            count = 0
            expected = []
            for prob, reject in zip(q, rejected):
                if reject:
                    candidate = active = -1
                    count = 0
                else:
                    next_label = int(prob.argmax())
                    count = count+1 if next_label == candidate else 1
                    candidate = next_label
                    if count >= 2:
                        active = candidate
                expected.append(active)
            np.testing.assert_array_equal(labels, expected)
            if arm.get('dropout'):
                assert rejected.all() and np.all(labels == -1)
        np.testing.assert_array_equal(saved['full_off_probabilities'], saved['full_structural_probabilities'])
    index = json.loads((ROOT/'feature_bank/delivery/INDEX.json').read_text(encoding='utf8'))
    assert index['song_decision_gui_acceptance']['sha256'] == sha(evidence_path)
