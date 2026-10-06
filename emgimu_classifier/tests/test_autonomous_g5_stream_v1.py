import pickle
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1
from emgimu.feature_bank.autonomous_g5_stream_v1 import AutonomousG5StreamV1
from emgimu.feature_bank.detected_g5_reader_v1 import DetectedG5ReaderV1
from emgimu.feature_bank.validated_unibo import ValidatedUniBoFamily


class MeanFamily(ValidatedUniBoFamily):
    def __init__(self):
        super().__init__('G5'); self.fitted_ = True

    def transform(self, batch):
        self._validate(batch)
        return batch.emg.mean(axis=1)[:, :1]


class IdentityScaler:
    def transform(self, x):
        return x


class Classifier:
    classes_ = np.arange(4)

    def predict_proba(self, x):
        p = np.column_stack((np.ones(len(x)), 1+x[:,0], np.ones((len(x),2))))
        return p/p.sum(axis=1, keepdims=True)


def setup():
    detector = AutonomousBoutDetectorV1().fit_rest(np.zeros((200,4)), 200., source_trial_ids=['rest-source'])
    reader = DetectedG5ReaderV1(MeanFamily(), IdentityScaler(), Classifier(), source_recording_ids=['model-source'])
    return detector, reader


def test_chunk_to_prediction_invariant_causal_delivery_and_source_immutability():
    detector, reader = setup(); frozen = pickle.dumps((detector, reader))
    x = np.zeros((1000,4)); x[400:600] = 1.
    for chunk in (1, 17, 1000):
        stream = AutonomousG5StreamV1(detector, reader); outputs = []
        for i in range(0, len(x), chunk):
            outputs += stream.feed(x[i:i+chunk], i, recording_id='eval')
        result, = outputs
        assert (result['start'], result['end']) == (380,628)
        assert result['complete_windows'] == 6 and result['unrepresented_tail_samples'] == 8
        assert not result['certified_full_coverage']
        # Source aggregation covers200 active samples among240 used samples.
        raw = np.array([1., 1+200/240, 1., 1.]); raw /= raw.sum()
        np.testing.assert_allclose(result['probability'], raw, atol=1e-7)
        assert result['emitted_at_sample'] >= result['end']-1
        assert 0 <= result['delivery_delay_samples'] < chunk
        assert stream.finish() is False
    assert pickle.dumps((detector, reader)) == frozen


def test_source_overlap_gap_and_end_censoring_never_emit_complete_predictions():
    detector, reader = setup(); stream = AutonomousG5StreamV1(detector, reader)
    for name in ('rest-source', 'model-source'):
        with pytest.raises(ValueError, match='disjoint'):
            stream.feed(np.zeros((50,4)), 0, recording_id=name)
    assert stream.feed(np.zeros((100,4)), 0, recording_id='eval') == []
    assert stream.feed(np.ones((250,4)), 100, recording_id='eval') == []
    assert stream.finish() is True
    assert stream.feed(np.zeros((50,4)), 0, recording_id='eval2') == []
    with pytest.raises(ValueError, match='gap'):
        stream.feed(np.zeros((50,4)), 51, recording_id='eval2')
    assert stream.detector.next_ is None


def test_native_chunk_to_prediction_parity_provenance_and_full_recording_coverage():
    root = Path(__file__).resolve().parents[1]
    here = root/'benchmarks/new_bank_v3'
    r = json.loads((here/'AUTONOMOUS_G5_STREAM_V1_RESULTS.json').read_text())
    for name, digest in r['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    boundary = json.loads((here/'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json').read_text())
    expected = {e['member']: e for e in boundary['recordings']}
    assert len(r['recordings']) == len(expected) == 28
    assert {e['member'] for e in r['recordings']} == set(expected)
    assert sum(e['events'] for e in r['recordings']) == 1257
    for e in r['recordings']:
        old = expected[e['member']]
        assert e['events'] == len(old['detected_bounds'])
        assert e['censored_end'] == old['censored_end']
        assert e['max_probability_error'] <= 1e-12
        assert 0 <= e['max_chunk_delay_samples'] < r['chunk_samples']
    assert r['source_state_immutable'] and not r['classifier_refitted']
    assert not r['completion_proven'] and not r['default_promoted']
