import pickle
import numpy as np
import pytest
from emgimu.feature_bank.autonomous_bouts_v1 import DetectedBout
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


class KnownClassifier:
    classes_ = np.arange(4)

    def predict_proba(self, x):
        out = np.column_stack((x[:, 0], np.ones((len(x), 3))))
        return out/out.sum(axis=1, keepdims=True)


def test_all_complete_windows_tail_reporting_and_source_temperature_are_exact():
    family = MeanFamily(); scaler = IdentityScaler(); model = KnownClassifier()
    reader = DetectedG5ReaderV1(family, scaler, model, source_recording_ids=['source'], temperature=2.)
    before = pickle.dumps((family, scaler, model))
    # Ten complete windows carrying values1..10 plus eleven tail samples.
    # The tail must be reported, not padded into an eleventh training window.
    x = np.repeat(np.arange(1., 11.), 40)[:, None]*np.ones((1, 4))
    x = np.vstack((x, np.full((11, 4), 1000.)))
    result = reader.read(DetectedBout(100, 511, x, 200.), recording_id='evaluation')
    expected = np.sqrt(np.array([5.5, 1., 1., 1.])); expected /= expected.sum()
    np.testing.assert_allclose(result['probability'], expected, atol=1e-12)
    assert result['complete_windows'] == 10 and result['unrepresented_tail_samples'] == 11
    assert result['prediction'] == 0 and result['boundary_kind'] == 'estimated'
    assert not result['certified_full_coverage']
    assert pickle.dumps((family, scaler, model)) == before
    with pytest.raises(ValueError, match='overlaps'):
        reader.read(DetectedBout(100, 511, x, 200.), recording_id='source')
    with pytest.raises(ValueError, match='contract'):
        reader.read(DetectedBout(100, 511, x, 250.), recording_id='evaluation')
