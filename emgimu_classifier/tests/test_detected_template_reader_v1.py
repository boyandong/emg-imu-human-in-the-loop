import pickle
import numpy as np
import pytest
from emgimu.feature_bank.autonomous_bouts_v1 import DetectedBout
from emgimu.feature_bank.detected_template_reader_v1 import DetectedTemplateReaderV1
from emgimu.feature_bank.document_path_v3 import DocumentTemporalTemplatesV3
from emgimu.feature_bank.temporal import CompleteSequenceBatch


def test_detected_bout_uses_full_rms_path_without_certifying_or_mutating_templates():
    axes = np.eye(8)[:2]
    source = CompleteSequenceBatch(np.repeat(axes[:, None], 32, axis=1), 1.,
                                   durations_seconds=np.ones(2), full_coverage=True)
    template = DocumentTemporalTemplatesV3().fit(source, [1, 2], trial_ids=['cal-a', 'cal-b'])
    reader = DetectedTemplateReaderV1(template, native_sample_rate_hz=250., source_recording_ids=['source-file'])
    frozen = pickle.dumps(template)
    x = np.repeat(axes[1][None], 500, axis=0)
    bout = DetectedBout(100, 600, x, 250.)
    result = reader.read(bout, recording_id='evaluation-file')
    np.testing.assert_allclose(result['distances'], [np.sqrt(2), 0.], atol=1e-8)
    assert result['nearest_class'] == 2
    assert result['boundary_kind'] == 'estimated'
    assert not result['certified_full_coverage']
    assert not result['calibrated_probability_available']
    assert pickle.dumps(template) == frozen
    for recording_id in ['source-file', 'cal-a']:
        with pytest.raises(ValueError, match='overlaps'):
            reader.read(bout, recording_id=recording_id)
    for bad in [DetectedBout(100, 600, x, 200.), DetectedBout(100, 200, x[:100], 250.)]:
        with pytest.raises(ValueError, match='contract'):
            reader.read(bad, recording_id='evaluation-file')


def test_full_path_includes_tail_instead_of_sparse_or_first_window_crops():
    axes = np.eye(8)[:2]
    source = CompleteSequenceBatch(np.repeat(axes[:, None], 32, axis=1), 1.,
                                   durations_seconds=np.ones(2), full_coverage=True)
    template = DocumentTemporalTemplatesV3().fit(source, [1, 2], trial_ids=['cal-a', 'cal-b'])
    reader = DetectedTemplateReaderV1(template, native_sample_rate_hz=200., source_recording_ids=['source'])
    x = np.repeat(axes[0][None], 384, axis=0); x[192:] = axes[1]
    result = reader.read(DetectedBout(0, 384, x, 200.), recording_id='evaluation')
    np.testing.assert_allclose(result['distances'][0], np.sqrt(2)/2, atol=1e-8)
    assert result['distances'][0] > .5
