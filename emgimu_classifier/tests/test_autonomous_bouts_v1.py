import pickle
import numpy as np
import pytest
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1
from emgimu.feature_bank.temporal import CompleteSequenceBatch


def model(rate=200., **policy):
    return AutonomousBoutDetectorV1(**policy).fit_rest(
        np.zeros((int(rate), 8)), rate, source_trial_ids=['source-rest'])


@pytest.mark.parametrize('rate', [200., 250.])
def test_step_boundaries_chunk_invariance_and_owned_native_samples(rate):
    x = np.zeros((int(4*rate), 8))
    x[int(2*rate):int(3*rate)] = 1.
    whole = model(rate)
    event, = whole.feed(x, 0, trial_id='evaluation')
    expected_start = int(2*rate)-int(np.ceil(.1*rate))
    expected_end = int(3*rate)+round(.025*rate)-1+int(np.ceil(.12*rate))
    assert (event.start, event.end) == (expected_start, expected_end)
    np.testing.assert_array_equal(event.emg, x[event.start:event.end])
    assert event.duration_seconds == (expected_end-expected_start)/rate
    assert not isinstance(event, CompleteSequenceBatch)
    chunked = model(rate)
    events = []
    for i in range(0, len(x), 17):
        events += chunked.feed(x[i:i+17], i, trial_id='evaluation')
    assert len(events) == 1
    assert (events[0].start, events[0].end) == (event.start, event.end)
    np.testing.assert_array_equal(events[0].emg, event.emg)
    x[:] = 19.
    assert np.max(event.emg) == 1.
    assert not event.emg.flags.writeable
    assert whole.finish() is False


def test_source_thresholds_are_frozen_and_short_bursts_are_rejected():
    m = model()
    thresholds = pickle.dumps((m.on_, m.off_, m.source_trial_ids_, m.counts_))
    x = np.zeros((1000, 8)); x[400:405] = 1000.; x[600:620] = 1.
    assert m.feed(x, 0, trial_id='eval') == []
    assert pickle.dumps((m.on_, m.off_, m.source_trial_ids_, m.counts_)) == thresholds


def test_censored_start_end_and_overflow_do_not_emit_complete_bouts():
    m = model()
    assert m.feed(np.ones((400, 8)), 0, trial_id='eval') == []
    assert m.finish() is False  # Never armed: stream starts active.
    x = np.zeros((600, 8)); x[400:] = 1.
    assert m.feed(x, 0, trial_id='eval') == []
    assert m.finish() is True
    m = model(max_duration_s=2.)
    x = np.zeros((1600, 8)); x[400:1400] = 1.
    assert m.feed(x, 0, trial_id='eval') == []
    assert m.finish() is False


def test_gaps_overlap_bad_samples_and_trial_reuse_invalidate_state():
    m = model()
    with pytest.raises(ValueError, match='disjoint'):
        m.feed(np.zeros((200, 8)), 0, trial_id='source-rest')
    m.feed(np.zeros((300, 8)), 0, trial_id='eval')
    with pytest.raises(ValueError, match='gap'):
        m.feed(np.ones((300, 8)), 301, trial_id='eval')
    assert m.feed(np.ones((300, 8)), 301, trial_id='eval') == []
    with pytest.raises(ValueError, match='finite'):
        m.feed(np.full((10, 8), np.nan), 601, trial_id='eval')
    assert m.next_ is None
