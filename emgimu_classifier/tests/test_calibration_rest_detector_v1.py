import pickle
import numpy as np
import pytest
from emgimu.feature_bank.autonomous_bouts_v1 import AutonomousBoutDetectorV1
from emgimu.feature_bank.personal_temporal_bouts_v1 import PersonalTemporalBoutsV1, TemporalBoutBatchV1
from emgimu.feature_bank.calibration_rest_detector_v1 import fit_calibration_rest_detector


@pytest.mark.parametrize('channels,rate', [(4, 200.), (8, 250.)])
def test_detector_matches_desktop_neutral_concat_and_never_uses_active_calibration(channels, rate):
    rng = np.random.default_rng(1)
    classes = ('neutral', 'pinch', 'fist', 'open')
    w = PersonalTemporalBoutsV1(source_bank_id='source', sample_rate_hz=rate,
        channel_ids=tuple(f'EMG{i+1}' for i in range(channels)), preprocessing_id='filtered', class_names=classes)
    raw = tuple(rng.normal(0, .1 if i == 0 else 3., (int(rate*2), channels)) for i in range(4))
    ids = tuple('cal'+str(i) for i in range(4)); records = tuple('record'+str(i) for i in range(4))
    batch = TemporalBoutBatchV1(raw, ids, records, (0,)*4, rate, w.channel_ids, 'filtered', 'complete_cued')
    labels = dict(zip(ids, classes)); profile = w.enroll(batch, labels, user_id='u', session_id='long')
    before = pickle.dumps((w, profile))
    detector, receipt = fit_calibration_rest_detector(w, batch, labels, profile, user_id='u', rest_label='neutral')
    direct = AutonomousBoutDetectorV1().fit_rest(raw[0], rate, source_trial_ids=('cal0',))
    assert (detector.on_, detector.off_, detector.counts_) == (direct.on_, direct.off_, direct.counts_)
    assert receipt['neutral_trial_ids'] == ['cal0'] and receipt['neutral_samples'] == int(rate*2)
    changed = TemporalBoutBatchV1((raw[0],)+tuple(x*1e6 for x in raw[1:]), ids, records, (0,)*4,
        rate, w.channel_ids, 'filtered', 'complete_cued')
    other, second = fit_calibration_rest_detector(w, changed, labels, profile, user_id='u', rest_label='neutral')
    assert (other.on_, other.off_) == (detector.on_, detector.off_)
    assert receipt['calibration_input_sha256'] != second['calibration_input_sha256']
    assert pickle.dumps((w, profile)) == before and detector.next_ is None
    with pytest.raises(ValueError, match='isolation'):
        fit_calibration_rest_detector(w, batch, labels, profile, user_id='u', rest_label='neutral', forbidden_recording_ids=['record0'])
    with pytest.raises(ValueError):
        fit_calibration_rest_detector(w, batch, labels, profile, user_id='other', rest_label='neutral')


def test_reordered_or_estimated_calibration_cannot_fit_registered_detector():
    w = PersonalTemporalBoutsV1(source_bank_id='source', sample_rate_hz=200., channel_ids=('a',),
        preprocessing_id='filtered', class_names=('neutral', 'active'))
    raw = (np.ones((400, 1)), np.ones((400, 1))*3)
    b = TemporalBoutBatchV1(raw, ('a', 'b'), ('ra', 'rb'), (0, 0), 200., ('a',), 'filtered', 'complete_cued')
    labels = {'a':'neutral', 'b':'active'}; p = w.enroll(b, labels, user_id='u', session_id='long')
    estimated = TemporalBoutBatchV1(raw, b.trial_ids, b.recording_ids, b.starts, 200., ('a',), 'filtered', 'estimated')
    with pytest.raises(ValueError, match='Complete'):
        fit_calibration_rest_detector(w, estimated, labels, p, user_id='u', rest_label='neutral')
    reordered = TemporalBoutBatchV1(raw[::-1], b.trial_ids[::-1], b.recording_ids[::-1], b.starts,
        200., ('a',), 'filtered', 'complete_cued')
    with pytest.raises(ValueError, match='profile-matching'):
        fit_calibration_rest_detector(w, reordered, labels, p, user_id='u', rest_label='neutral')
