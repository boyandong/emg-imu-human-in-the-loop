"""Bind a neutral-only detector to an immutable full-action calibration profile."""
import hashlib
import pickle
import numpy as np
from .autonomous_bouts_v1 import AutonomousBoutDetectorV1
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1


def fit_calibration_rest_detector(workflow, batch, labels, profile, *, user_id, rest_label,
                                 forbidden_trial_ids=(), forbidden_recording_ids=()):
    """Fit fixed detector thresholds from neutral calibration only, never queries.

    All calibration records must match the saved temporal profile. The neutral
    sequences are concatenated exactly as in the frozen V5 desktop registration;
    its cross-sequence 25ms smoothing context is retained, not silently changed.
    The returned detector is reset and can be copied for independent streams.
    """
    if not isinstance(workflow, PersonalTemporalBoutsV1):
        raise ValueError('Explicit temporal workflow required')
    workflow._input(batch)
    workflow._profile(profile, user_id)
    if (batch.boundary_kind != 'complete_cued' or batch.trial_ids != profile.trial_ids
            or batch.recording_ids != profile.recording_ids
            or set(batch.trial_ids) & set(forbidden_trial_ids)
            or set(batch.recording_ids) & set(forbidden_recording_ids)
            or rest_label not in workflow.class_names or not isinstance(labels, dict)
            or set(labels) != set(batch.trial_ids) or set(labels.values()) != set(workflow.class_names)):
        raise ValueError('Complete profile-matching calibration, separate labels and evaluation isolation required')
    positions = [i for i, trial in enumerate(batch.trial_ids) if labels[trial] == rest_label]
    rest = np.concatenate([batch.sequences[i] for i in positions])
    trial_ids = tuple(batch.trial_ids[i] for i in positions)
    detector = AutonomousBoutDetectorV1().fit_rest(rest, batch.sample_rate_hz, source_trial_ids=trial_ids)
    identity = hashlib.sha256(pickle.dumps((batch, labels), protocol=4)).hexdigest()
    receipt = dict(schema='calibration_rest_detector_v1', temporal_contract_id=workflow.contract_id,
        temporal_profile_id=profile.profile_id, user_id=user_id, session_id=profile.session_id, kind=profile.kind,
        calibration_input_sha256=identity, neutral_trial_ids=list(trial_ids),
        neutral_recording_ids=[batch.recording_ids[i] for i in positions],
        neutral_samples=len(rest), sample_rate_hz=batch.sample_rate_hz, channel_ids=list(batch.channel_ids),
        preprocessing_id=batch.preprocessing_id, off=detector.off_, on=detector.on_,
        smoothing_samples=detector.width_, confirmation_samples=list(detector.counts_),
        neutral_only=True, current_queries_used=False, fixed_default_policy=list(detector.policy),
        concatenated_neutral_sequences_match_desktop=True, physical_validation_proven=False)
    return detector, receipt
