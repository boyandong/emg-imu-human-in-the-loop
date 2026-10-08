"""Opt-in explicit physical-channel mapping throughout personal/session flow."""
import numpy as np
from .electrode_layout_v1 import RingElectrodeLayoutV1
from .force_nested_oof import SubjectWindows
from .session_pipeline import DocumentSessionCalibrationPipelineV2


class MappedSessionCalibrationPipelineV3:
    """Operator-attested layout, not software-authenticated physical geometry.

    Every call names its observed columns. Source layout identity is bound to
    session state; all source/calibration/evaluation leakage guards are retained.
    This does not change any live or public benchmark deployment default.
    """

    def __init__(self, *, rest_label, layout):
        if not isinstance(layout, RingElectrodeLayoutV1):
            raise ValueError('Explicit RingElectrodeLayoutV1 required')
        self.layout = layout
        self.pipeline = DocumentSessionCalibrationPipelineV2(rest_label=rest_label, ring_topology=True)

    @staticmethod
    def _validate_trials(data):
        ids = np.asarray(data.trials, dtype=object)
        if ids.shape != (data.batch.windows,) or any(not isinstance(t, str) or not t.strip() for t in ids):
            raise ValueError('Aligned explicit native trial identities required')

    def _mapped_data(self, data, observed_channel_ids, *, source=False):
        self._validate_trials(data)
        if source:
            batch = self.layout.apply_source(data.batch, observed_channel_ids=observed_channel_ids)
        else:
            batch = self.layout.apply_evaluation(data.batch, observed_channel_ids=observed_channel_ids,
                                                  model_layout_id=self.source_layout_id_)
        return SubjectWindows(batch, data.labels, data.subjects, data.trials)

    def fit_long_term(self, source, *, observed_channel_ids):
        mapped = self._mapped_data(source, observed_channel_ids, source=True)
        self.pipeline.fit_long_term(mapped)
        self.source_layout_id_ = self.layout.layout_id
        return self

    def _check(self, session=None):
        if not hasattr(self, 'source_layout_id_'):
            raise RuntimeError('Mapped long-term source must be fit first')
        if self.layout.layout_id != self.source_layout_id_:
            raise ValueError('Physical layout differs from fitted source')
        if session is not None and (not isinstance(session, dict)
                                    or session.get('physical_layout_id') != self.source_layout_id_):
            raise ValueError('Session physical layout differs from fitted source')

    def calibrate_session(self, calibration, *, observed_channel_ids):
        self._check()
        mapped = self._mapped_data(calibration, observed_channel_ids)
        state = self.pipeline.calibrate_session(mapped)
        state['physical_layout_id'] = self.source_layout_id_
        return state

    def predict(self, data, session=None, *, observed_channel_ids):
        self._check(session)
        return self.pipeline.predict(self._mapped_data(data, observed_channel_ids), session)

    def predict_unlabeled(self, batch, subjects, trial_ids, session=None, *, observed_channel_ids):
        self._check(session)
        mapped = self.layout.apply_evaluation(batch, observed_channel_ids=observed_channel_ids,
                                              model_layout_id=self.source_layout_id_)
        return self.pipeline.predict_unlabeled(mapped, subjects, trial_ids, session)
