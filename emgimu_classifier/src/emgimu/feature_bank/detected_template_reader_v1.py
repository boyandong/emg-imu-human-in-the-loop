"""Explicit estimated-boundary readout for source-fitted document DTW templates."""
import numpy as np
from .autonomous_bouts_v1 import DetectedBout
from .document_path_v3 import DocumentTemporalTemplatesV3, document_envelope_path, document_dtw_distance


class DetectedTemplateReaderV1:
    """Use an entire detected native bout; never fabricate certified coverage.

    The nearest template is an uncalibrated distance decision. The output
    explicitly retains estimated boundaries and is separate from the existing
    complete-oracle transform API. Source recording IDs must be supplied in
    addition to the template's individual calibration-bout IDs.
    """
    def __init__(self, templates, *, native_sample_rate_hz, source_recording_ids):
        if not isinstance(templates, DocumentTemporalTemplatesV3):
            raise TypeError('Document V3 templates required')
        templates._check()
        if not np.isfinite(native_sample_rate_hz) or native_sample_rate_hz <= 0:
            raise ValueError('Positive native rate required')
        ids = tuple(source_recording_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError('Explicit unique source recording IDs required')
        self.templates = templates
        self.native_rate = float(native_sample_rate_hz)
        self.source_recording_ids = frozenset(ids)

    def read(self, bout, *, recording_id):
        if not isinstance(bout, DetectedBout):
            raise TypeError('Explicit estimated detected bout required')
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError('Explicit recording ID required')
        if recording_id in self.source_recording_ids or recording_id in self.templates.calibration_trial_ids_:
            raise ValueError('Evaluation recording overlaps calibration')
        x = np.asarray(bout.emg, dtype=float)
        bins = self.templates.samples_
        if (bout.sample_rate_hz != self.native_rate or x.ndim != 2
                or x.shape[1] != self.templates.channels_ or len(x) < bins
                or len(x) != bout.end-bout.start or bout.start < 0
                or bout.duration_seconds < 1. or not np.isfinite(x).all()):
            raise ValueError('Detected native bout violates duration or sensor contract')
        envelope = np.stack([np.sqrt(np.mean(part**2, axis=0)) for part in np.array_split(x, bins)])
        path = document_envelope_path(envelope)
        distances = np.array([document_dtw_distance(path, t, self.templates.band_)
                              for t in self.templates.templates_])
        return {'recording_id': recording_id, 'start': bout.start, 'end': bout.end,
                'native_duration_seconds': bout.duration_seconds,
                'boundary_kind': 'estimated', 'certified_full_coverage': False,
                'classes': self.templates.classes_.copy(), 'distances': distances,
                'nearest_class': self.templates.classes_[np.argmin(distances)],
                'calibrated_probability_available': False}
