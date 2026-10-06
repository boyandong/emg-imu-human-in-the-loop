"""Read a complete detected interval with an existing source-fitted G5 classifier."""
import numpy as np
from .autonomous_bouts_v1 import DetectedBout
from .core import FeatureBatch
from .validated_unibo import ValidatedUniBoFamily
from .force_nested_oof import temperature_probability


class DetectedG5ReaderV1:
    def __init__(self, family, scaler, classifier, *, source_recording_ids, temperature=1.):
        if not isinstance(family, ValidatedUniBoFamily) or family.group != 'G5':
            raise ValueError('Source-fitted native UniBo G5 family required')
        family._check()
        if not np.array_equal(classifier.classes_, np.arange(4)):
            raise ValueError('Canonical four-class classifier required')
        ids = tuple(source_recording_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError('Explicit unique source recording IDs required')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Finite positive source temperature required')
        self.family = family; self.scaler = scaler; self.classifier = classifier
        self.source_recording_ids = frozenset(ids); self.temperature = float(temperature)

    def read(self, bout, *, recording_id):
        if not isinstance(bout, DetectedBout):
            raise TypeError('Explicit estimated detected bout required')
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError('Explicit recording ID required')
        if recording_id in self.source_recording_ids:
            raise ValueError('Evaluation recording overlaps source')
        x = np.asarray(bout.emg, dtype=float)
        if (bout.sample_rate_hz != 200. or x.ndim != 2 or x.shape[1] != 4
                or len(x) != bout.end-bout.start or bout.start < 0
                or bout.duration_seconds < 1. or not np.isfinite(x).all()):
            raise ValueError('Native four-channel/200Hz full-interval contract required')
        # Match the frozen training aggregation: every complete nonoverlapping
        # 200ms window, mean once across the bout; no invented padded samples.
        used = len(x)//40*40
        windows = x[:used].reshape(-1, 40, 4).astype(np.float32)
        features = self.family.transform(FeatureBatch(windows, 200.)).mean(axis=0, keepdims=True)
        raw = self.classifier.predict_proba(self.scaler.transform(features))
        probability = temperature_probability(raw, self.temperature)[0]
        return {'recording_id': recording_id, 'start': bout.start, 'end': bout.end,
                'boundary_kind': 'estimated', 'certified_full_coverage': False,
                'complete_windows': len(windows), 'unrepresented_tail_samples': len(x)-used,
                'probability': probability, 'prediction': int(np.argmax(probability)),
                'source_temperature': self.temperature}
