"""Opt-in F8 weights from calibrated session drift, with source-only scales."""
from itertools import combinations
import numpy as np
from .document_session_v3 import DocumentSessionDescriptorV3
from .affine_spd_anchor import affine_spd_distance


class DocumentSessionRouterV1:
    providers = ('TD24', 'pattern', 'SPD', 'log_bands')

    def fit_source(self, summary):
        if not isinstance(summary, DocumentSessionDescriptorV3) or not hasattr(summary, 'reference_'):
            raise ValueError('Fitted document F8 V3 source summary required')
        self.user_ = summary.user_id_
        self.sessions_ = tuple(sorted(summary.source_session_ids_))
        self.channels_ = summary.channels_; self.rate_ = summary.rate_
        self.classes_ = tuple(str(c) for c in sorted(summary.reference_))
        pairs = list(combinations(sorted(summary.reference_), 2))
        if not pairs:
            raise ValueError('At least two source classes required')
        ref = summary.reference_
        self.scales_ = {
            'pattern': float(np.mean([np.linalg.norm(ref[a]['pattern']-ref[b]['pattern']) for a, b in pairs])),
            'SPD': float(np.mean([affine_spd_distance(ref[a]['covariance'], ref[b]['covariance']) for a, b in pairs])),
            'log_bands': float(np.mean([np.abs(ref[a]['log_bands']-ref[b]['log_bands']).mean() for a, b in pairs])),
            'log_scale': float(np.mean([abs(ref[a]['log_scale']-ref[b]['log_scale']) for a, b in pairs])),
        }
        return self

    def weights(self, descriptor):
        if not hasattr(self, 'scales_'):
            raise RuntimeError('Source fit required')
        if (descriptor.get('version') != 'document-f8-v3' or descriptor.get('user_id') != self.user_
                or tuple(descriptor.get('source_sessions', ())) != self.sessions_
                or descriptor.get('sensor_channels') != self.channels_
                or descriptor.get('sample_rate_hz') != self.rate_
                or set(descriptor.get('classes', {})) != set(self.classes_)):
            raise ValueError('F8 descriptor identity or sensor contract differs')
        entries = [descriptor['classes'][c] for c in self.classes_]
        def normalized(value, key):
            scale = self.scales_[key]
            # An uninformative source geometry contributes no invented penalty.
            return float(value/scale) if scale > 1e-10 else 0.
        risks = np.array([
            normalized(np.mean([abs(e['log_global_activation_shift']) for e in entries]), 'log_scale')
                + np.mean([np.mean(np.abs(e['channel_quality_score_shift'])) for e in entries]),
            normalized(np.mean([e['scale_pattern_residual_norm'] for e in entries]), 'pattern'),
            normalized(np.mean([e['affine_invariant_trace_covariance_distance'] for e in entries]), 'SPD'),
            normalized(np.mean([e['mean_absolute_log_band_residual'] for e in entries]), 'log_bands'),
        ])
        if not np.isfinite(risks).all() or np.any(risks < 0):
            raise ValueError('Finite nonnegative session risks required')
        raw = np.maximum(np.exp(-np.clip(risks, 0., 5.)), .05)
        return raw/raw.sum(), risks

    def fuse(self, probability, descriptor):
        p = np.asarray(probability, dtype=float)
        if p.ndim != 3 or p.shape[0] != 4 or p.shape[2] != len(self.classes_):
            raise ValueError('Ordered four-provider [provider,trial,class] probabilities required')
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=2), 1.):
            raise ValueError('Valid normalized provider probabilities required')
        weights, risks = self.weights(descriptor)
        return np.tensordot(weights, p, axes=(0, 0)), weights, risks
