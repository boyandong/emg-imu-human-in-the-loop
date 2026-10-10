"""Availability-aware probability fusion with explicit quality and rejection.

Providers, probability calibration and population/personal weights come from
the frozen V1 source policy. Quality scores are externally computed observations
bound to an explicit, prespecified quality policy; this interface never fits a
quality threshold, a probability temperature or a confidence cutoff on queries.
No score means unavailable quality, not evidence of a healthy sensor.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .available_bank_fusion_v1 import AvailableBankFusionPolicyV1


@dataclass(frozen=True, slots=True)
class AvailableBankQualityFusionV2:
    source: AvailableBankFusionPolicyV1
    quality_policy_id: str
    minimum_confidence: float = 0.
    unknown_label: str = 'Unknown'

    def __post_init__(self):
        if not isinstance(self.source, AvailableBankFusionPolicyV1):
            raise ValueError('Frozen source fusion policy required')
        if not isinstance(self.quality_policy_id, str) or not self.quality_policy_id.strip():
            raise ValueError('Explicit prespecified quality policy identity required')
        if (not np.isfinite(self.minimum_confidence)
                or not 0 <= self.minimum_confidence <= 1):
            raise ValueError('Prespecified confidence cutoff must be in [0,1]')
        if (not isinstance(self.unknown_label, str) or not self.unknown_label.strip()
                or self.unknown_label in self.source.classes):
            raise ValueError('Unknown label must be explicit and outside gesture classes')

    @property
    def policy_id(self):
        fields = [self.source.policy_id, self.quality_policy_id,
                  self.minimum_confidence, self.unknown_label]
        return hashlib.sha256(json.dumps(fields, separators=(',', ':')).encode()).hexdigest()

    def calibrate(self, calibration, *, available=None, forbidden_evaluation_trials=()):
        """Use the unchanged trial-balanced V1 personal/population procedure."""
        return self.source.calibrate(calibration, available=available,
                                     forbidden_evaluation_trials=forbidden_evaluation_trials)

    def predict(self, state, probabilities, *, evaluation_trials,
                provider_trial_ids, provider_classes, quality=None,
                quality_trial_ids=None, quality_policy_id=None):
        """Return scoreable probabilities, row weights and explicit decisions.

        Quality may cover a subset of currently available providers. Unobserved
        scores leave their weight unchanged and remain explicitly unavailable.
        All observed scores must already be bounded in [0,1]; invalid values
        are rejected rather than silently clipped. All-quality-zero rows retain
        the quality-free distribution for scoring and emit Unknown. A separate
        strict confidence cutoff can also emit Unknown.

        The document's weight-times-quality operation is normalized onto the
        probability simplex. Positive scores are rescaled before multiplication
        to avoid incorrectly rejecting very small but positive common quality.
        This is an arithmetic interface, not a hardware-fault detector.
        """
        trials = tuple(evaluation_trials)
        base = self.source.predict(state, probabilities, evaluation_trials=trials,
                                   provider_trial_ids=provider_trial_ids,
                                   provider_classes=provider_classes)
        names = base['provider_names']
        weights = np.asarray(base['weights'], dtype=np.float64)
        scores = np.ones((len(trials), len(names)), dtype=np.float64)
        observed = np.zeros(len(names), dtype=bool)
        if quality is None:
            if quality_trial_ids is not None or quality_policy_id is not None:
                raise ValueError('Quality axes or identity supplied without observations')
        else:
            if (not isinstance(quality, Mapping) or not quality
                    or not set(quality) <= set(names)):
                raise ValueError('Quality must identify currently available providers')
            if quality_policy_id != self.quality_policy_id:
                raise ValueError('Quality observations belong to a different prespecified policy')
            if not isinstance(quality_trial_ids, Mapping) or set(quality_trial_ids) != set(quality):
                raise ValueError('Every observed quality provider needs explicit trial axes')
            for name, values in quality.items():
                if tuple(quality_trial_ids[name]) != trials:
                    raise ValueError('Quality trial order differs from prediction')
                values = np.asarray(values, dtype=np.float64)
                if (values.shape != (len(trials),) or not np.isfinite(values).all()
                        or np.any(values < 0) or np.any(values > 1)):
                    raise ValueError('Quality needs finite aligned scores in [0,1]')
                index = names.index(name)
                scores[:, index] = values
                observed[index] = True

        # A zero-weight branch cannot rescue an otherwise rejected row. Positive
        # common scaling of q does not change a normalized weighted mixture.
        largest = np.max(np.where(weights[None, :] > 0, scores, 0.), axis=1)
        all_rejected = largest == 0.
        positive = np.where(weights[None, :] > 0, scores, 0.)
        scaled = np.divide(positive, largest[:, None], out=np.zeros_like(scores),
                           where=largest[:, None] > 0)
        effective = scaled * weights[None, :]
        effective[all_rejected] = weights
        effective /= effective.sum(axis=1, keepdims=True)
        arrays = np.stack([np.asarray(probabilities[name], dtype=np.float64) for name in names], axis=1)
        fused = np.einsum('nk,nkh->nh', effective, arrays)
        # Keep the existing distribution bit-identical in the no-quality case.
        if quality is None:
            fused = base['probabilities'].copy()
        fused[all_rejected] = base['probabilities'][all_rejected]
        low_confidence = fused.max(axis=1) < self.minimum_confidence
        reasons = tuple('all_quality_rejected' if all_rejected[i]
                        else 'low_confidence' if low_confidence[i] else ''
                        for i in range(len(trials)))
        labels = np.asarray(self.source.classes, dtype=object)[fused.argmax(axis=1)]
        rejected = all_rejected | low_confidence
        labels[rejected] = self.unknown_label
        return {
            **base, 'probabilities': fused,
            'quality_free_probabilities': base['probabilities'].copy(),
            'effective_weights': effective, 'quality_scores': scores,
            'quality_available': tuple(bool(v) for v in observed),
            'quality_unavailable': tuple(n for i, n in enumerate(names) if not observed[i]),
            'quality_policy_id': self.quality_policy_id if quality is not None else None,
            'decision_policy_id': self.policy_id, 'evaluation_trials': trials,
            'labels': tuple(labels), 'rejected': rejected,
            'rejection_reason': reasons,
        }
