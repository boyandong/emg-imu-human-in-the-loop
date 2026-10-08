"""Opt-in arbitrary-provider fusion with calibration-only weights and missing branches.

Availability is caller-declared, not an inferred hardware-quality diagnosis.
Representations and population policy must already be fitted on source data.
"""
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
import numpy as np
from .document_reliability_v2 import DocumentReliabilityWeightsV2


def _ids(values, name):
    ids = tuple(values)
    if not ids or any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError(name + ' needs unique explicit native trial IDs')
    return ids


@dataclass(frozen=True, slots=True)
class AvailableBankFusionStateV1:
    policy_id: str
    providers: tuple[str, ...]
    weights: tuple[float, ...]
    source_trials: tuple[str, ...]
    calibration_trials: tuple[str, ...]
    mode: str
    n_cal_trials: int
    alpha: float | None


@dataclass(frozen=True, slots=True)
class AvailableBankFusionPolicyV1:
    classes: tuple
    providers: tuple[str, ...]
    population: tuple[float, ...]
    n0: float
    temperature: float
    source_policy_id: str
    source_trials: tuple[str, ...]

    def __post_init__(self):
        # Reuse the frozen document-exact numerical policy; this wrapper adds
        # availability and immutable provenance, without changing V2 behavior.
        p = DocumentReliabilityWeightsV2(self.classes, self.providers, self.population,
                                         self.n0, self.temperature, self.source_policy_id)
        if any(not isinstance(v, str) or not v.strip() for v in p.family_ids):
            raise ValueError('Explicit provider names required')
        object.__setattr__(self, 'classes', p.classes)
        object.__setattr__(self, 'providers', p.family_ids)
        object.__setattr__(self, 'population', p.population)
        object.__setattr__(self, 'source_trials', _ids(self.source_trials, 'source'))

    @property
    def policy_id(self):
        fields = [self.classes, self.providers, self.population, self.n0,
                  self.temperature, self.source_policy_id, self.source_trials]
        return hashlib.sha256(json.dumps(fields, separators=(',', ':')).encode()).hexdigest()

    def _available(self, available):
        active = tuple(self.providers if available is None else available)
        if not active or len(set(active)) != len(active) or not set(active) <= set(self.providers):
            raise ValueError('Unique available source-selected providers required')
        return tuple(v for v in self.providers if v in active)

    def calibrate(self, calibration: Mapping, *, available=None, forbidden_evaluation_trials=()):
        """Skip available branches lacking calibration; zero-shot uses population.

        Calibration rows can repeat a native trial, but each trial is weighted
        equally by DocumentReliabilityWeightsV2. No evaluation labels accepted.
        """
        if not isinstance(calibration, Mapping) or not set(calibration) <= set(self.providers):
            raise ValueError('Calibration mapping contains unknown providers')
        active = self._available(available)
        if set(calibration) - set(active): raise ValueError('Unavailable provider supplied calibration')
        if calibration:
            active = tuple(v for v in active if v in calibration)
        prior = np.array([self.population[self.providers.index(v)] for v in active])
        if prior.sum() <= 0: raise ValueError('Available providers have zero population mass')
        prior /= prior.sum()
        if not calibration:
            return AvailableBankFusionStateV1(self.policy_id, active, tuple(prior), self.source_trials,
                                               (), 'source_population_zero_shot', 0, 1.)
        policy = DocumentReliabilityWeightsV2(self.classes, active, tuple(prior), self.n0,
                                              self.temperature, self.source_policy_id)
        stats = policy.calculate(calibration, forbidden_trial_ids=self.source_trials + tuple(forbidden_evaluation_trials))
        # V2 already proves all providers refer to identical labelled trials.
        ids = tuple(sorted(set(calibration[active[0]][2])))
        return AvailableBankFusionStateV1(self.policy_id, active, tuple(float(w) for w in stats['final']),
                                           self.source_trials, ids, 'calibration_only_document_reliability',
                                           stats['n_cal_trials'], float(stats['alpha']))

    def from_fitted_weights(self, weights, *, calibration_trials, available=None):
        """Import an independently frozen policy, without claiming V1 fitted it.

        Used for replay of existing source/calibration-selected weights. The
        caller supplies their provenance; inference cannot prove fitting history.
        """
        active = self._available(available)
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(active),) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1., rtol=0., atol=1e-10):
            raise ValueError('Normalized finite available-provider weights required')
        cal = _ids(calibration_trials, 'calibration') if len(calibration_trials) else ()
        if set(cal) & set(self.source_trials): raise ValueError('Source/calibration trial overlap')
        return AvailableBankFusionStateV1(self.policy_id, active, tuple(float(v) for v in w),
                                           self.source_trials, cal, 'imported_frozen_weights', len(cal), None)

    def predict(self, state, probabilities, *, evaluation_trials, provider_trial_ids, provider_classes):
        """Fuse unique per-trial rows, dropping missing branches and renormalizing.

        Missing branches do not cause any refit or calibration update. No target
        quality threshold, probability temperature or weight tuning occurs here.
        """
        if not isinstance(state, AvailableBankFusionStateV1) or state.policy_id != self.policy_id:
            raise ValueError('State belongs to a different source policy')
        if state.source_trials != self.source_trials or not set(state.providers) <= set(self.providers):
            raise ValueError('State source/provider identity mismatch')
        if not state.providers or len(set(state.providers)) != len(state.providers):
            raise ValueError('Unique state providers required')
        if state.providers != tuple(v for v in self.providers if v in state.providers):
            raise ValueError('State provider order differs from source policy')
        weights = np.asarray(state.weights, dtype=float)
        if weights.shape != (len(state.providers),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-10):
            raise ValueError('Invalid frozen state weights')
        if state.calibration_trials:
            _ids(state.calibration_trials, 'calibration')
        if state.n_cal_trials != len(state.calibration_trials):
            raise ValueError('State calibration count differs from native IDs')
        if state.mode == 'source_population_zero_shot':
            if state.calibration_trials or state.alpha != 1.: raise ValueError('Invalid zero-shot state')
        elif state.mode == 'calibration_only_document_reliability':
            if not state.calibration_trials or state.alpha != self.n0 / (self.n0 + state.n_cal_trials):
                raise ValueError('Invalid calibration shrinkage state')
        elif state.mode == 'imported_frozen_weights':
            if state.alpha is not None: raise ValueError('Imported state cannot claim fitted shrinkage')
        else:
            raise ValueError('Unknown state fitting mode')
        if set(state.calibration_trials) & set(self.source_trials): raise ValueError('Source/calibration overlap')
        evaluation = _ids(evaluation_trials, 'evaluation')
        if set(evaluation) & (set(self.source_trials) | set(state.calibration_trials)):
            raise ValueError('Source/calibration/evaluation trial overlap')
        if not isinstance(probabilities, Mapping) or not probabilities or not set(probabilities) <= set(state.providers):
            raise ValueError('Nonempty active-provider probabilities required')
        if set(provider_trial_ids) != set(probabilities) or set(provider_classes) != set(probabilities):
            raise ValueError('Each available provider needs trial and class axes')
        names = tuple(v for v in state.providers if v in probabilities)
        selected = np.array([weights[state.providers.index(v)] for v in names])
        if selected.sum() <= 0: raise ValueError('Remaining providers have zero fitted weight')
        selected /= selected.sum()
        arrays = []
        for name in names:
            if tuple(provider_trial_ids[name]) != evaluation or tuple(provider_classes[name]) != self.classes:
                raise ValueError('Provider trial/class order differs')
            q = np.asarray(probabilities[name], dtype=float)
            if q.shape != (len(evaluation), len(self.classes)) or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1) or not np.allclose(q.sum(1), 1., rtol=0., atol=1e-8):
                raise ValueError('Normalized finite aligned probabilities required')
            arrays.append(q)
        fused = np.tensordot(selected, np.stack(arrays), axes=(0, 0))
        return {'probabilities': fused, 'provider_names': names, 'weights': tuple(selected),
                'missing_at_calibration': tuple(v for v in self.providers if v not in state.providers),
                'missing_at_prediction': tuple(v for v in state.providers if v not in probabilities),
                'mode': state.mode, 'n_cal_trials': state.n_cal_trials, 'policy_id': self.policy_id}
