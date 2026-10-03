"""Opt-in document-exact D/E reliability from labelled calibration trials.

The caller must supply a source/inner-validation-selected population policy.
No held-out performance is used here to select n0, temperature or providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


EPS = 1e-10


@dataclass(frozen=True, slots=True)
class DocumentReliabilityWeightsV2:
    classes: tuple[int | str, ...]
    family_ids: tuple[str, ...]
    population: tuple[float, ...]
    n0: float
    temperature: float
    source_policy_id: str

    def __post_init__(self) -> None:
        population = np.asarray(self.population, dtype=np.float64)
        if (len(self.classes) < 2 or len(set(self.classes)) != len(self.classes)
                or not self.family_ids or len(set(self.family_ids)) != len(self.family_ids)
                or population.shape != (len(self.family_ids),) or not np.isfinite(population).all()
                or np.any(population < 0) or not np.isclose(population.sum(), 1, atol=1e-10)):
            raise ValueError("unique classes/families and normalized source population weights required")
        if (not np.isfinite(self.n0) or self.n0 <= 0 or not np.isfinite(self.temperature)
                or self.temperature <= 0 or not isinstance(self.source_policy_id, str)
                or not self.source_policy_id.strip()):
            raise ValueError("positive source-selected n0/temperature and policy identity required")
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "family_ids", tuple(self.family_ids))
        object.__setattr__(self, "population", tuple(float(value) for value in population))

    @staticmethod
    def _trials(features: np.ndarray, labels: np.ndarray, trial_ids: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
        ids = np.asarray(trial_ids, dtype=object)
        if (x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all()
                or y.ndim != 1 or ids.ndim != 1 or len(x) != len(y) or len(x) != len(ids)
                or any(not isinstance(item, str) or not item.strip() for item in ids)):
            raise ValueError("finite aligned calibration rows, labels and nonempty trial IDs required")
        vectors, trial_labels = {}, {}
        for trial in sorted(set(ids.tolist())):
            mask = ids == trial
            unique = np.unique(y[mask])
            if len(unique) != 1:
                raise ValueError("one calibration trial cannot have multiple gesture labels")
            vectors[trial] = x[mask].mean(axis=0)
            trial_labels[trial] = unique[0]
        return vectors, trial_labels

    def calculate(self, calibration: Mapping[str, tuple[np.ndarray, np.ndarray, Sequence[str]]],
                  *, forbidden_trial_ids: Sequence[str] = ()) -> dict:
        if set(calibration) != set(self.family_ids):
            raise ValueError("every source-selected family needs aligned calibration trials")
        forbidden = set(forbidden_trial_ids)
        canonical_labels = None
        between, within = [], []
        for family in self.family_ids:
            vectors, labels = self._trials(*calibration[family])
            if forbidden.intersection(vectors):
                raise ValueError("source/evaluation trials cannot fit personal reliability")
            if set(labels.values()) != set(self.classes):
                raise ValueError("calibration must cover every source-selected gesture class")
            if canonical_labels is None:
                canonical_labels = labels
            elif labels != canonical_labels:
                raise ValueError("families must describe the same trial identities and labels")
            prototypes = {label: np.stack([vectors[trial] for trial in labels if labels[trial] == label]).mean(axis=0)
                          for label in self.classes}
            between.append(float(np.mean([np.linalg.norm(prototypes[a] - prototypes[b])
                                          for a, b in combinations(self.classes, 2)])))
            within.append(float(np.mean([np.mean([np.linalg.norm(vectors[trial] - prototypes[label])
                                                 for trial in labels if labels[trial] == label])
                                         for label in self.classes])))
        n_cal = len(canonical_labels)
        b, w = np.asarray(between), np.asarray(within)
        reliability = b / (w + EPS)
        log_reliability = np.log(reliability + EPS)
        logits = log_reliability / self.temperature
        exp = np.exp(logits - logits.max())
        personal = exp / exp.sum()
        alpha = self.n0 / (self.n0 + n_cal)
        final = alpha * np.asarray(self.population) + (1 - alpha) * personal
        if not np.isfinite(final).all() or not np.isclose(final.sum(), 1, atol=1e-12):
            raise ValueError("personal reliability must produce normalized finite weights")
        return {"between": b, "within": w, "reliability": reliability,
                "log_reliability": log_reliability, "personal": personal,
                "n_cal_trials": n_cal, "alpha": alpha, "final": final}

    def personal(self, calibration: Mapping[str, tuple[np.ndarray, np.ndarray, Sequence[str]]],
                 *, forbidden_trial_ids: Sequence[str] = ()) -> np.ndarray:
        return self.calculate(calibration, forbidden_trial_ids=forbidden_trial_ids)["final"]
