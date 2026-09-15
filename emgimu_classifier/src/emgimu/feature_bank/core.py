from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Windows plus only the context that a source actually provides."""

    emg: np.ndarray
    sample_rate_hz: float
    imu: np.ndarray | None = None
    posture: np.ndarray | None = None

    def __post_init__(self) -> None:
        emg = np.asarray(self.emg)
        if emg.ndim != 3 or emg.shape[1] < 3 or emg.shape[2] < 1:
            raise ValueError("emg must have shape [windows,samples>=3,channels>=1]")
        if not np.all(np.isfinite(emg)):
            raise ValueError("emg contains NaN or Inf")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.imu is not None:
            imu = np.asarray(self.imu)
            if imu.ndim != 3 or imu.shape[0] != emg.shape[0] or imu.shape[2] != 6:
                raise ValueError("imu must have shape [windows,samples,6]")
        if self.posture is not None and len(np.asarray(self.posture)) != emg.shape[0]:
            raise ValueError("posture must have one value per window")

    @property
    def windows(self) -> int:
        return int(self.emg.shape[0])

    @property
    def channels(self) -> int:
        return int(self.emg.shape[2])

    def take(self, indices: np.ndarray) -> "FeatureBatch":
        index = np.asarray(indices)
        return FeatureBatch(
            np.asarray(self.emg)[index],
            self.sample_rate_hz,
            None if self.imu is None else np.asarray(self.imu)[index],
            None if self.posture is None else np.asarray(self.posture)[index],
        )


class FeatureFamily(ABC):
    family_id: str
    fitted_: bool = False

    @abstractmethod
    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "FeatureFamily":
        raise NotImplementedError

    @abstractmethod
    def transform(self, batch: FeatureBatch) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def fit_transform(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> np.ndarray:
        return self.fit(batch, labels).transform(batch)

    def _check(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.family_id} must be fit on the training fold first")


class FeatureRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], FeatureFamily]] = {}

    def register(self, family_id: str, factory: Callable[[], FeatureFamily]) -> None:
        if family_id in self._factories:
            raise KeyError(f"duplicate feature family: {family_id}")
        self._factories[family_id] = factory

    def create(self, family_id: str) -> FeatureFamily:
        try:
            return self._factories[family_id]()
        except KeyError as exc:
            raise KeyError(f"unknown feature family: {family_id}") from exc

    def create_many(self, family_ids: Iterable[str]) -> list[FeatureFamily]:
        return [self.create(item) for item in family_ids]

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(self._factories)
