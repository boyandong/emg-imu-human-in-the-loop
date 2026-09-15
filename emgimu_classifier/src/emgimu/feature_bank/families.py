from __future__ import annotations

from itertools import combinations

import numpy as np

from .core import FeatureBatch, FeatureFamily, FeatureRegistry


EPS = 1e-10


def _channel_names(channels: int) -> tuple[str, ...]:
    return tuple(f"ch{index + 1}" for index in range(channels))


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=1))


def _envelope(x: np.ndarray, width: int) -> np.ndarray:
    values = np.abs(np.asarray(x, dtype=np.float64))
    width = max(1, min(int(width), values.shape[1]))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.stack(
        [np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, values[:, :, c])
         for c in range(values.shape[2])],
        axis=2,
    )


def _covariances(x: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = max(values.shape[1] - 1, 1)
    cov = np.einsum("ntc,ntd->ncd", centered, centered) / denominator
    channels = values.shape[2]
    trace = np.trace(cov, axis1=1, axis2=2)
    scale = trace / channels
    cov = (1.0 - shrinkage) * cov + shrinkage * scale[:, None, None] * np.eye(channels)
    cov += EPS * np.eye(channels)[None, :, :]
    return cov / np.maximum(np.trace(cov, axis1=1, axis2=2)[:, None, None], EPS)


def _vech(matrices: np.ndarray) -> np.ndarray:
    channels = matrices.shape[1]
    rows, cols = np.triu_indices(channels)
    output = matrices[:, rows, cols].copy()
    output[:, rows != cols] *= np.sqrt(2.0)
    return output


def _sym_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    values = np.maximum(values, EPS) ** power
    return (vectors * values) @ vectors.T


def _sym_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * np.log(np.maximum(values, EPS))) @ vectors.T


class LocalDetailFamily(FeatureFamily):
    family_id = "F0_local_detail"

    def __init__(self) -> None:
        self.thresholds_: np.ndarray | None = None
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "LocalDetailFamily":
        x = np.asarray(batch.emg, dtype=np.float64)
        differences = np.abs(np.diff(x, axis=1)).reshape(-1, batch.channels)
        median = np.median(differences, axis=0)
        mad = np.median(np.abs(differences - median), axis=0)
        self.thresholds_ = np.maximum(median + 3.0 * 1.4826 * mad, EPS)
        metrics = ("rms", "mav", "wl", "zc", "ssc", "wamp")
        self._names = tuple(f"F0.{metric}.{channel}" for metric in metrics for channel in _channel_names(batch.channels))
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        x = np.asarray(batch.emg, dtype=np.float64)
        if x.shape[2] != len(self.thresholds_):
            raise ValueError("channel count differs from fitted F0 state")
        rms = _rms(x)
        mav = np.mean(np.abs(x), axis=1)
        differences = np.diff(x, axis=1)
        wl = np.sum(np.abs(differences), axis=1)
        threshold = self.thresholds_[None, None, :]
        zc = np.sum((x[:, :-1] * x[:, 1:] < 0) & (np.abs(differences) > threshold), axis=1)
        left = x[:, 1:-1] - x[:, :-2]
        right = x[:, 1:-1] - x[:, 2:]
        ssc = np.sum((left * right > 0) & (np.maximum(np.abs(left), np.abs(right)) > threshold), axis=1)
        wamp = np.sum(np.abs(differences) > threshold, axis=1)
        return np.nan_to_num(np.concatenate((rms, mav, wl, zc, ssc, wamp), axis=1)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class ScalePatternFamily(FeatureFamily):
    family_id = "F1_scale_pattern_x1h"

    def __init__(self) -> None:
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "ScalePatternFamily":
        self._names = tuple(f"F1.rms_pattern.{channel}" for channel in _channel_names(batch.channels))
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        rms = _rms(batch.emg)
        global_scale = np.sqrt(np.mean(rms * rms, axis=1, keepdims=True))
        return (rms / np.maximum(global_scale, EPS)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class TraceCovarianceFamily(FeatureFamily):
    family_id = "F2a_trace_covariance"

    def __init__(self, shrinkage: float = 0.05) -> None:
        self.shrinkage = float(shrinkage)
        self.channels_: int | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "TraceCovarianceFamily":
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError("shrinkage must be in [0,1)")
        self.channels_ = batch.channels
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.channels_:
            raise ValueError("channel count differs from fitted covariance state")
        return _vech(_covariances(batch.emg, self.shrinkage)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        rows, cols = np.triu_indices(self.channels_)
        return tuple(f"F2a.cov.ch{a + 1}.ch{b + 1}" for a, b in zip(rows, cols))


class CspSpatialFamily(FeatureFamily):
    family_id = "F2b_csp"

    def __init__(self, components_per_tail: int = 1, regularization: float = 1e-5) -> None:
        self.components_per_tail = int(components_per_tail)
        self.regularization = float(regularization)
        self.filters_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "CspSpatialFamily":
        if labels is None:
            raise ValueError("CSP requires training labels")
        y = np.asarray(labels)
        if len(y) != batch.windows:
            raise ValueError("labels must match windows")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("CSP requires at least two classes")
        cov = _covariances(batch.emg)
        filters = []
        keep = min(self.components_per_tail, max(1, batch.channels // 2))
        for label in classes:
            positive = cov[y == label].mean(axis=0)
            negative = cov[y != label].mean(axis=0)
            total = positive + negative + self.regularization * np.eye(batch.channels)
            whitened = _sym_power(total, -0.5) @ positive @ _sym_power(total, -0.5)
            values, vectors = np.linalg.eigh((whitened + whitened.T) * 0.5)
            selected = np.concatenate((vectors[:, -keep:], vectors[:, :keep]), axis=1)
            filters.append(_sym_power(total, -0.5) @ selected)
        self.filters_ = np.concatenate(filters, axis=1)
        self.classes_ = classes
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.filters_.shape[0]:
            raise ValueError("channel count differs from fitted CSP state")
        projected = np.einsum("ntc,ck->ntk", np.asarray(batch.emg, dtype=np.float64), self.filters_)
        variance = np.var(projected, axis=1)
        normalized = variance / np.maximum(variance.sum(axis=1, keepdims=True), EPS)
        return np.log(np.maximum(normalized, EPS)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return tuple(f"F2b.csp.{index + 1}" for index in range(self.filters_.shape[1]))


class SpdTangentFamily(FeatureFamily):
    family_id = "F2c_spd_tangent"

    def __init__(self, shrinkage: float = 0.05) -> None:
        self.shrinkage = float(shrinkage)
        self.reference_: np.ndarray | None = None

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "SpdTangentFamily":
        covariance = _covariances(batch.emg, self.shrinkage)
        log_mean = np.mean(np.stack([_sym_log(item) for item in covariance]), axis=0)
        values, vectors = np.linalg.eigh((log_mean + log_mean.T) * 0.5)
        self.reference_ = (vectors * np.exp(values)) @ vectors.T
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.reference_.shape[0]:
            raise ValueError("channel count differs from fitted SPD reference")
        inverse_root = _sym_power(self.reference_, -0.5)
        tangent = np.stack([_sym_log(inverse_root @ item @ inverse_root) for item in _covariances(batch.emg, self.shrinkage)])
        return _vech(tangent).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        rows, cols = np.triu_indices(self.reference_.shape[0])
        return tuple(f"F2c.tangent.ch{a + 1}.ch{b + 1}" for a, b in zip(rows, cols))


class RingGeometryFamily(FeatureFamily):
    family_id = "F3_ring_geometry"

    def __init__(self, envelope_ms: float = 25.0) -> None:
        self.envelope_ms = float(envelope_ms)
        self.channels_: int | None = None
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "RingGeometryFamily":
        if batch.channels < 4:
            raise ValueError("ring geometry requires at least four circularly ordered channels")
        self.channels_ = batch.channels
        lags = range(1, batch.channels // 2 + 1)
        names = []
        for lag in lags:
            names.extend(f"F3.rlcs.lag{lag}.{metric}" for metric in ("mean", "std"))
        names.extend(f"F3.ces.component{index + 1}" for index in range(batch.channels))
        for lag in lags:
            names.extend(f"F3.ringcov.lag{lag}.{metric}" for metric in ("mean", "median", "std", "q25", "q75", "temporal_delta"))
        self._names = tuple(names)
        self.fitted_ = True
        return self

    @staticmethod
    def _correlation(envelope: np.ndarray) -> np.ndarray:
        centered = envelope - envelope.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.sum(centered * centered, axis=1))
        denominator = np.maximum(scale[:, :, None] * scale[:, None, :], EPS)
        correlation = np.einsum("ntc,ntd->ncd", centered, centered) / denominator
        return np.nan_to_num(correlation)

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.channels_:
            raise ValueError("channel count differs from fitted ring state")
        width = max(1, round(batch.sample_rate_hz * self.envelope_ms / 1000.0))
        envelope = _envelope(batch.emg, width)
        correlation = self._correlation(envelope)
        covariance = _covariances(envelope)
        midpoint = envelope.shape[1] // 2
        early = _covariances(envelope[:, :midpoint])
        late = _covariances(envelope[:, midpoint:])
        pieces = []
        for lag in range(1, batch.channels // 2 + 1):
            values = np.stack([correlation[:, channel, (channel + lag) % batch.channels] for channel in range(batch.channels)], axis=1)
            pieces.extend((values.mean(axis=1, keepdims=True), values.std(axis=1, keepdims=True)))
        eigenvalues = np.linalg.eigvalsh(correlation)[:, ::-1]
        pieces.append(eigenvalues / np.maximum(eigenvalues.sum(axis=1, keepdims=True), EPS))
        for lag in range(1, batch.channels // 2 + 1):
            values = np.stack([covariance[:, channel, (channel + lag) % batch.channels] for channel in range(batch.channels)], axis=1)
            early_values = np.stack([early[:, channel, (channel + lag) % batch.channels] for channel in range(batch.channels)], axis=1)
            late_values = np.stack([late[:, channel, (channel + lag) % batch.channels] for channel in range(batch.channels)], axis=1)
            pieces.extend((values.mean(axis=1, keepdims=True), np.median(values, axis=1, keepdims=True), values.std(axis=1, keepdims=True), np.quantile(values, 0.25, axis=1, keepdims=True), np.quantile(values, 0.75, axis=1, keepdims=True), np.abs(late_values.mean(axis=1, keepdims=True) - early_values.mean(axis=1, keepdims=True))))
        return np.nan_to_num(np.concatenate(pieces, axis=1)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class SpectralStateFamily(FeatureFamily):
    family_id = "F4_spectral_state"

    def __init__(self, band_count: int = 4, cepstral_coefficients: int = 4) -> None:
        self.band_count = int(band_count)
        self.cepstral_coefficients = int(cepstral_coefficients)
        self.channels_: int | None = None
        self.bands_: tuple[tuple[float, float], ...] = ()
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "SpectralStateFamily":
        nyquist = batch.sample_rate_hz / 2.0
        low = min(20.0, nyquist * 0.1)
        high = min(450.0, nyquist * 0.95)
        if high <= low:
            raise ValueError("sampling rate cannot support an EMG spectral band")
        edges = np.linspace(low, high, self.band_count + 1)
        self.bands_ = tuple((float(a), float(b)) for a, b in zip(edges[:-1], edges[1:]))
        self.channels_ = batch.channels
        names = [f"F4.band{band + 1}.orientation.{channel}" for band in range(self.band_count) for channel in _channel_names(batch.channels)]
        names.extend(f"F4.{metric}.{channel}" for metric in ("total_power", "centroid", "median_frequency", "entropy") for channel in _channel_names(batch.channels))
        names.extend(f"F4.cepstral{k + 1}.{metric}" for k in range(self.cepstral_coefficients) for metric in ("mean", "std"))
        self._names = tuple(names)
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.channels_:
            raise ValueError("channel count differs from fitted spectral state")
        x = np.asarray(batch.emg, dtype=np.float64)
        tapered = (x - x.mean(axis=1, keepdims=True)) * np.hanning(x.shape[1])[None, :, None]
        psd = np.abs(np.fft.rfft(tapered, axis=1)) ** 2 / max(x.shape[1], 1)
        frequency = np.fft.rfftfreq(x.shape[1], 1.0 / batch.sample_rate_hz)
        band_energy = []
        for index, (low, high) in enumerate(self.bands_):
            mask = (frequency >= low) & (frequency <= high if index == len(self.bands_) - 1 else frequency < high)
            energy = psd[:, mask].sum(axis=1)
            band_energy.append(energy / np.maximum(np.linalg.norm(energy, axis=1, keepdims=True), EPS))
        total = psd.sum(axis=1)
        centroid = np.sum(psd * frequency[None, :, None], axis=1) / np.maximum(total, EPS)
        cumulative = np.cumsum(psd, axis=1)
        median_index = np.argmax(cumulative >= (0.5 * total[:, None, :]), axis=1)
        median_frequency = frequency[median_index]
        distribution = psd / np.maximum(total[:, None, :], EPS)
        entropy = -np.sum(distribution * np.log(np.maximum(distribution, EPS)), axis=1) / np.log(max(psd.shape[1], 2))
        log_spectrum = np.log(np.maximum(psd, EPS))
        bins = log_spectrum.shape[1]
        n = np.arange(bins)
        cepstral = []
        for k in range(1, self.cepstral_coefficients + 1):
            coefficient = np.sum(log_spectrum * np.cos(np.pi * (n + 0.5) * k / bins)[None, :, None], axis=1)
            cepstral.extend((coefficient.mean(axis=1, keepdims=True), coefficient.std(axis=1, keepdims=True)))
        output = np.concatenate((*band_energy, total, centroid, median_frequency, entropy, *cepstral), axis=1)
        return np.nan_to_num(output).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class TemporalFormFamily(FeatureFamily):
    family_id = "F5_temporal_form"

    def __init__(self, envelope_ms: float = 25.0) -> None:
        self.envelope_ms = float(envelope_ms)
        self.channels_: int | None = None
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "TemporalFormFamily":
        self.channels_ = batch.channels
        channels = _channel_names(batch.channels)
        metrics = ("early_rms", "late_rms", "late_minus_early", "log_early_late", "slope", "peak_time", "temporal_entropy")
        names = [f"F5.{metric}.{channel}" for metric in metrics for channel in channels]
        names.append("F5.spatial_map_velocity")
        self._names = tuple(names)
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        if batch.channels != self.channels_:
            raise ValueError("channel count differs from fitted temporal state")
        x = np.asarray(batch.emg, dtype=np.float64)
        midpoint = x.shape[1] // 2
        early, late = _rms(x[:, :midpoint]), _rms(x[:, midpoint:])
        envelope = _envelope(x, max(1, round(batch.sample_rate_hz * self.envelope_ms / 1000.0)))
        time = np.linspace(-0.5, 0.5, x.shape[1])
        slope = np.einsum("t,ntc->nc", time, envelope - envelope.mean(axis=1, keepdims=True)) / np.maximum(np.sum(time * time), EPS)
        peak_time = np.argmax(envelope, axis=1) / max(x.shape[1] - 1, 1)
        temporal_distribution = envelope / np.maximum(envelope.sum(axis=1, keepdims=True), EPS)
        temporal_entropy = -np.sum(temporal_distribution * np.log(np.maximum(temporal_distribution, EPS)), axis=1) / np.log(max(x.shape[1], 2))
        spatial = envelope / np.maximum(envelope.sum(axis=2, keepdims=True), EPS)
        velocity = np.linalg.norm(np.diff(spatial, axis=1), axis=2).mean(axis=1, keepdims=True)
        output = np.concatenate((early, late, late - early, np.log((early + EPS) / (late + EPS)), slope, peak_time, temporal_entropy, velocity), axis=1)
        return np.nan_to_num(output).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class BodyContextFamily(FeatureFamily):
    family_id = "F6_body_context"

    def __init__(self) -> None:
        self.postures_: tuple[str, ...] = ()
        self.use_imu_: bool = False
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "BodyContextFamily":
        self.use_imu_ = batch.imu is not None
        self.postures_ = () if batch.posture is None else tuple(sorted(str(value) for value in np.unique(batch.posture)))
        if not self.use_imu_ and not self.postures_:
            raise ValueError("body context requires real IMU or posture labels")
        names = []
        if self.use_imu_:
            names.extend(f"F6.{sensor}.{metric}" for sensor in ("accel", "gyro") for metric in ("mean_norm", "std_norm", "rms_norm", "max_norm", "range_norm"))
            names.extend(f"F6.gravity.{axis}" for axis in ("x", "y", "z"))
        names.extend(f"F6.posture.{value}" for value in self.postures_)
        self._names = tuple(names)
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        pieces = []
        if self.use_imu_:
            if batch.imu is None:
                raise ValueError("fitted body context requires IMU")
            imu = np.asarray(batch.imu, dtype=np.float64)
            for sensor in (imu[:, :, :3], imu[:, :, 3:]):
                norm = np.linalg.norm(sensor, axis=2)
                pieces.append(np.column_stack((norm.mean(axis=1), norm.std(axis=1), np.sqrt(np.mean(norm * norm, axis=1)), norm.max(axis=1), np.ptp(norm, axis=1))))
            gravity = imu[:, :, :3].mean(axis=1)
            pieces.append(gravity / np.maximum(np.linalg.norm(gravity, axis=1, keepdims=True), EPS))
        if self.postures_:
            if batch.posture is None:
                raise ValueError("fitted body context requires posture labels")
            values = np.asarray(batch.posture).astype(str)
            unknown = sorted(set(values) - set(self.postures_))
            if unknown:
                raise ValueError(f"unknown posture labels: {unknown}")
            pieces.append(np.column_stack([values == posture for posture in self.postures_]))
        return np.nan_to_num(np.concatenate(pieces, axis=1)).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


class QualityFamily(FeatureFamily):
    family_id = "F9_quality"

    def __init__(self, adc_min: float | None = None, adc_max: float | None = None, line_frequency_hz: float = 50.0) -> None:
        self.adc_min = adc_min
        self.adc_max = adc_max
        self.line_frequency_hz = float(line_frequency_hz)
        self.reference_median_: np.ndarray | None = None
        self.reference_mad_: np.ndarray | None = None
        self.reference_covariance_: np.ndarray | None = None
        self._names: tuple[str, ...] = ()

    def fit(self, batch: FeatureBatch, labels: np.ndarray | None = None) -> "QualityFamily":
        activation = _rms(batch.emg)
        self.reference_median_ = np.median(activation, axis=0)
        self.reference_mad_ = np.maximum(np.median(np.abs(activation - self.reference_median_), axis=0), EPS)
        self.reference_covariance_ = _covariances(batch.emg).mean(axis=0)
        per_channel = ("zero_fraction", "variance", "flatline_fraction", "clip_fraction", "line_noise_ratio", "amplitude_z")
        self._names = tuple(f"F9.{metric}.{channel}" for metric in per_channel for channel in _channel_names(batch.channels)) + ("F9.covariance_anomaly", "F9.bad_channel_count", "F9.mean_quality", "F9.min_quality", "F9.quality_variance")
        self.fitted_ = True
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        self._check()
        x = np.asarray(batch.emg, dtype=np.float64)
        zero_fraction = np.mean(np.abs(x) <= EPS, axis=1)
        variance = np.var(x, axis=1)
        flatline = np.mean(np.abs(np.diff(x, axis=1)) <= EPS, axis=1)
        if self.adc_min is None or self.adc_max is None:
            clip = np.zeros_like(variance)
        else:
            tolerance = max((self.adc_max - self.adc_min) * 1e-6, EPS)
            clip = np.mean((x <= self.adc_min + tolerance) | (x >= self.adc_max - tolerance), axis=1)
        tapered = (x - x.mean(axis=1, keepdims=True)) * np.hanning(x.shape[1])[None, :, None]
        psd = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
        frequency = np.fft.rfftfreq(x.shape[1], 1.0 / batch.sample_rate_hz)
        if self.line_frequency_hz >= batch.sample_rate_hz / 2.0:
            line_ratio = np.zeros_like(variance)
        else:
            line = np.abs(frequency - self.line_frequency_hz) <= 1.0
            neighbor = ((frequency >= self.line_frequency_hz - 6) & (frequency <= self.line_frequency_hz - 3)) | ((frequency >= self.line_frequency_hz + 3) & (frequency <= self.line_frequency_hz + 6))
            line_ratio = psd[:, line].mean(axis=1) / np.maximum(psd[:, neighbor].mean(axis=1), EPS)
        activation = _rms(x)
        amplitude_z = (activation - self.reference_median_) / (1.4826 * self.reference_mad_)
        covariance_distance = np.linalg.norm(_covariances(x) - self.reference_covariance_[None, :, :], axis=(1, 2))[:, None]
        anomaly = np.maximum.reduce((zero_fraction, flatline, clip, np.clip(np.abs(amplitude_z) / 6.0, 0.0, 1.0)))
        quality = 1.0 - np.clip(anomaly, 0.0, 1.0)
        summaries = np.column_stack(((quality < 0.5).sum(axis=1), quality.mean(axis=1), quality.min(axis=1), quality.var(axis=1)))
        output = np.concatenate((zero_fraction, variance, flatline, clip, line_ratio, amplitude_z, covariance_distance, summaries), axis=1)
        return np.nan_to_num(output).astype(np.float32)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._check()
        return self._names


def default_registry() -> FeatureRegistry:
    from .temporal import PathSignatureFamily, TemporalTemplateFamily

    registry = FeatureRegistry()
    for family in (
        LocalDetailFamily,
        ScalePatternFamily,
        TraceCovarianceFamily,
        CspSpatialFamily,
        SpdTangentFamily,
        RingGeometryFamily,
        SpectralStateFamily,
        TemporalFormFamily,
        TemporalTemplateFamily,
        PathSignatureFamily,
        BodyContextFamily,
        QualityFamily,
    ):
        registry.register(family.family_id, family)
    return registry
