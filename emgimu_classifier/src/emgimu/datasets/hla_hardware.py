from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..signal import polyphase_resample
from .hla_schema import ChannelSpec


DEFAULT_STREAM_RATES_HZ = (200, 500, 1000)
ANATOMY_REGIONS = ("flexor", "extensor", "other", "unknown")


@dataclass(frozen=True, slots=True)
class MultiRateWindow:
    source_rate_hz: float
    streams: Mapping[int, np.ndarray]
    availability: np.ndarray
    channel_quality: np.ndarray

    def __post_init__(self) -> None:
        rates = tuple(self.streams)
        if rates != tuple(sorted(rates)):
            raise ValueError("streams must be sorted by target rate")
        channels = {values.shape[1] for values in self.streams.values()}
        if len(channels) != 1:
            raise ValueError("all streams must have the same channel count")
        if len(self.availability) != len(DEFAULT_STREAM_RATES_HZ):
            raise ValueError("availability must describe the default stream rates")
        if channels and len(self.channel_quality) != next(iter(channels)):
            raise ValueError("channel_quality length must match stream channel count")


def build_multirate_window(
    emg: np.ndarray,
    source_rate_hz: float,
    *,
    channel_quality: np.ndarray | None = None,
    stream_rates_hz: Sequence[int] = DEFAULT_STREAM_RATES_HZ,
) -> MultiRateWindow:
    """Build anti-aliased streams without inventing unavailable bandwidth."""

    values = np.asarray(emg, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 1:
        raise ValueError("emg must have shape [samples>=3, channels>=1]")
    if not np.isfinite(values).all() or source_rate_hz <= 0:
        raise ValueError("emg must be finite and source_rate_hz must be positive")
    rates = tuple(map(int, stream_rates_hz))
    if rates != tuple(sorted(set(rates))) or any(rate <= 0 for rate in rates):
        raise ValueError("stream_rates_hz must be unique, sorted, and positive")
    quality = (
        np.ones(values.shape[1], dtype=np.float32)
        if channel_quality is None else np.asarray(channel_quality, dtype=np.float32)
    )
    if quality.shape != (values.shape[1],) or not np.isfinite(quality).all():
        raise ValueError("channel_quality must be one finite value per channel")
    if np.any((quality < 0) | (quality > 1)):
        raise ValueError("channel_quality must lie in [0,1]")

    streams: dict[int, np.ndarray] = {}
    availability = np.zeros(len(rates), dtype=bool)
    for index, rate in enumerate(rates):
        if source_rate_hz + 1e-9 < rate:
            continue
        availability[index] = True
        if np.isclose(source_rate_hz, rate):
            stream = values.copy()
        else:
            stream = polyphase_resample(values, source_rate_hz, float(rate))
        streams[rate] = np.asarray(stream, dtype=np.float32)
    if not streams:
        raise ValueError(
            f"source rate {source_rate_hz:g} Hz is below every configured stream rate {rates}"
        )
    return MultiRateWindow(float(source_rate_hz), streams, availability, quality)


def channel_metadata_matrix(channels: Sequence[ChannelSpec]) -> np.ndarray:
    """Encode optional geometry/anatomy with explicit availability masks.

    Columns are xyz, xyz-known, sin/cos ring angle, angle-known, followed by
    flexor/extensor/other/unknown anatomy one-hot values.
    """

    rows: list[list[float]] = []
    for channel in channels:
        if channel.position_xyz is None:
            xyz = (0.0, 0.0, 0.0)
            xyz_known = 0.0
        else:
            xyz = channel.position_xyz
            xyz_known = 1.0
        if channel.ring_angle_deg is None:
            angle = (0.0, 0.0)
            angle_known = 0.0
        else:
            radians = np.deg2rad(channel.ring_angle_deg)
            angle = (float(np.sin(radians)), float(np.cos(radians)))
            angle_known = 1.0
        region = (channel.anatomical_region or "unknown").lower()
        if region not in ANATOMY_REGIONS:
            region = "other"
        anatomy = [float(region == name) for name in ANATOMY_REGIONS]
        rows.append([
            *map(float, xyz), xyz_known, *angle, angle_known, *anatomy,
        ])
    return np.asarray(rows, dtype=np.float32)
