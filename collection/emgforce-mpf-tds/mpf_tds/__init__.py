"""Single-participant MPF+TDS paper-based reproduction for EMGForce."""

from .collector import (
    GESTURE_ORDER,
    ActivityDetectorConfig,
    AdaptiveActivityDetector,
    ContinuousGestureRecorder,
    RecordedGestureEvent,
)
from .features import MPFConfig, MultiBandMatrixPowerFeatures
from .model import MPFTDSConfig, MPFTDSNetwork

__all__ = [
    "GESTURE_ORDER",
    "ActivityDetectorConfig",
    "AdaptiveActivityDetector",
    "ContinuousGestureRecorder",
    "RecordedGestureEvent",
    "MPFConfig",
    "MultiBandMatrixPowerFeatures",
    "MPFTDSConfig",
    "MPFTDSNetwork",
]
