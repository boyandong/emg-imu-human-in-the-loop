"""Canonical external benchmark datasets, separate from formal Session 1-4 data."""

from .benchmark import (
    BenchmarkDatasetError,
    BenchmarkTrial,
    check_benchmark_dataset,
    load_benchmark_trial,
    read_benchmark_report,
)
from .hla_schema import (
    ChannelSpec,
    DatasetManifestV2,
    EMGTrial,
    OntologyEntry,
    OntologyRelation,
    load_hla_manifest,
    load_hla_trial,
    check_hla_dataset,
)
from .hla_splits import PersonalizationSplit, TrialRef, build_personalization_split
from .hla_hardware import (
    DEFAULT_STREAM_RATES_HZ,
    MultiRateWindow,
    build_multirate_window,
    channel_metadata_matrix,
)
from .hla_features import (
    G0_TOKEN_NAMES,
    G5_TOKEN_NAMES,
    extract_g0_tokens,
    extract_g5_tokens,
    extract_hla_feature_tokens,
    hla_feature_names,
)
from .hla_windows import HLA_MAIN_PROTOCOL, TrialWindows, WindowProtocol, window_trial

__all__ = [
    "BenchmarkDatasetError",
    "BenchmarkTrial",
    "check_benchmark_dataset",
    "load_benchmark_trial",
    "read_benchmark_report",
    "ChannelSpec",
    "DatasetManifestV2",
    "EMGTrial",
    "OntologyEntry",
    "OntologyRelation",
    "load_hla_manifest",
    "load_hla_trial",
    "check_hla_dataset",
    "PersonalizationSplit",
    "TrialRef",
    "build_personalization_split",
    "DEFAULT_STREAM_RATES_HZ",
    "MultiRateWindow",
    "build_multirate_window",
    "channel_metadata_matrix",
    "G0_TOKEN_NAMES",
    "G5_TOKEN_NAMES",
    "extract_g0_tokens",
    "extract_g5_tokens",
    "extract_hla_feature_tokens",
    "hla_feature_names",
    "HLA_MAIN_PROTOCOL",
    "TrialWindows",
    "WindowProtocol",
    "window_trial",
]
