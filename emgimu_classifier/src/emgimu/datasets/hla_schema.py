from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..state import Gesture
from .benchmark import BenchmarkDatasetError


HLA_SCHEMA_VERSION = "2.0"


class OntologyRelation(str, Enum):
    """How safely a source gesture can enter a cross-dataset label space."""

    EXACT = "exact"
    RELATED = "related"
    DATASET_ONLY = "dataset_only"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    channel_id: str
    name: str | None = None
    position_xyz: tuple[float, float, float] | None = None
    coordinate_system: str | None = None
    anatomical_region: str | None = None
    ring_angle_deg: float | None = None
    usable_band_hz: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.channel_id.strip():
            raise ValueError("channel_id cannot be empty")
        if self.position_xyz is not None:
            if len(self.position_xyz) != 3 or not np.isfinite(self.position_xyz).all():
                raise ValueError("position_xyz must contain three finite coordinates")
        if self.ring_angle_deg is not None and not np.isfinite(self.ring_angle_deg):
            raise ValueError("ring_angle_deg must be finite")
        if self.usable_band_hz is not None:
            low, high = self.usable_band_hz
            if not np.isfinite((low, high)).all() or low < 0 or high <= low:
                raise ValueError("usable_band_hz must be a finite increasing nonnegative pair")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "name": self.name,
            "position_xyz": None if self.position_xyz is None else list(self.position_xyz),
            "coordinate_system": self.coordinate_system,
            "anatomical_region": self.anatomical_region,
            "ring_angle_deg": self.ring_angle_deg,
            "usable_band_hz": None if self.usable_band_hz is None else list(self.usable_band_hz),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelSpec":
        position = value.get("position_xyz")
        band = value.get("usable_band_hz")
        return cls(
            channel_id=str(value["channel_id"]),
            name=None if value.get("name") is None else str(value["name"]),
            position_xyz=None if position is None else tuple(map(float, position)),
            coordinate_system=(
                None if value.get("coordinate_system") is None
                else str(value["coordinate_system"])
            ),
            anatomical_region=(
                None if value.get("anatomical_region") is None
                else str(value["anatomical_region"])
            ),
            ring_angle_deg=(
                None if value.get("ring_angle_deg") is None
                else float(value["ring_angle_deg"])
            ),
            usable_band_hz=None if band is None else tuple(map(float, band)),
        )


@dataclass(frozen=True, slots=True)
class OntologyEntry:
    source_value: str
    source_name: str
    task_label: int | None
    relation: OntologyRelation
    canonical_gesture: Gesture | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", OntologyRelation(self.relation))
        if not self.source_value or not self.source_name:
            raise ValueError("source_value and source_name cannot be empty")
        if self.task_label is not None and self.task_label < 0:
            raise ValueError("task_label must be nonnegative when present")
        if self.canonical_gesture is not None:
            object.__setattr__(self, "canonical_gesture", Gesture(self.canonical_gesture))
        if self.relation is OntologyRelation.EXACT:
            if self.canonical_gesture is None or self.canonical_gesture is Gesture.UNKNOWN:
                raise ValueError("exact ontology entries require a known canonical_gesture")
        elif self.canonical_gesture is not None:
            raise ValueError("related and dataset_only entries cannot enter the canonical label space")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_value": self.source_value,
            "source_name": self.source_name,
            "task_label": self.task_label,
            "relation": self.relation.value,
            "canonical_gesture": (
                None if self.canonical_gesture is None else int(self.canonical_gesture)
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OntologyEntry":
        canonical = value.get("canonical_gesture")
        return cls(
            source_value=str(value["source_value"]),
            source_name=str(value["source_name"]),
            task_label=None if value.get("task_label") is None else int(value["task_label"]),
            relation=OntologyRelation(str(value["relation"])),
            canonical_gesture=None if canonical is None else Gesture(int(canonical)),
            notes=str(value.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifestV2:
    dataset_id: str
    dataset_version: str
    adapter_version: str
    source_url: str
    license_id: str
    native_sample_rate_hz: float
    has_imu: bool
    channel_layouts: Mapping[str, tuple[ChannelSpec, ...]]
    ontology: tuple[OntologyEntry, ...]
    session_semantics: str = "unspecified"
    schema_version: str = HLA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HLA_SCHEMA_VERSION:
            raise ValueError(f"unsupported HLA schema version {self.schema_version!r}")
        if not self.dataset_id or self.native_sample_rate_hz <= 0:
            raise ValueError("dataset_id and a positive native_sample_rate_hz are required")
        if not self.channel_layouts:
            raise ValueError("at least one channel layout is required")
        normalized: dict[str, tuple[ChannelSpec, ...]] = {}
        for layout_id, channels in self.channel_layouts.items():
            values = tuple(channels)
            if not layout_id or not values:
                raise ValueError("layout identifiers and channel lists cannot be empty")
            identifiers = [channel.channel_id for channel in values]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate channel_id in layout {layout_id}")
            normalized[str(layout_id)] = values
        object.__setattr__(self, "channel_layouts", normalized)
        ontology = tuple(self.ontology)
        source_values = [entry.source_value for entry in ontology]
        if len(source_values) != len(set(source_values)):
            raise ValueError("ontology source_value entries must be unique")
        object.__setattr__(self, "ontology", ontology)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "adapter_version": self.adapter_version,
            "source_url": self.source_url,
            "license_id": self.license_id,
            "native_sample_rate_hz": self.native_sample_rate_hz,
            "has_imu": self.has_imu,
            "channel_layouts": {
                key: [channel.to_dict() for channel in channels]
                for key, channels in self.channel_layouts.items()
            },
            "ontology": [entry.to_dict() for entry in self.ontology],
            "session_semantics": self.session_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifestV2":
        layouts = value.get("channel_layouts")
        if not isinstance(layouts, Mapping):
            raise ValueError("channel_layouts must be an object")
        ontology = value.get("ontology")
        if not isinstance(ontology, Sequence) or isinstance(ontology, (str, bytes)):
            raise ValueError("ontology must be an array")
        return cls(
            schema_version=str(value["schema_version"]),
            dataset_id=str(value["dataset_id"]),
            dataset_version=str(value.get("dataset_version", "unknown")),
            adapter_version=str(value["adapter_version"]),
            source_url=str(value["source_url"]),
            license_id=str(value["license_id"]),
            native_sample_rate_hz=float(value["native_sample_rate_hz"]),
            has_imu=bool(value["has_imu"]),
            channel_layouts={
                str(key): tuple(ChannelSpec.from_dict(item) for item in items)
                for key, items in layouts.items()
            },
            ontology=tuple(OntologyEntry.from_dict(item) for item in ontology),
            session_semantics=str(value.get("session_semantics", "unspecified")),
        )

    def ontology_by_source(self) -> dict[str, OntologyEntry]:
        return {entry.source_value: entry for entry in self.ontology}


@dataclass(frozen=True, slots=True)
class EMGTrial:
    path: Path
    dataset_id: str
    subject_id: str
    session_id: str
    trial_id: str
    channel_layout_id: str
    sample_rate_hz: float
    timestamp_ms: np.ndarray
    emg: np.ndarray
    source_label: np.ndarray
    task_label: np.ndarray
    canonical_label: np.ndarray
    stable_mask: np.ndarray
    posture: str | None = None
    imu: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.timestamp_ms)
        if n < 2 or self.emg.ndim != 2 or len(self.emg) != n:
            raise ValueError("timestamp_ms and emg must describe at least two aligned samples")
        for name in ("source_label", "task_label", "canonical_label", "stable_mask"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length does not match timestamp_ms")
        if self.imu is not None and (self.imu.ndim != 2 or len(self.imu) != n):
            raise ValueError("imu must align with timestamp_ms")
        if not np.isfinite(self.timestamp_ms).all() or not np.isfinite(self.emg).all():
            raise ValueError("timestamp_ms and emg must be finite")
        if self.imu is not None and not np.isfinite(self.imu).all():
            raise ValueError("imu must be finite")
        if np.any(np.diff(self.timestamp_ms) <= 0):
            raise ValueError("timestamp_ms must be strictly increasing")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BenchmarkDatasetError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkDatasetError(f"{path} must contain a JSON object")
    return value


def _legacy_unibo_manifest(root: Path, value: Mapping[str, Any]) -> DatasetManifestV2:
    label_map = _read_json(root / "label_map.json")
    source_labels = label_map.get("source_labels")
    if value.get("dataset_id") != "unibo-inail" or not isinstance(source_labels, Mapping):
        raise BenchmarkDatasetError("only UniBo schema v1 can be upgraded to HLA schema v2")
    regions = {
        "extensor_carpi_ulnaris": "extensor",
        "extensor_digitorum_communis": "extensor",
        "flexor_carpi_radialis": "flexor",
        "flexor_carpi_ulnaris": "flexor",
    }
    channels = tuple(
        ChannelSpec(f"ch{index + 1}", str(name), anatomical_region=regions.get(str(name)))
        for index, name in enumerate(value["channel_names"])
    )
    ontology: list[OntologyEntry] = []
    for source_value, source in source_labels.items():
        source_name = str(source["name"])
        task = int(source["hand_label"])
        if source_name == "power_grip":
            relation = OntologyRelation.RELATED
            canonical = None
            notes = "Related to a closed-hand class but not semantically identical to generic fist."
        elif bool(source["eligible"]):
            relation = OntologyRelation.EXACT
            canonical = Gesture(task)
            notes = ""
        else:
            relation = OntologyRelation.DATASET_ONLY
            canonical = None
            notes = "Excluded from the frozen four-class UniBo pilot."
        ontology.append(OntologyEntry(
            str(source_value), source_name,
            None if task == int(Gesture.UNKNOWN) else task,
            relation, canonical, notes,
        ))
    return DatasetManifestV2(
        dataset_id="unibo-inail",
        dataset_version="official-main",
        adapter_version=str(value["adapter_version"]),
        source_url=str(value["source_url"]),
        license_id=str(value["license_id"]),
        native_sample_rate_hz=float(value["source_sample_rate_hz"]),
        has_imu=bool(value["has_imu"]),
        channel_layouts={"unibo_named_4": channels},
        ontology=tuple(ontology),
        session_semantics="recording_day",
    )


def load_hla_manifest(root: str | Path) -> DatasetManifestV2:
    dataset_root = Path(root)
    value = _read_json(dataset_root / "manifest.json")
    if str(value.get("schema_version")) == HLA_SCHEMA_VERSION:
        try:
            return DatasetManifestV2.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkDatasetError(f"invalid HLA manifest: {exc}") from exc
    return _legacy_unibo_manifest(dataset_root, value)


def _scalar(container: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(container[key])
    if value.size != 1:
        raise BenchmarkDatasetError(f"{key} must be scalar")
    return value.reshape(-1)[0]


def load_hla_trial(path: str | Path, manifest: DatasetManifestV2) -> EMGTrial:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as handle:
            files = set(handle.files)
            legacy = "dataset_id" not in files
            if legacy and manifest.dataset_id != "unibo-inail":
                raise BenchmarkDatasetError("legacy trials are supported only for UniBo")
            dataset_id = manifest.dataset_id if legacy else str(_scalar(handle, "dataset_id"))
            subject_id = str(_scalar(handle, "subject_id"))
            session_id = str(_scalar(handle, "session_id"))
            trial_id = str(_scalar(handle, "trial_id"))
            layout_id = (
                "unibo_named_4" if legacy else str(_scalar(handle, "channel_layout_id"))
            )
            rate = (
                1000.0 / float(np.median(np.diff(np.asarray(handle["timestamp_ms"]))))
                if legacy else float(_scalar(handle, "sample_rate_hz"))
            )
            timestamp = np.asarray(handle["timestamp_ms"], dtype=np.float64).reshape(-1)
            emg = np.asarray(handle["emg"], dtype=np.float64)
            stable = np.asarray(handle["stable_mask"], dtype=bool).reshape(-1)
            source_label = np.asarray(
                handle["source_relabel"] if legacy else handle["source_label"]
            ).astype(str).reshape(-1)
            task_label = np.asarray(
                handle["hand_label"] if legacy else handle["task_label"], dtype=np.int16
            ).reshape(-1)
            if legacy:
                mapping = manifest.ontology_by_source()
                canonical_label = np.asarray([
                    int(mapping[value].canonical_gesture)
                    if value in mapping and mapping[value].relation is OntologyRelation.EXACT
                    else int(Gesture.UNKNOWN)
                    for value in source_label
                ], dtype=np.int16)
                posture = str(int(_scalar(handle, "posture_label")))
            else:
                canonical_label = np.asarray(handle["canonical_label"], dtype=np.int16).reshape(-1)
                posture = None if "posture" not in files else str(_scalar(handle, "posture"))
            imu = None if "imu" not in files else np.asarray(handle["imu"], dtype=np.float64)
    except BenchmarkDatasetError:
        raise
    except Exception as exc:
        raise BenchmarkDatasetError(f"cannot read HLA trial {source}: {exc}") from exc
    if dataset_id != manifest.dataset_id:
        raise BenchmarkDatasetError(
            f"trial dataset_id {dataset_id!r} does not match manifest {manifest.dataset_id!r}"
        )
    if layout_id not in manifest.channel_layouts:
        raise BenchmarkDatasetError(f"unknown channel layout {layout_id!r}")
    if emg.ndim != 2 or emg.shape[1] != len(manifest.channel_layouts[layout_id]):
        raise BenchmarkDatasetError("EMG channel count does not match channel layout")
    try:
        return EMGTrial(
            source, dataset_id, subject_id, session_id, trial_id, layout_id, rate,
            timestamp, emg, source_label, task_label, canonical_label, stable, posture, imu,
        )
    except ValueError as exc:
        raise BenchmarkDatasetError(f"invalid HLA trial {source}: {exc}") from exc


def write_hla_manifest(path: str | Path, manifest: DatasetManifestV2) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_hla_trial(path: str | Path, trial: EMGTrial) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, np.ndarray] = {
        "dataset_id": np.asarray(trial.dataset_id),
        "subject_id": np.asarray(trial.subject_id),
        "session_id": np.asarray(trial.session_id),
        "trial_id": np.asarray(trial.trial_id),
        "channel_layout_id": np.asarray(trial.channel_layout_id),
        "sample_rate_hz": np.asarray(trial.sample_rate_hz),
        "timestamp_ms": np.asarray(trial.timestamp_ms, dtype=np.float64),
        "emg": np.asarray(trial.emg, dtype=np.float32),
        "source_label": np.asarray(trial.source_label).astype(str),
        "task_label": np.asarray(trial.task_label, dtype=np.int16),
        "canonical_label": np.asarray(trial.canonical_label, dtype=np.int16),
        "stable_mask": np.asarray(trial.stable_mask, dtype=np.bool_),
    }
    if trial.posture is not None:
        values["posture"] = np.asarray(trial.posture)
    if trial.imu is not None:
        values["imu"] = np.asarray(trial.imu, dtype=np.float32)
    np.savez_compressed(destination, **values)


def check_hla_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = load_hla_manifest(dataset_root)
    except BenchmarkDatasetError as exc:
        return {"status": "error", "errors": [str(exc)], "warnings": [], "trial_count": 0}
    paths = sorted((dataset_root / "trials").rglob("*.npz"))
    if not paths:
        errors.append("no HLA trial NPZ files found")
    identities: set[tuple[str, str]] = set()
    ontology = manifest.ontology_by_source()
    subjects: set[str] = set()
    sessions: set[tuple[str, str]] = set()
    layout_counts: dict[str, int] = {}
    for path in paths:
        try:
            trial = load_hla_trial(path, manifest)
        except BenchmarkDatasetError as exc:
            errors.append(str(exc))
            continue
        identity = (trial.channel_layout_id, trial.trial_id)
        if identity in identities:
            errors.append(f"duplicate layout/trial identity: {identity}")
        identities.add(identity)
        subjects.add(trial.subject_id)
        sessions.add((trial.subject_id, trial.session_id))
        layout_counts[trial.channel_layout_id] = layout_counts.get(trial.channel_layout_id, 0) + 1
        observed_rate = 1000.0 / float(np.median(np.diff(trial.timestamp_ms)))
        if not np.isclose(observed_rate, trial.sample_rate_hz, rtol=0.01):
            errors.append(
                f"{path} timestamps imply {observed_rate:.3f} Hz, declared {trial.sample_rate_hz:.3f} Hz"
            )
        unknown_sources = sorted(set(map(str, np.unique(trial.source_label))) - set(ontology))
        if unknown_sources:
            errors.append(f"{path} has source labels absent from ontology: {unknown_sources}")
        for source_value, canonical in zip(trial.source_label, trial.canonical_label):
            entry = ontology.get(str(source_value))
            expected = (
                int(entry.canonical_gesture)
                if entry is not None and entry.relation is OntologyRelation.EXACT
                else int(Gesture.UNKNOWN)
            )
            if int(canonical) != expected:
                errors.append(
                    f"{path} exposes source label {source_value!r} as canonical {int(canonical)}; "
                    f"expected {expected}"
                )
                break
        if manifest.has_imu != (trial.imu is not None):
            warnings.append(f"{path} IMU presence differs from dataset-level declaration")
    return {
        "status": "ok" if not errors else "error",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "trial_count": len(paths),
        "subject_count": len(subjects),
        "subject_session_count": len(sessions),
        "trials_by_layout": dict(sorted(layout_counts.items())),
    }
