from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from ...state import Gesture
from ..benchmark import BenchmarkDatasetError
from ..hla_schema import (
    ChannelSpec,
    DatasetManifestV2,
    EMGTrial,
    OntologyEntry,
    OntologyRelation,
    check_hla_dataset,
    write_hla_manifest,
    write_hla_trial,
)
from .base import AdapterResult


SOURCE_URL = "https://physionet.org/content/grabmyo/1.1.0/"
SOURCE_RATE_HZ = 2048.0
ADAPTER_VERSION = "1.0.0"
EXPECTED_PHYSICAL_TRIALS = 3 * 43 * 17 * 7
RECORD_PATTERN = re.compile(
    r"^session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)\.hea$",
    re.IGNORECASE,
)
GAIN_PATTERN = re.compile(
    r"^(?P<gain>[-+0-9.eE]+)(?:\((?P<baseline>[-+]?\d+)\))?/(?P<unit>\S+)$"
)
GESTURES = (
    "lateral_prehension",
    "thumb_adduction",
    "thumb_little_opposition",
    "thumb_index_opposition",
    "thumb_index_extension",
    "thumb_little_extension",
    "index_middle_extension",
    "little_finger_extension",
    "index_finger_extension",
    "thumb_finger_extension",
    "wrist_extension",
    "wrist_flexion",
    "forearm_supination",
    "forearm_pronation",
    "hand_open",
    "hand_close",
    "rest",
)


def audit_grabmyo_source(
    source_root: str | Path, *, require_complete: bool = False,
) -> dict[str, object]:
    """Check record identity, data pairs, and formal-dataset completeness."""
    source = Path(source_root)
    if not source.is_dir():
        raise BenchmarkDatasetError(f"GRABMyo source directory does not exist: {source}")
    identities: list[tuple[int, int, int, int]] = []
    missing_data: list[str] = []
    for path in sorted(source.rglob("*.hea")):
        match = RECORD_PATTERN.match(path.name)
        if match is None:
            continue
        identity = tuple(map(int, match.groups()))
        session, participant, gesture, repetition = identity
        if not (
            1 <= session <= 3 and 1 <= participant <= 43
            and 1 <= gesture <= 17 and 1 <= repetition <= 7
        ):
            raise BenchmarkDatasetError(f"out-of-range GRABMyo record name: {path.name}")
        identities.append(identity)
        if not path.with_suffix(".dat").is_file():
            missing_data.append(path.relative_to(source).as_posix())
    if not identities:
        raise BenchmarkDatasetError(f"no GRABMyo WFDB headers found under {source}")
    if len(identities) != len(set(identities)):
        raise BenchmarkDatasetError("duplicate GRABMyo session/participant/gesture/trial records")
    if missing_data:
        raise BenchmarkDatasetError(
            f"{len(missing_data)} GRABMyo headers have no matching .dat file; "
            f"first missing pair: {missing_data[0]}"
        )
    complete = len(identities) == EXPECTED_PHYSICAL_TRIALS
    if require_complete and not complete:
        raise BenchmarkDatasetError(
            f"incomplete GRABMyo source: found {len(identities)} of "
            f"{EXPECTED_PHYSICAL_TRIALS} physical trials"
        )
    return {
        "status": "complete" if complete else "partial",
        "physical_trials": len(identities),
        "expected_physical_trials": EXPECTED_PHYSICAL_TRIALS,
        "subjects": len({identity[1] for identity in identities}),
        "sessions": len({identity[0] for identity in identities}),
        "gestures": len({identity[2] for identity in identities}),
        "repetitions": len({identity[3] for identity in identities}),
        "missing_data_pairs": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ontology() -> tuple[OntologyEntry, ...]:
    exact = {15: Gesture.OPEN, 16: Gesture.FIST, 17: Gesture.NEUTRAL}
    related = {4: "Thumb-index opposition is related to pinch but is not assumed identical."}
    rows: list[OntologyEntry] = []
    for gesture, name in enumerate(GESTURES, start=1):
        if gesture in exact:
            relation = OntologyRelation.EXACT
            canonical = exact[gesture]
            notes = ""
        elif gesture in related:
            relation = OntologyRelation.RELATED
            canonical = None
            notes = related[gesture]
        else:
            relation = OntologyRelation.DATASET_ONLY
            canonical = None
            notes = "No exact class exists in the current shared hand ontology."
        rows.append(OntologyEntry(str(gesture), name, gesture - 1, relation, canonical, notes))
    return tuple(rows)


def _ring_channels(prefix: str, count: int, first_index: int = 1) -> tuple[ChannelSpec, ...]:
    per_ring = count // 2
    return tuple(
        ChannelSpec(
            channel_id=f"{prefix.lower()}{index}",
            name=f"{prefix}{index}",
            ring_angle_deg=((index - first_index) % per_ring) * 360.0 / per_ring,
            anatomical_region=None,
            usable_band_hz=(10.0, 450.0),
        )
        for index in range(first_index, first_index + count)
    )


def _manifest() -> DatasetManifestV2:
    return DatasetManifestV2(
        dataset_id="grabmyo",
        dataset_version="1.1.0",
        adapter_version=ADAPTER_VERSION,
        source_url=SOURCE_URL,
        license_id="ODC-By-1.0",
        native_sample_rate_hz=SOURCE_RATE_HZ,
        has_imu=False,
        channel_layouts={
            "grabmyo_forearm_16": _ring_channels("F", 16),
            "grabmyo_wrist_12": _ring_channels("W", 12),
        },
        ontology=_ontology(),
        session_semantics="recording_day",
    )


def _load_wfdb_record(header_path: Path) -> tuple[np.ndarray, tuple[str, ...], float]:
    lines = [line.strip() for line in header_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise BenchmarkDatasetError(f"empty WFDB header: {header_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise BenchmarkDatasetError(f"invalid WFDB record line: {lines[0]}")
    signal_count, sample_rate, sample_count = int(first[1]), float(first[2]), int(first[3])
    if signal_count != 32 or not np.isclose(sample_rate, SOURCE_RATE_HZ):
        raise BenchmarkDatasetError(
            f"{header_path} expected 32 signals at {SOURCE_RATE_HZ:g} Hz; "
            f"got {signal_count} at {sample_rate:g} Hz"
        )
    if len(lines) != signal_count + 1:
        raise BenchmarkDatasetError(f"{header_path} has {len(lines) - 1} signal lines; expected {signal_count}")
    gains: list[float] = []
    baselines: list[int] = []
    names: list[str] = []
    data_name: str | None = None
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1] != "16":
            raise BenchmarkDatasetError(f"unsupported WFDB signal line: {line}")
        if data_name is None:
            data_name = fields[0]
        elif data_name != fields[0]:
            raise BenchmarkDatasetError("one-record multi-file WFDB input is not supported")
        gain_match = GAIN_PATTERN.match(fields[2])
        if gain_match is None:
            raise BenchmarkDatasetError(f"unsupported WFDB gain field: {fields[2]}")
        gain = float(gain_match.group("gain"))
        if gain <= 0:
            raise BenchmarkDatasetError("WFDB gains must be positive")
        gains.append(gain)
        baselines.append(int(gain_match.group("baseline") or 0))
        names.append(fields[-1])
    data_path = header_path.with_name(str(data_name))
    raw = np.fromfile(data_path, dtype="<i2")
    expected = sample_count * signal_count
    if len(raw) != expected:
        raise BenchmarkDatasetError(
            f"{data_path} contains {len(raw)} samples; header declares {expected}"
        )
    digital = raw.reshape(sample_count, signal_count).astype(np.float64)
    physical = (digital - np.asarray(baselines)) / np.asarray(gains)
    if not np.isfinite(physical).all():
        raise BenchmarkDatasetError(f"{data_path} converts to non-finite physical values")
    return physical.astype(np.float32), tuple(names), sample_rate


class GrabMyoAdapter:
    dataset_id = "grabmyo"

    def adapt(self, source_root: str | Path, output_root: str | Path) -> AdapterResult:
        source = Path(source_root)
        output = Path(output_root)
        audit_grabmyo_source(source)
        if output.exists():
            raise BenchmarkDatasetError(f"output already exists; refusing to overwrite: {output}")
        headers: list[tuple[Path, int, int, int, int]] = []
        for path in sorted(source.rglob("*.hea")):
            match = RECORD_PATTERN.match(path.name)
            if match:
                session, participant, gesture, repetition = map(int, match.groups())
                if not (1 <= session <= 3 and 1 <= participant <= 43 and 1 <= gesture <= 17 and 1 <= repetition <= 7):
                    raise BenchmarkDatasetError(f"out-of-range GRABMyo record name: {path.name}")
                headers.append((path, session, participant, gesture, repetition))
        if not headers:
            raise BenchmarkDatasetError(f"no GRABMyo WFDB headers found under {source}")
        identities = [(session, participant, gesture, repetition) for _, session, participant, gesture, repetition in headers]
        if len(identities) != len(set(identities)):
            raise BenchmarkDatasetError("duplicate GRABMyo session/participant/gesture/trial records")

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        manifest = _manifest()
        ontology = manifest.ontology_by_source()
        warnings: list[str] = []
        source_rows: list[dict[str, object]] = []
        counts: Counter[str] = Counter()
        try:
            write_hla_manifest(stage / "manifest.json", manifest)
            for header, session, participant, gesture, repetition in headers:
                emg, names, rate = _load_wfdb_record(header)
                name_to_index = {name: index for index, name in enumerate(names)}
                expected_names = {*(f"F{i}" for i in range(1, 17)), *(f"W{i}" for i in range(1, 13))}
                missing_names = sorted(expected_names - set(name_to_index))
                if missing_names:
                    raise BenchmarkDatasetError(f"{header} is missing channels {missing_names}")
                subject_id = f"p{participant:02d}"
                session_id = f"day{session}"
                trial_id = (
                    f"grabmyo-{subject_id}-{session_id}-g{gesture:02d}-r{repetition:02d}"
                )
                entry = ontology[str(gesture)]
                canonical = (
                    int(entry.canonical_gesture)
                    if entry.relation is OntologyRelation.EXACT else int(Gesture.UNKNOWN)
                )
                for layout_id, channel_names in (
                    ("grabmyo_forearm_16", tuple(f"F{i}" for i in range(1, 17))),
                    ("grabmyo_wrist_12", tuple(f"W{i}" for i in range(1, 13))),
                ):
                    selected = emg[:, [name_to_index[name] for name in channel_names]]
                    n = len(selected)
                    destination = (
                        stage / "trials" / subject_id / session_id / layout_id / f"{trial_id}.npz"
                    )
                    trial = EMGTrial(
                        path=destination,
                        dataset_id=self.dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        trial_id=trial_id,
                        channel_layout_id=layout_id,
                        sample_rate_hz=rate,
                        timestamp_ms=np.arange(n, dtype=np.float64) * 1000.0 / rate,
                        emg=selected,
                        source_label=np.asarray([str(gesture)] * n),
                        task_label=np.full(n, gesture - 1, dtype=np.int16),
                        canonical_label=np.full(n, canonical, dtype=np.int16),
                        stable_mask=np.ones(n, dtype=bool),
                    )
                    write_hla_trial(destination, trial)
                data_path = header.with_suffix(".dat")
                source_rows.append({
                    "record": header.relative_to(source).with_suffix("").as_posix(),
                    "header_sha256": _sha256(header),
                    "data_sha256": _sha256(data_path),
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "source_gesture": gesture,
                    "gesture": GESTURES[gesture - 1],
                    "repetition": repetition,
                    "samples": len(emg),
                })
                counts[GESTURES[gesture - 1]] += 1
            if len(headers) != EXPECTED_PHYSICAL_TRIALS:
                warnings.append(
                    f"found {len(headers)} of {EXPECTED_PHYSICAL_TRIALS} expected physical trials"
                )
            _write_json(stage / "splits.json", {
                "protocol_id": "grabmyo_recording_days_v1",
                "session_semantics": manifest.session_semantics,
                "development_session": "day1",
                "cross_session_evaluation": ["day2", "day3"],
                "subjects": sorted({row["subject_id"] for row in source_rows}),
                "sensor_views": ["grabmyo_forearm_16", "grabmyo_wrist_12"],
            })
            _write_json(stage / "reports" / "source_manifest.json", {
                "source_url": SOURCE_URL,
                "physical_trials": source_rows,
            })
            _write_json(stage / "reports" / "statistics.json", {
                "subjects": len({row["subject_id"] for row in source_rows}),
                "recording_days": len({(row["subject_id"], row["session_id"]) for row in source_rows}),
                "physical_trials": len(source_rows),
                "sensor_view_trials": len(source_rows) * 2,
                "physical_trials_by_gesture": dict(sorted(counts.items())),
                "warnings": warnings,
            })
            (stage / ".gitignore").write_text("trials/\n", encoding="utf-8")
            integrity = check_hla_dataset(stage)
            integrity["physical_trial_count"] = len(source_rows)
            integrity["warnings"] = sorted(set([*warnings, *integrity["warnings"]]))
            _write_json(stage / "reports" / "integrity.json", integrity)
            if integrity["status"] != "ok":
                raise BenchmarkDatasetError(
                    "converted GRABMyo dataset failed integrity check: " + "; ".join(integrity["errors"])
                )
            stage.replace(output)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return AdapterResult(
            self.dataset_id, output, output / "manifest.json", len(source_rows),
            tuple(sorted(set(warnings))),
        )
