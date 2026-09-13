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


SOURCE_URL = "https://github.com/rajkundu/myoband"
SOURCE_RATE_HZ = 200.0
CHANNEL_COUNT = 8
ADAPTER_VERSION = "1.0.0"
BLOCKS = ("training0", "Test0", "Test1")
GESTURES = (
    "neutral",
    "radial_deviation",
    "wrist_flexion",
    "ulnar_deviation",
    "wrist_extension",
    "hand_close",
    "hand_open",
)
FILE_PATTERN = re.compile(r"^classe_(\d+)\.dat$", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"^(Female|Male)(\d+)$", re.IGNORECASE)


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
    entries: list[OntologyEntry] = []
    exact = {
        0: Gesture.NEUTRAL,
        5: Gesture.FIST,
        6: Gesture.OPEN,
    }
    for index, name in enumerate(GESTURES):
        entries.append(OntologyEntry(
            str(index), name, index,
            OntologyRelation.EXACT if index in exact else OntologyRelation.DATASET_ONLY,
            exact.get(index),
            "" if index in exact else "No exact class exists in the current shared hand ontology.",
        ))
    return tuple(entries)


def _manifest() -> DatasetManifestV2:
    return DatasetManifestV2(
        dataset_id="myo-armband-evaluation",
        dataset_version="rajkundu-myoband-master",
        adapter_version=ADAPTER_VERSION,
        source_url=SOURCE_URL,
        license_id="GPL-3.0-repository",
        native_sample_rate_hz=SOURCE_RATE_HZ,
        has_imu=False,
        channel_layouts={
            "myo_ring_8": tuple(
                ChannelSpec(
                    channel_id=f"myo-{index + 1}",
                    name=f"myo_channel_{index + 1}",
                    ring_angle_deg=index * 45.0,
                    anatomical_region=None,
                    usable_band_hz=(0.0, 100.0),
                )
                for index in range(CHANNEL_COUNT)
            )
        },
        ontology=_ontology(),
        session_semantics="within_session_acquisition_block",
    )


class MyoArmbandEvaluationAdapter:
    dataset_id = "myo-armband-evaluation"

    def adapt(self, source_root: str | Path, output_root: str | Path) -> AdapterResult:
        source = Path(source_root)
        evaluation = source / "EvaluationDataset" if (source / "EvaluationDataset").is_dir() else source
        output = Path(output_root)
        if not evaluation.is_dir():
            raise BenchmarkDatasetError(f"Myo EvaluationDataset does not exist: {evaluation}")
        if output.exists():
            raise BenchmarkDatasetError(f"output already exists; refusing to overwrite: {output}")
        subjects = sorted(
            (path for path in evaluation.iterdir() if path.is_dir() and SUBJECT_PATTERN.match(path.name)),
            key=lambda path: (path.name.lower().startswith("male"), int(SUBJECT_PATTERN.match(path.name).group(2))),
        )
        if not subjects:
            raise BenchmarkDatasetError(f"no FemaleN/MaleN subject folders found under {evaluation}")

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        warnings: list[str] = []
        sources: list[dict[str, object]] = []
        counts: Counter[str] = Counter()
        trial_count = 0
        manifest = _manifest()
        canonical = {
            entry.source_value: (
                int(entry.canonical_gesture)
                if entry.relation is OntologyRelation.EXACT else int(Gesture.UNKNOWN)
            )
            for entry in manifest.ontology
        }
        try:
            write_hla_manifest(stage / "manifest.json", manifest)
            for subject_path in subjects:
                match = SUBJECT_PATTERN.match(subject_path.name)
                sex, number = match.groups()
                subject_id = f"{sex.lower()}{int(number):02d}"
                for block_name in BLOCKS:
                    block_path = subject_path / block_name
                    if not block_path.is_dir():
                        warnings.append(f"{subject_path.name} is missing {block_name}")
                        continue
                    discovered: dict[int, Path] = {}
                    for path in block_path.iterdir():
                        file_match = FILE_PATTERN.match(path.name)
                        if file_match:
                            discovered[int(file_match.group(1))] = path
                    missing = sorted(set(range(28)) - set(discovered))
                    if missing:
                        warnings.append(
                            f"{subject_path.name}/{block_name} is missing class files {missing}"
                        )
                    for file_index, path in sorted(discovered.items()):
                        if file_index < 0:
                            raise BenchmarkDatasetError(f"negative class file index: {path}")
                        raw = np.fromfile(path, dtype=np.int16)
                        if len(raw) % CHANNEL_COUNT:
                            raise BenchmarkDatasetError(
                                f"{path} has {len(raw)} int16 values, not divisible by {CHANNEL_COUNT}"
                            )
                        emg = raw.astype(np.float32).reshape(-1, CHANNEL_COUNT)
                        if len(emg) < 3:
                            raise BenchmarkDatasetError(f"{path} contains fewer than three samples")
                        gesture = file_index % len(GESTURES)
                        repetition = file_index // len(GESTURES)
                        trial_id = (
                            f"myo-{subject_id}-{block_name.lower()}-"
                            f"g{gesture:02d}-r{repetition:02d}"
                        )
                        n = len(emg)
                        source_values = np.asarray([str(gesture)] * n)
                        trial = EMGTrial(
                            path=stage / "trials" / subject_id / block_name.lower() / f"{trial_id}.npz",
                            dataset_id=self.dataset_id,
                            subject_id=subject_id,
                            session_id=block_name.lower(),
                            trial_id=trial_id,
                            channel_layout_id="myo_ring_8",
                            sample_rate_hz=SOURCE_RATE_HZ,
                            timestamp_ms=np.arange(n, dtype=np.float64) * 1000.0 / SOURCE_RATE_HZ,
                            emg=emg,
                            source_label=source_values,
                            task_label=np.full(n, gesture, dtype=np.int16),
                            canonical_label=np.full(n, canonical[str(gesture)], dtype=np.int16),
                            stable_mask=np.ones(n, dtype=bool),
                        )
                        write_hla_trial(trial.path, trial)
                        sources.append({
                            "path": path.relative_to(evaluation).as_posix(),
                            "sha256": _sha256(path),
                            "subject_id": subject_id,
                            "block": block_name.lower(),
                            "task_label": gesture,
                            "gesture": GESTURES[gesture],
                            "repetition": repetition,
                            "samples": n,
                        })
                        counts[GESTURES[gesture]] += 1
                        trial_count += 1
            _write_json(stage / "splits.json", {
                "protocol_id": "myo_official_evaluation_blocks_v1",
                "session_semantics": manifest.session_semantics,
                "calibration_candidate_blocks": ["training0"],
                "evaluation_blocks": ["test0", "test1"],
                "subjects": sorted({row["subject_id"] for row in sources}),
                "note": "Blocks are not claimed to be separate days or redon sessions.",
            })
            _write_json(stage / "reports" / "source_manifest.json", {
                "source_root_name": evaluation.name,
                "source_url": SOURCE_URL,
                "files": sources,
            })
            _write_json(stage / "reports" / "statistics.json", {
                "subjects": len({row["subject_id"] for row in sources}),
                "trials": trial_count,
                "trials_by_gesture": dict(sorted(counts.items())),
                "blocks": [name.lower() for name in BLOCKS],
                "warnings": sorted(set(warnings)),
            })
            (stage / ".gitignore").write_text("trials/\n", encoding="utf-8")
            integrity = check_hla_dataset(stage)
            integrity["warnings"] = sorted(set([*warnings, *integrity["warnings"],
                "Verify that the repository GPL-3.0 license covers intended dataset redistribution.",
            ]))
            _write_json(stage / "reports" / "integrity.json", integrity)
            if integrity["status"] != "ok":
                raise BenchmarkDatasetError(
                    "converted Myo dataset failed integrity check: " + "; ".join(integrity["errors"])
                )
            stage.replace(output)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return AdapterResult(
            self.dataset_id, output, output / "manifest.json", trial_count,
            tuple(sorted(set(warnings))),
        )
