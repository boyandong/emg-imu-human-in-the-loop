from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...signal import polyphase_resample
from ...state import Gesture
from ..benchmark import BenchmarkDatasetError, check_benchmark_dataset
from .base import AdapterResult


SOURCE_URL = "https://github.com/pulp-bio/unibo-inail-semg-dataset"
SOURCE_RATE_HZ = 500.0
TARGET_RATE_HZ = 200.0
ADAPTER_VERSION = "1.0.0"
# The official archive uses ``user1_day1_posture1.mat``.  Accept the earlier
# underscored spelling as well so locally renamed copies remain readable.
FILENAME_PATTERN = re.compile(
    r"^(?:unibo_)?user_?(\d+)_day_?(\d+)_posture_?(\d+)\.mat$",
    re.IGNORECASE,
)

SOURCE_LABELS: dict[int, dict[str, object]] = {
    1: {"name": "rest", "hand_label": int(Gesture.NEUTRAL), "eligible": True},
    2: {"name": "power_grip", "hand_label": int(Gesture.FIST), "eligible": True},
    3: {"name": "two_finger_pinch", "hand_label": int(Gesture.PINCH), "eligible": True},
    4: {"name": "three_finger_pinch", "hand_label": int(Gesture.UNKNOWN), "eligible": False},
    5: {"name": "pointing_index", "hand_label": int(Gesture.UNKNOWN), "eligible": False},
    6: {"name": "open_hand", "hand_label": int(Gesture.OPEN), "eligible": True},
}

POSTURE_LABELS = {
    1: "proximal",
    2: "distal",
    3: "distal_palm_down",
    4: "distal_arm_45deg_up",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_vector(value: np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if len(array) != length:
        raise BenchmarkDatasetError(f"{name} length {len(array)} does not match EMG length {length}")
    if not np.isfinite(array).all() or not np.allclose(array, np.round(array)):
        raise BenchmarkDatasetError(f"{name} must contain finite integer values")
    return np.round(array).astype(np.int16)


def _load_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
        values = loadmat(path, variable_names=("emg", "label", "relabel", "gestureCounter"))
    except Exception as exc:
        raise BenchmarkDatasetError(f"cannot read {path}: {exc}") from exc
    required = {"emg", "label", "relabel", "gestureCounter"}
    missing = sorted(required - set(values))
    if missing:
        raise BenchmarkDatasetError(f"{path} missing MAT fields: {', '.join(missing)}")
    emg = np.asarray(values["emg"], dtype=np.float64)
    if emg.ndim != 2 or emg.shape[1] != 4 or len(emg) < 2:
        raise BenchmarkDatasetError(f"{path} EMG must have shape [samples>=2,4]")
    if not np.isfinite(emg).all():
        raise BenchmarkDatasetError(f"{path} EMG contains NaN or Inf")
    label = _integer_vector(values["label"], "label", len(emg))
    relabel = _integer_vector(values["relabel"], "relabel", len(emg))
    counter = _integer_vector(values["gestureCounter"], "gestureCounter", len(emg))
    for name, array in (("label", label), ("relabel", relabel)):
        invalid = sorted(set(map(int, np.unique(array))) - set(SOURCE_LABELS))
        if invalid:
            raise BenchmarkDatasetError(f"{path} has invalid {name} values: {invalid}")
    if np.any(counter < 0):
        raise BenchmarkDatasetError(f"{path} gestureCounter contains negative values")
    return emg, label, relabel, counter


def _contiguous_runs(counter: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.r_[True, counter[1:] != counter[:-1]])
    ends = np.r_[starts[1:], len(counter)]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def _source_gesture(label: np.ndarray, counter_value: int, path: Path) -> int:
    active = sorted(set(map(int, np.unique(label))) - {1})
    if counter_value == 0:
        if active:
            raise BenchmarkDatasetError(f"{path} counter=0 segment contains active labels {active}")
        return 1
    if len(active) != 1:
        raise BenchmarkDatasetError(
            f"{path} repetition counter={counter_value} must contain one active gesture; got {active}"
        )
    return active[0]


def _resample_labels(values: np.ndarray, target_length: int) -> np.ndarray:
    source_indices = np.rint(
        np.arange(target_length, dtype=np.float64) * SOURCE_RATE_HZ / TARGET_RATE_HZ
    ).astype(np.int64)
    return values[np.clip(source_indices, 0, len(values) - 1)]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _capabilities_markdown() -> str:
    return """# UniBo-INAIL benchmark 能力边界

## 可以验证

- Neutral / Fist / 2-finger Pinch / Open 的 H 四分类。
- 同一受试者跨日泛化，以及七名受试者上的总体趋势。
- 四种静态手臂姿态下的鲁棒性和跨姿态变化。
- 与通道数无关的时域特征、时序建模、归一化和域适配方法。

## 不能验证

- 手臂运动方向 D、运动阶段、动作提示到稳定输出的端到端延迟。
- D+H 组合状态、A、M、C 或 EMG-IMU 融合；数据集没有 IMU。
- 八通道环形腕带的循环移位鲁棒性；UniBo 是四块命名肌肉电极且不是环形布局。
- 当前八通道原始输入头本身。不得把四通道插值、复制或补零成假八通道。

## 主 benchmark 标签

主任务只使用 Rest、Power grip、2-finger pinch 和 Open hand。3-finger pinch 与
Pointing index 保留在标准化文件中，但整段标记为 benchmark_eligible=false，
不参与四分类，也不贡献 Neutral 尾段。
"""


class UniBoInailAdapter:
    dataset_id = "unibo-inail"

    def adapt(self, source_root: str | Path, output_root: str | Path) -> AdapterResult:
        source = Path(source_root)
        output = Path(output_root)
        if not source.is_dir():
            raise BenchmarkDatasetError(f"source directory does not exist: {source}")
        if output.exists():
            raise BenchmarkDatasetError(f"output already exists; refusing to overwrite: {output}")
        files: list[tuple[Path, int, int, int]] = []
        for path in sorted(source.rglob("*.mat")):
            match = FILENAME_PATTERN.match(path.name)
            if match:
                subject, day, posture = map(int, match.groups())
                if not (1 <= subject <= 7 and 1 <= day <= 8 and 1 <= posture <= 4):
                    raise BenchmarkDatasetError(f"out-of-range UniBo filename: {path.name}")
                files.append((path, subject, day, posture))
        if not files:
            raise BenchmarkDatasetError(f"no UniBo MAT files found under {source}")
        keys = [(subject, day, posture) for _, subject, day, posture in files]
        if len(keys) != len(set(keys)):
            raise BenchmarkDatasetError("duplicate subject/day/posture MAT files")

        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        warnings: list[str] = []
        source_records: list[dict[str, object]] = []
        trial_records: list[dict[str, object]] = []
        class_samples: Counter[str] = Counter()
        total_samples = 0
        unmapped_samples = 0
        posture_trials: Counter[str] = Counter()
        session_trials: Counter[str] = Counter()
        repetitions: dict[tuple[int, int, int, int], int] = defaultdict(int)
        excluded_counter_zero_runs = 0
        excluded_counter_zero_samples = 0
        excluded_counter_zero_active_samples = 0
        try:
            for path, subject, day, posture in files:
                emg, label, relabel, counter = _load_source(path)
                file_trial_count = 0
                for start, end in _contiguous_runs(counter):
                    counter_value = int(counter[start])
                    # The official files use counter=0 outside numbered
                    # repetitions.  Some recordings retain active labels in
                    # these boundary regions, so exclude them rather than
                    # silently treating them as either rest or a trial.
                    if counter_value == 0:
                        excluded_counter_zero_runs += 1
                        excluded_counter_zero_samples += end - start
                        excluded_counter_zero_active_samples += int(
                            np.count_nonzero(label[start:end] != 1)
                        )
                        continue
                    gesture = _source_gesture(label[start:end], counter_value, path)
                    if counter_value > 0:
                        repetitions[(subject, day, posture, gesture)] += 1
                    subject_id = f"u{subject:02d}"
                    session_id = f"d{day:02d}"
                    trial_id = (
                        f"unibo-{subject_id}-{session_id}-p{posture:02d}-"
                        f"g{gesture:02d}-r{counter_value:02d}"
                    )
                    source_info = SOURCE_LABELS[gesture]
                    eligible = bool(source_info["eligible"])
                    emg_target = polyphase_resample(
                        emg[start:end], SOURCE_RATE_HZ, TARGET_RATE_HZ,
                    ).astype(np.float32)
                    target_length = len(emg_target)
                    source_duration = (end - start) / SOURCE_RATE_HZ
                    target_duration = target_length / TARGET_RATE_HZ
                    if abs(source_duration - target_duration) > 1.0 / TARGET_RATE_HZ + 1e-12:
                        raise BenchmarkDatasetError(
                            f"{path} repetition duration changed by more than one target sample"
                        )
                    target_label = _resample_labels(label[start:end], target_length).astype(np.int16)
                    target_relabel = _resample_labels(relabel[start:end], target_length).astype(np.int16)
                    hand_label = np.asarray(
                        [SOURCE_LABELS[int(value)]["hand_label"] for value in target_relabel],
                        dtype=np.int16,
                    )
                    stable_mask = (
                        eligible
                        & (target_label == target_relabel)
                        & (hand_label != int(Gesture.UNKNOWN))
                    )
                    timestamp_ms = (
                        np.arange(target_length, dtype=np.float64) * 1000.0 / TARGET_RATE_HZ
                    )
                    relative = (
                        Path("trials") / subject_id / session_id / f"p{posture:02d}" /
                        f"{trial_id}.npz"
                    )
                    destination = stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        destination,
                        subject_id=np.asarray(subject_id),
                        session_id=np.asarray(session_id),
                        trial_id=np.asarray(trial_id),
                        timestamp_ms=timestamp_ms,
                        emg=emg_target,
                        hand_label=hand_label,
                        posture_label=np.asarray(posture, dtype=np.int16),
                        source_label=target_label,
                        source_relabel=target_relabel,
                        stable_mask=stable_mask,
                        benchmark_eligible=np.asarray(eligible, dtype=np.bool_),
                        source_gesture=np.asarray(gesture, dtype=np.int16),
                    )
                    file_trial_count += 1
                    total_samples += target_length
                    if not eligible:
                        unmapped_samples += target_length
                    posture_trials[str(posture)] += 1
                    session_trials[f"{subject_id}/{session_id}"] += 1
                    if eligible:
                        stable_values = hand_label[stable_mask]
                        for value, count in zip(*np.unique(stable_values, return_counts=True)):
                            class_samples[Gesture(int(value)).name] += int(count)
                    trial_records.append({
                        "trial_id": trial_id,
                        "path": relative.as_posix(),
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "posture_label": posture,
                        "source_gesture": gesture,
                        "benchmark_eligible": eligible,
                        "samples": target_length,
                    })
                source_records.append({
                    "path": path.relative_to(source).as_posix(),
                    "sha256": _sha256(path),
                    "subject_id": f"u{subject:02d}",
                    "session_id": f"d{day:02d}",
                    "posture_label": posture,
                    "source_samples": len(emg),
                    "trial_count": file_trial_count,
                })

            expected = {(s, d, p) for s in range(1, 8) for d in range(1, 9) for p in range(1, 5)}
            missing_sources = sorted(expected - set(keys))
            if missing_sources:
                warnings.append(f"missing {len(missing_sources)} of 224 subject/day/posture sources")
            if excluded_counter_zero_active_samples:
                warnings.append(
                    "excluded "
                    f"{excluded_counter_zero_active_samples} actively labelled samples from "
                    "counter-zero regions outside numbered repetitions"
                )
            for subject, day, posture in sorted(set(keys)):
                for gesture in range(2, 7):
                    count = repetitions.get((subject, day, posture, gesture), 0)
                    if not 9 <= count <= 16:
                        warnings.append(
                            f"u{subject:02d}/d{day:02d}/p{posture:02d}/g{gesture:02d} "
                            f"has {count} repetitions; expected 9-16"
                        )

            groups = {
                "train": [f"u{s:02d}/d{d:02d}" for s in range(1, 8) for d in range(1, 6)],
                "validation": [f"u{s:02d}/d06" for s in range(1, 8)],
                "test": [f"u{s:02d}/d{d:02d}" for s in range(1, 8) for d in (7, 8)],
            }
            label_map = {
                str(key): value for key, value in SOURCE_LABELS.items()
            }
            _write_json(stage / "label_map.json", {
                "dataset_id": self.dataset_id,
                "source_label_field": "relabel",
                "source_labels": label_map,
                "posture_labels": {str(k): v for k, v in POSTURE_LABELS.items()},
            })
            _write_json(stage / "splits.json", {
                "protocol": "subject-dependent chronological cross-day",
                "unit": "subject_id/session_id",
                "train_days": [1, 2, 3, 4, 5],
                "validation_days": [6],
                "test_days": [7, 8],
                "groups": groups,
            })
            manifest = {
                "schema_version": "1.0",
                "dataset_id": self.dataset_id,
                "adapter_version": ADAPTER_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": SOURCE_URL,
                "license_id": "LGPL-2.1",
                "source_sample_rate_hz": SOURCE_RATE_HZ,
                "target_sample_rate_hz": TARGET_RATE_HZ,
                "emg_channels": 4,
                "channel_names": [
                    "extensor_carpi_ulnaris",
                    "extensor_digitorum_communis",
                    "flexor_carpi_radialis",
                    "flexor_carpi_ulnaris",
                ],
                "channel_layout": "named_muscles_non_circular",
                "emg_units": "source_native_unspecified",
                "has_imu": False,
                "subject_count": len({record["subject_id"] for record in source_records}),
                "subject_day_session_count": len(session_trials),
                "subject_day_posture_source_count": len(source_records),
                "trial_count": len(trial_records),
                "eligible_trial_count": sum(bool(row["benchmark_eligible"]) for row in trial_records),
                "source_files": source_records,
            }
            _write_json(stage / "manifest.json", manifest)
            _write_json(stage / "reports" / "statistics.json", {
                "subjects": manifest["subject_count"],
                "subject_day_sessions": manifest["subject_day_session_count"],
                "subject_day_posture_sources": manifest["subject_day_posture_source_count"],
                "trials": manifest["trial_count"],
                "eligible_trials": manifest["eligible_trial_count"],
                "stable_samples_by_hand_label": dict(sorted(class_samples.items())),
                "stable_duration_seconds_by_hand_label": {
                    key: value / TARGET_RATE_HZ for key, value in sorted(class_samples.items())
                },
                "total_resampled_samples": total_samples,
                "total_resampled_duration_seconds": total_samples / TARGET_RATE_HZ,
                "unmapped_samples": unmapped_samples,
                "unmapped_sample_fraction": (
                    unmapped_samples / total_samples if total_samples else 0.0
                ),
                "trials_by_posture": dict(sorted(posture_trials.items())),
                "trials_by_subject_session": dict(sorted(session_trials.items())),
                "unmapped_active_gestures": [4, 5],
                "missing_source_count": len(missing_sources),
                "excluded_counter_zero_runs": excluded_counter_zero_runs,
                "excluded_counter_zero_samples": excluded_counter_zero_samples,
                "excluded_counter_zero_active_samples": excluded_counter_zero_active_samples,
            })
            (stage / "reports" / "CAPABILITIES.md").write_text(
                _capabilities_markdown(), encoding="utf-8",
            )
            # Converted signal arrays are intentionally local-only.  Metadata
            # and reports can still be versioned as a benchmark baseline.
            (stage / ".gitignore").write_text("trials/\n", encoding="utf-8")
            precheck = check_benchmark_dataset(stage)
            integrity = {
                **precheck,
                "warnings": sorted(set([*warnings, *precheck["warnings"]])),
                "expected_subjects": 7,
                "expected_days_per_subject": 8,
                "expected_postures_per_day": 4,
                "expected_sources": 224,
                "discovered_sources": len(source_records),
                "missing_sources": [
                    {"subject": s, "day": d, "posture": p}
                    for s, d, p in missing_sources
                ],
            }
            _write_json(stage / "reports" / "integrity.json", integrity)
            if precheck["status"] != "ok":
                raise BenchmarkDatasetError(
                    "converted benchmark failed integrity check: " + "; ".join(precheck["errors"])
                )
            stage.replace(output)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return AdapterResult(
            dataset_id=self.dataset_id,
            output_root=output,
            manifest_path=output / "manifest.json",
            trial_count=len(trial_records),
            warnings=tuple(sorted(set(warnings))),
        )
