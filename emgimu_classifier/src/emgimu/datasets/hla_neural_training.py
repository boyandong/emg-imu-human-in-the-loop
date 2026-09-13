from __future__ import annotations

import hashlib
import csv
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from .hla_features import extract_hla_feature_tokens
from .hla_hardware import DEFAULT_STREAM_RATES_HZ, build_multirate_window, channel_metadata_matrix
from .hla_neural import HLAEncoderConfig, HLAMultiDatasetModel, _require_torch
from .hla_schema import load_hla_manifest, load_hla_trial
from .hla_windows import HLA_MAIN_PROTOCOL, window_trial

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None


@dataclass(frozen=True, slots=True)
class HLANeuralRunConfig:
    representation: str
    sensor_view: str
    target_subject: str
    validation_subject: str | None = None
    seed: int = 42
    maximum_subjects: int | None = None
    maximum_windows_per_trial: int | None = 10
    batch_size: int = 128
    maximum_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    neutral_label: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.representation not in {"R0", "R1-core", "R2", "R3", "R2-wide"}:
            raise ValueError("unknown HLA representation")
        if not self.sensor_view or not self.target_subject:
            raise ValueError("sensor_view and target_subject are required")
        if self.maximum_subjects is not None and self.maximum_subjects < 3:
            raise ValueError("at least three subjects are needed for train/validation/test")
        if self.maximum_windows_per_trial is not None and self.maximum_windows_per_trial < 1:
            raise ValueError("maximum_windows_per_trial must be positive or None")
        if self.batch_size < 1 or self.maximum_epochs < 1 or self.patience < 1:
            raise ValueError("training counts must be positive")


@dataclass(slots=True)
class _Examples:
    streams: dict[int, np.ndarray]
    availability: np.ndarray
    features: np.ndarray | None
    metadata: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    trials: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def _select(count: int, maximum: int | None) -> np.ndarray:
    if maximum is None or count <= maximum:
        return np.arange(count)
    return np.unique(np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64))


def _collect_examples(dataset_root: Path, config: HLANeuralRunConfig) -> tuple[_Examples, int]:
    manifest = load_hla_manifest(dataset_root)
    if config.sensor_view not in manifest.channel_layouts:
        raise ValueError(f"unknown sensor view {config.sensor_view!r}")
    trials_root = dataset_root / "trials"
    all_paths: list[tuple[Path, str]] = []
    for path in sorted(trials_root.rglob("*.npz")):
        relative = path.relative_to(trials_root)
        # V2 adapters store trials as subject/session/layout/file. Filtering from
        # the auditable path index avoids decompressing every large trial twice.
        if len(relative.parts) >= 4 and relative.parts[-2] == config.sensor_view:
            all_paths.append((path, relative.parts[0]))
        elif len(relative.parts) < 4:
            # The schema itself does not require the canonical adapter layout.
            # Preserve compatibility for small/custom V2 datasets at the cost of
            # one metadata read; official large adapters take the fast path above.
            trial = load_hla_trial(path, manifest)
            if trial.channel_layout_id == config.sensor_view:
                all_paths.append((path, trial.subject_id))
    subjects = sorted({subject for _, subject in all_paths})
    if config.target_subject not in subjects:
        raise ValueError(f"target subject {config.target_subject!r} is absent")
    if config.maximum_subjects is not None:
        kept = [config.target_subject] + [s for s in subjects if s != config.target_subject]
        subjects = sorted(kept[:config.maximum_subjects])
        if config.target_subject not in subjects:
            raise AssertionError("target subject was lost during subject selection")
    subject_set = set(subjects)
    uses_raw = config.representation in {"R2", "R3", "R2-wide"}
    raw_rows: dict[int, list[np.ndarray]] = {rate: [] for rate in DEFAULT_STREAM_RATES_HZ}
    availability_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    subject_rows: list[str] = []
    trial_rows: list[str] = []
    timestamp_rows: list[float] = []
    uses_features = config.representation in {"R0", "R1-core", "R3"}
    groups = ("G0",) if config.representation == "R0" else ("G0", "G5")
    metadata = channel_metadata_matrix(manifest.channel_layouts[config.sensor_view])
    stream_shapes: dict[int, tuple[int, int]] = {}
    for path, subject in all_paths:
        if subject not in subject_set:
            continue
        trial = load_hla_trial(path, manifest)
        if trial.subject_id != subject or trial.channel_layout_id != config.sensor_view:
            raise ValueError(f"trial metadata disagrees with its V2 path index: {path}")
        windows = window_trial(trial)
        for index in _select(len(windows), config.maximum_windows_per_trial):
            emg = windows.emg[index]
            if uses_raw:
                multi = build_multirate_window(emg, trial.sample_rate_hz)
                for rate, stream in multi.streams.items():
                    shape = tuple(stream.shape)
                    previous = stream_shapes.setdefault(rate, shape)
                    if previous != shape:
                        raise ValueError(f"{rate} Hz stream shape varies within one sensor view")
                for rate in DEFAULT_STREAM_RATES_HZ:
                    if rate in multi.streams:
                        raw_rows[rate].append(multi.streams[rate].T)
                availability_rows.append(multi.availability.astype(np.float32))
            else:
                availability_rows.append(np.zeros(len(DEFAULT_STREAM_RATES_HZ), dtype=np.float32))
            if uses_features:
                feature_rows.append(extract_hla_feature_tokens(emg, groups))
            label_rows.append(int(windows.labels[index]))
            subject_rows.append(trial.subject_id)
            trial_rows.append(trial.trial_id)
            timestamp_rows.append(float(windows.start_timestamp_ms[index]))
    if not label_rows:
        raise ValueError("no eligible windows found")
    availability = np.stack(availability_rows)
    streams: dict[int, np.ndarray] = {}
    for rate_index, rate in enumerate(DEFAULT_STREAM_RATES_HZ):
        if np.all(availability[:, rate_index]):
            streams[rate] = np.stack(raw_rows[rate]).astype(np.float32)
        elif np.any(availability[:, rate_index]):
            raise ValueError("one training run cannot mix availability within a sensor view")
    task_labels = [entry.task_label for entry in manifest.ontology if entry.task_label is not None]
    class_count = max(task_labels) + 1
    return _Examples(
        streams=streams,
        availability=availability,
        features=np.stack(feature_rows).astype(np.float32) if uses_features else None,
        metadata=metadata,
        labels=np.asarray(label_rows, dtype=np.int64),
        subjects=np.asarray(subject_rows),
        trials=np.asarray(trial_rows),
        timestamps=np.asarray(timestamp_rows, dtype=np.float64),
    ), class_count


def _fit_normalization(examples: _Examples, train: np.ndarray) -> dict[str, Any]:
    scales: dict[int, float] = {}
    for rate, values in examples.streams.items():
        scale = float(np.percentile(np.abs(values[train]), 95))
        scales[rate] = max(scale, 1e-8)
    feature_mean = feature_std = None
    if examples.features is not None:
        tokens = examples.features[train].reshape(-1, examples.features.shape[-1]).astype(np.float64)
        feature_mean = tokens.mean(axis=0)
        feature_std = np.maximum(tokens.std(axis=0), 1e-6)
    return {"stream_scales": scales, "feature_mean": feature_mean, "feature_std": feature_std}


def _batch(
    examples: _Examples,
    indices: np.ndarray,
    normalization: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    assert torch is not None
    streams = {
        rate: torch.as_tensor(values[indices] / normalization["stream_scales"][rate], device=device)
        for rate, values in examples.streams.items()
    }
    feature_tokens = None
    if examples.features is not None:
        normalized = (
            examples.features[indices] - normalization["feature_mean"]
        ) / normalization["feature_std"]
        feature_tokens = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    batch = len(indices)
    channels = len(examples.metadata)
    return {
        "streams": streams,
        "stream_availability": torch.as_tensor(examples.availability[indices], device=device),
        "metadata": torch.as_tensor(examples.metadata, device=device).unsqueeze(0).expand(batch, -1, -1),
        "quality": torch.ones((batch, channels), device=device),
        "channel_mask": torch.ones((batch, channels), dtype=torch.bool, device=device),
        "feature_tokens": feature_tokens,
    }


def _predict(
    model: HLAMultiDatasetModel,
    dataset_id: str,
    examples: _Examples,
    indices: np.ndarray,
    normalization: Mapping[str, Any],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    assert torch is not None
    model.eval()
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            logits = model(dataset_id, **_batch(examples, selected, normalization, device))["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            confidences.append(probabilities.max(dim=-1).values.cpu().numpy())
            predictions.append(probabilities.argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(confidences)


def _metrics(actual: np.ndarray, predicted: np.ndarray, neutral_label: int) -> dict[str, float]:
    labels = sorted(map(int, np.unique(actual)))
    active = [label for label in labels if label != neutral_label]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "active_macro_f1": float(f1_score(actual, predicted, labels=active, average="macro", zero_division=0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_provenance() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_neural_subject_fold(
    dataset_root: str | Path,
    output_root: str | Path,
    config: HLANeuralRunConfig,
) -> Path:
    """Train one source-only-selected LOSO fold for an HLA representation."""
    _require_torch()
    assert torch is not None and F is not None
    started = time.time()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(config.seed)
    dataset = Path(dataset_root)
    output = Path(output_root)
    if output.exists():
        raise ValueError(f"output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    examples, class_count = _collect_examples(dataset, config)
    source_subjects = sorted(set(examples.subjects.tolist()) - {config.target_subject})
    validation_subject = config.validation_subject or source_subjects[-1]
    if validation_subject not in source_subjects:
        raise ValueError("validation subject must be a non-target source subject")
    test_mask = examples.subjects == config.target_subject
    validation_mask = examples.subjects == validation_subject
    train_mask = ~(test_mask | validation_mask)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("train, validation, and evaluation roles must all be nonempty")
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)
    normalization = _fit_normalization(examples, train_mask)
    feature_dim = None if examples.features is None else examples.features.shape[-1]
    encoder_config = HLAEncoderConfig(
        representation=config.representation,
        metadata_dim=examples.metadata.shape[-1],
        feature_dim=feature_dim,
    )
    manifest = load_hla_manifest(dataset)
    model = HLAMultiDatasetModel({manifest.dataset_id: class_count}, encoder_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    counts = np.bincount(examples.labels[train_mask], minlength=class_count)
    weights = np.divide(
        counts.sum(), class_count * counts,
        out=np.zeros(class_count, dtype=np.float64), where=counts > 0,
    )
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=config.device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    stale = 0
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        losses: list[float] = []
        for start in range(0, len(shuffled), config.batch_size):
            selected = shuffled[start:start + config.batch_size]
            target = torch.as_tensor(examples.labels[selected], device=config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(manifest.dataset_id, **_batch(
                examples, selected, normalization, config.device,
            ))["logits"]
            loss = F.cross_entropy(logits, target, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction, _ = _predict(
            model, manifest.dataset_id, examples, validation_indices, normalization,
            config.batch_size, config.device,
        )
        validation = _metrics(
            examples.labels[validation_indices], validation_prediction, config.neutral_label,
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **validation})
        score = validation["active_macro_f1"]
        if score > best_score + 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predicted, confidence = _predict(
        model, manifest.dataset_id, examples, test_indices, normalization,
        config.batch_size, config.device,
    )
    actual = examples.labels[test_indices]
    metrics = _metrics(actual, predicted, config.neutral_label)
    checkpoint = output / "model.pt"
    torch.save({
        "model_state": best_state,
        "encoder_config": encoder_config.to_dict(),
        "dataset_classes": {manifest.dataset_id: class_count},
        "normalization": normalization,
        "run_config": asdict(config),
    }, checkpoint)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    per_class = precision_recall_fscore_support(
        actual, predicted, labels=list(range(class_count)), zero_division=0,
    )
    report = {
        "status": "complete", "dataset_id": manifest.dataset_id,
        "representation": config.representation, "target_subject": config.target_subject,
        "validation_subject": validation_subject,
        "train_subjects": sorted(set(examples.subjects[train_mask].tolist())),
        "window_counts": {
            "train": int(train_mask.sum()), "validation": int(validation_mask.sum()),
            "evaluation": int(test_mask.sum()),
        },
        "metrics": metrics, "best_validation_active_macro_f1": best_score,
        "epochs_completed": len(history), "model_sha256": _sha256(checkpoint),
        "elapsed_seconds": time.time() - started,
        "per_class": [
            {"task_label": label, "precision": float(per_class[0][label]),
             "recall": float(per_class[1][label]), "f1": float(per_class[2][label]),
             "support": int(per_class[3][label])}
            for label in range(class_count)
        ],
    }
    source_commit, source_dirty = _git_provenance()
    report["source_commit"] = source_commit
    report["source_worktree_dirty"] = source_dirty
    report["dataset_manifest_sha256"] = _sha256(dataset / "manifest.json")
    (output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prediction_rows = [
        {"trial_id": str(examples.trials[index]),
         "start_timestamp_ms": float(examples.timestamps[index]),
         "actual_label": int(truth), "predicted_label": int(guess),
         "confidence": float(score)}
        for index, truth, guess, score in zip(test_indices, actual, predicted, confidence)
    ]
    _write_csv(
        output / "predictions.csv",
        ("trial_id", "start_timestamp_ms", "actual_label", "predicted_label", "confidence"),
        prediction_rows,
    )
    _write_csv(
        output / "metrics.csv",
        ("split", "accuracy", "macro_f1", "active_macro_f1"),
        [{"split": "evaluation", **metrics}],
    )
    _write_csv(
        output / "per_class.csv",
        ("task_label", "precision", "recall", "f1", "support"),
        report["per_class"],
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(class_count)))
    confusion_rows = [
        {"actual_label": actual_label, "predicted_label": predicted_label,
         "count": int(matrix[actual_label, predicted_label])}
        for actual_label in range(class_count) for predicted_label in range(class_count)
    ]
    _write_csv(
        output / "confusion_matrix.csv", ("actual_label", "predicted_label", "count"),
        confusion_rows,
    )
    ordering = np.argsort(-confidence, kind="stable")
    risk_rows: list[dict[str, float]] = []
    for coverage in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, int(np.ceil(len(ordering) * coverage)))
        kept = ordering[:count]
        risk_rows.append({
            "coverage": coverage,
            "risk": 1.0 - float(accuracy_score(actual[kept], predicted[kept])),
        })
    _write_csv(output / "risk_coverage.csv", ("coverage", "risk"), risk_rows)
    _write_csv(
        output / "calibration_curve.csv", ("status",),
        [{"status": "not_applicable_zero_shot_loso"}],
    )
    (output / "model.sha256").write_text(
        f"{report['model_sha256']}  model.pt\n", encoding="utf-8",
    )
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() else None
        ),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8",
    )
    (output / "test_log.txt").write_text(
        "Formal runs must replace this line with the exact pre-run test command and result.\n",
        encoding="utf-8",
    )
    return output
