from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .hla_features import extract_hla_feature_tokens
from .hla_schema import load_hla_manifest, load_hla_trial
from .hla_windows import HLA_MAIN_PROTOCOL, window_trial


@dataclass(frozen=True, slots=True)
class ClassicalRunConfig:
    protocol_id: str = HLA_MAIN_PROTOCOL.protocol_id
    feature_groups: tuple[str, ...] = ("G0",)
    seed: int = 42
    c: float = 1.0
    gamma: str = "scale"
    neutral_label: int = 0
    maximum_subjects: int | None = None
    maximum_windows_per_trial: int | None = None


def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def _select_windows(count: int, maximum: int | None) -> np.ndarray:
    if maximum is None or count <= maximum:
        return np.arange(count)
    return np.unique(np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64))


def _load_rows(
    dataset_root: Path,
    config: ClassicalRunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = load_hla_manifest(dataset_root)
    features: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    trials: list[str] = []
    timestamps: list[float] = []
    for path in sorted((dataset_root / "trials").rglob("*.npz")):
        trial = load_hla_trial(path, manifest)
        windows = window_trial(trial, HLA_MAIN_PROTOCOL, label_space="task")
        for index in _select_windows(len(windows), config.maximum_windows_per_trial):
            token = extract_hla_feature_tokens(windows.emg[index], config.feature_groups)
            features.append(token.reshape(-1))
            labels.append(int(windows.labels[index]))
            subjects.append(trial.subject_id)
            trials.append(trial.trial_id)
            timestamps.append(float(windows.start_timestamp_ms[index]))
    if not features:
        raise ValueError("dataset produced no eligible windows")
    dimensions = {len(row) for row in features}
    if len(dimensions) != 1:
        raise ValueError(
            "classical specialist requires one fixed channel layout; use one sensor view at a time"
        )
    return (
        np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int16),
        np.asarray(subjects), np.asarray(trials), np.asarray(timestamps, dtype=np.float64),
    )


def _confidence(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    ordered = np.sort(values, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    return (1.0 / (1.0 + np.exp(-margin))).astype(np.float64)


def run_loso_svm(
    dataset_root: str | Path,
    output_root: str | Path,
    config: ClassicalRunConfig = ClassicalRunConfig(),
) -> Path:
    """Run a trial-safe LOSO RBF-SVM specialist and emit auditable artifacts."""
    started = time.time()
    dataset = Path(dataset_root)
    output = Path(output_root)
    if output.exists():
        raise ValueError(f"output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)
    x, y, subject, trial_id, timestamp = _load_rows(dataset, config)
    all_subjects = sorted(set(subject.tolist()))
    selected_subjects = (
        all_subjects if config.maximum_subjects is None
        else all_subjects[:config.maximum_subjects]
    )
    if len(selected_subjects) < 2:
        raise ValueError("LOSO requires at least two selected subjects")
    selected = np.isin(subject, selected_subjects)
    x, y, subject, trial_id, timestamp = (
        value[selected] for value in (x, y, subject, trial_id, timestamp)
    )
    class_labels = sorted(map(int, np.unique(y)))
    active_labels = [label for label in class_labels if label != config.neutral_label]
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    model_hashes: dict[str, str] = {}
    models = output / "models"
    models.mkdir()
    for target in selected_subjects:
        test = subject == target
        train = ~test
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(C=config.c, gamma=config.gamma, kernel="rbf", class_weight="balanced")),
        ])
        pipeline.fit(x[train], y[train])
        predicted = pipeline.predict(x[test]).astype(np.int16)
        confidence = _confidence(pipeline.decision_function(x[test]))
        actual = y[test]
        metric_rows.append({
            "target_subject": target,
            "train_windows": int(train.sum()),
            "evaluation_windows": int(test.sum()),
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=class_labels, average="macro", zero_division=0)),
            "active_macro_f1": float(f1_score(actual, predicted, labels=active_labels, average="macro", zero_division=0)),
        })
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=class_labels, zero_division=0,
        )
        for label, p, r, score, count in zip(class_labels, precision, recall, f1, support):
            class_rows.append({
                "target_subject": target, "task_label": label, "precision": float(p),
                "recall": float(r), "f1": float(score), "support_windows": int(count),
            })
        matrix = confusion_matrix(actual, predicted, labels=class_labels)
        for row_index, actual_label in enumerate(class_labels):
            for column_index, predicted_label in enumerate(class_labels):
                confusion_rows.append({
                    "target_subject": target, "actual_label": actual_label,
                    "predicted_label": predicted_label, "count": int(matrix[row_index, column_index]),
                })
        test_indices = np.flatnonzero(test)
        for index, truth, guess, score in zip(test_indices, actual, predicted, confidence):
            prediction_rows.append({
                "target_subject": target, "trial_id": str(trial_id[index]),
                "start_timestamp_ms": float(timestamp[index]), "actual_label": int(truth),
                "predicted_label": int(guess), "confidence": float(score),
            })
        ordering = np.argsort(-confidence, kind="stable")
        for coverage in (0.1, 0.25, 0.5, 0.75, 1.0):
            count = max(1, int(np.ceil(len(ordering) * coverage)))
            kept = ordering[:count]
            risk_rows.append({
                "target_subject": target, "coverage": coverage,
                "risk": 1.0 - float(accuracy_score(actual[kept], predicted[kept])),
            })
        model_path = models / f"{target}.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(pipeline, stream, protocol=pickle.HIGHEST_PROTOCOL)
        model_hashes[target] = _sha256(model_path)

    fields = {
        "metrics.csv": ["target_subject", "train_windows", "evaluation_windows", "accuracy", "macro_f1", "active_macro_f1"],
        "per_class.csv": ["target_subject", "task_label", "precision", "recall", "f1", "support_windows"],
        "predictions.csv": ["target_subject", "trial_id", "start_timestamp_ms", "actual_label", "predicted_label", "confidence"],
        "confusion_matrix.csv": ["target_subject", "actual_label", "predicted_label", "count"],
        "risk_coverage.csv": ["target_subject", "coverage", "risk"],
    }
    rows = {
        "metrics.csv": metric_rows, "per_class.csv": class_rows,
        "predictions.csv": prediction_rows, "confusion_matrix.csv": confusion_rows,
        "risk_coverage.csv": risk_rows,
    }
    for name, columns in fields.items():
        _write_csv(output / name, columns, rows[name])
    _write_csv(output / "calibration_curve.csv", ["status"], [{"status": "not_applicable_zero_shot_loso"}])
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8",
    )
    (output / "model.sha256").write_text(
        "".join(f"{digest}  models/{name}.pkl\n" for name, digest in sorted(model_hashes.items())),
        encoding="utf-8",
    )
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "elapsed_seconds": time.time() - started,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    manifest = load_hla_manifest(dataset)
    representation = "R0" if config.feature_groups == ("G0",) else "R1-core"
    source_commit, source_dirty = _git_provenance()
    summary = {
        "status": "complete", "run_kind": f"B0_LOSO_RBF_SVM_{representation}",
        "dataset_id": manifest.dataset_id, "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
        "subjects": selected_subjects, "source_commit": source_commit,
        "source_worktree_dirty": source_dirty,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in metric_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in metric_rows])),
        "mean_active_macro_f1": float(np.mean([row["active_macro_f1"] for row in metric_rows])),
        "limitations": [
            "Window predictions overlap; subjects, sessions, or trials are the inferential units.",
            "This is a dataset-specific specialist baseline, not a cross-hardware universal model.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "test_log.txt").write_text("Run unit tests separately and record the command/output here for formal runs.\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an HLA trial-safe LOSO RBF-SVM baseline")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--features", choices=("R0", "R1-core"), default="R0")
    parser.add_argument("--maximum-subjects", type=int)
    parser.add_argument("--maximum-windows-per-trial", type=int)
    args = parser.parse_args(argv)
    groups = ("G0",) if args.features == "R0" else ("G0", "G5")
    run_loso_svm(args.dataset_root, args.output_root, ClassicalRunConfig(
        feature_groups=groups, maximum_subjects=args.maximum_subjects,
        maximum_windows_per_trial=args.maximum_windows_per_trial,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
