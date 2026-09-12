from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..baseline import BaselinePredictor, TemperatureScaler, _softmax
from .benchmark import BenchmarkDatasetError, check_benchmark_dataset
from .unibo_baseline import (
    HAND_CLASSES,
    _git_metadata,
    _json_dump,
    _sha256,
    _write_confusion_matrices,
    _write_predictions,
    _write_risk_coverage,
    evaluate_unibo_probabilities,
    hierarchical_segment_weights,
    load_unibo_raw_windows,
)

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # pragma: no cover - actionable dependency error
    torch = None
    functional = None
    nn = None


@dataclass(frozen=True, slots=True)
class UniBoTcnConfig:
    window_samples: int = 40
    hop_samples: int = 40
    max_train_windows_per_trial_label: int = 6
    hidden_channels: int = 32
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    minimum_validation_coverage: float = 0.90
    minimum_class_coverage: float = 0.75
    random_seed: int = 42

    def validate(self) -> None:
        if self.window_samples < 3 or self.hop_samples < 1:
            raise ValueError("window_samples must be >=3 and hop_samples must be positive")
        if self.max_train_windows_per_trial_label < 1:
            raise ValueError("max_train_windows_per_trial_label must be positive")
        if self.hidden_channels < 4 or not self.dilations:
            raise ValueError("hidden_channels must be >=4 and dilations cannot be empty")
        if any(value < 1 for value in self.dilations):
            raise ValueError("all dilations must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive")


if nn is not None:
    class CausalDepthwiseBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.left_padding = 2 * dilation
            self.depthwise = nn.Conv1d(
                channels, channels, kernel_size=3, dilation=dilation, groups=channels,
            )
            self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
            self.batch_norm = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            update = self.depthwise(functional.pad(values, (self.left_padding, 0)))
            update = self.pointwise(functional.gelu(update))
            update = self.dropout(self.batch_norm(update))
            return functional.gelu(values + update)


    class UniBoFourChannelTCN(nn.Module):
        """Small causal temporal baseline for four named, non-circular EMG channels."""

        def __init__(self, hidden_channels: int, dilations: tuple[int, ...], dropout: float) -> None:
            super().__init__()
            self.hidden_channels = int(hidden_channels)
            self.dilations = tuple(map(int, dilations))
            self.dropout = float(dropout)
            self.input_projection = nn.Conv1d(4, self.hidden_channels, kernel_size=1)
            self.blocks = nn.Sequential(*[
                CausalDepthwiseBlock(self.hidden_channels, dilation, self.dropout)
                for dilation in self.dilations
            ])
            self.output = nn.Sequential(
                nn.Linear(self.hidden_channels * 2, self.hidden_channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, len(HAND_CLASSES)),
            )

        def forward(self, emg: "torch.Tensor") -> "torch.Tensor":
            if emg.ndim != 3 or emg.shape[-1] != 4:
                raise ValueError("UniBo TCN input must be [batch,time,4]")
            encoded = self.blocks(functional.gelu(self.input_projection(emg.transpose(1, 2))))
            pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=2)], dim=1)
            return self.output(pooled)

else:  # pragma: no cover
    class UniBoFourChannelTCN:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for the UniBo TCN baseline")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("install the project's neural extra before running the TCN baseline")


def _normalize(values: "torch.Tensor", global_scale: float) -> "torch.Tensor":
    scaled = values / max(float(global_scale), 1e-8)
    return torch.sign(scaled) * torch.log1p(torch.abs(scaled))


def _batches(length: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator):
    indices = np.arange(length)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start:start + batch_size]


def _class_weights(labels: np.ndarray, sample_weights: np.ndarray, device: str) -> "torch.Tensor":
    counts = np.bincount(labels, weights=sample_weights, minlength=len(HAND_CLASSES))
    values = counts.sum() / np.maximum(counts, 1e-12)
    values /= values.mean()
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _collect_logits(
    model: "UniBoFourChannelTCN",
    emg: np.ndarray,
    *,
    global_scale: float,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for indices in _batches(len(emg), batch_size, shuffle=False, rng=rng):
            batch = torch.as_tensor(emg[indices], dtype=torch.float32, device=device)
            rows.append(model(_normalize(batch, global_scale)).cpu().numpy())
    return np.concatenate(rows)


def _weighted_validation_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    class_weights: np.ndarray,
) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    losses = -log_probabilities[np.arange(len(labels)), labels] * class_weights[labels]
    return float(np.average(losses, weights=weights))


def _parameter_count(model: "UniBoFourChannelTCN") -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run_tcn_baseline(
    dataset_root: str | Path,
    output_root: str | Path,
    run_id: str,
    *,
    config: UniBoTcnConfig | None = None,
    evaluate_test: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    _require_torch()
    cfg = config or UniBoTcnConfig()
    cfg.validate()
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the installed PyTorch build has no CUDA support")
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    final_model = output / "models" / run_id
    final_result = output / "results" / run_id
    if final_model.exists() or final_result.exists():
        raise FileExistsError(f"run_id already exists and will not be overwritten: {run_id}")
    integrity = check_benchmark_dataset(
        dataset,
        splits_to_check=None if evaluate_test else ("train", "validation"),
    )
    if integrity["status"] != "ok":
        raise BenchmarkDatasetError(f"benchmark integrity failed: {integrity['errors']}")

    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output))
    model_stage = stage / "model"
    result_stage = stage / "result"
    model_stage.mkdir()
    result_stage.mkdir()
    started = time.time()
    try:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)
        rng = np.random.default_rng(cfg.random_seed)
        print(json.dumps({"event": "loading_train_validation", "device": selected_device}), flush=True)
        loaded = load_unibo_raw_windows(
            dataset, ("train", "validation"), window_samples=cfg.window_samples,
            hop_samples=cfg.hop_samples,
            max_windows_per_trial_label={
                "train": cfg.max_train_windows_per_trial_label,
                "validation": None,
            },
        )
        train = loaded["train"]
        validation = loaded["validation"]
        if set(np.unique(train.labels)) != set(HAND_CLASSES) or set(np.unique(validation.labels)) != set(HAND_CLASSES):
            raise RuntimeError("train and validation must both contain all four hand classes")
        train_weights = hierarchical_segment_weights(train)
        validation_weights = hierarchical_segment_weights(validation)
        global_scale = float(np.percentile(np.abs(train.emg), 95))
        if not np.isfinite(global_scale) or global_scale <= 0:
            raise RuntimeError("could not derive a positive train-only global EMG scale")
        model = UniBoFourChannelTCN(
            cfg.hidden_channels, cfg.dilations, cfg.dropout,
        ).to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        class_weight_tensor = _class_weights(train.labels, train_weights, selected_device)
        class_weight_array = class_weight_tensor.detach().cpu().numpy()
        print(json.dumps({
            "event": "windows_ready", "train_windows": len(train),
            "validation_windows": len(validation), "train_trials": len(np.unique(train.trial_id)),
            "validation_trials": len(np.unique(validation.trial_id)),
            "global_scale": global_scale, "parameters": _parameter_count(model),
        }), flush=True)

        from sklearn.metrics import f1_score

        best_score = -float("inf")
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, "torch.Tensor"] | None = None
        history: list[dict[str, float | int]] = []
        stale = 0
        for epoch in range(cfg.max_epochs):
            model.train()
            weighted_loss_sum = 0.0
            weight_sum = 0.0
            for indices in _batches(len(train), cfg.batch_size, shuffle=True, rng=rng):
                emg = torch.as_tensor(train.emg[indices], dtype=torch.float32, device=selected_device)
                labels = torch.as_tensor(train.labels[indices], dtype=torch.long, device=selected_device)
                weights = torch.as_tensor(train_weights[indices], dtype=torch.float32, device=selected_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(_normalize(emg, global_scale))
                losses = functional.cross_entropy(
                    logits, labels, weight=class_weight_tensor, reduction="none",
                )
                loss = (losses * weights).sum() / weights.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                weighted_loss_sum += float((losses.detach() * weights).sum())
                weight_sum += float(weights.sum())
            validation_logits = _collect_logits(
                model, validation.emg, global_scale=global_scale,
                batch_size=cfg.batch_size, device=selected_device,
            )
            validation_prediction = validation_logits.argmax(axis=1)
            validation_f1 = float(f1_score(
                validation.labels, validation_prediction, labels=HAND_CLASSES,
                average="macro", sample_weight=validation_weights, zero_division=0,
            ))
            validation_loss = _weighted_validation_loss(
                validation_logits, validation.labels, validation_weights, class_weight_array,
            )
            row = {
                "epoch": epoch + 1,
                "train_loss": weighted_loss_sum / max(weight_sum, 1e-12),
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_f1,
            }
            history.append(row)
            print(json.dumps({"event": "epoch_complete", **row}), flush=True)
            improved = (
                validation_f1 > best_score + 1e-6 or
                (abs(validation_f1 - best_score) <= 1e-6 and validation_loss < best_loss)
            )
            if improved:
                best_score = validation_f1
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        if best_state is None:
            raise RuntimeError("TCN training produced no checkpoint")
        model.load_state_dict(best_state)
        model.to(selected_device).eval()
        validation_logits = _collect_logits(
            model, validation.emg, global_scale=global_scale,
            batch_size=cfg.batch_size, device=selected_device,
        )
        temperature = TemperatureScaler().fit(
            validation_logits, validation.labels, HAND_CLASSES, validation_weights,
        ).temperature
        validation_probabilities = _softmax(validation_logits, temperature)
        validation_indices = validation_probabilities.argmax(axis=1)
        threshold = BaselinePredictor._select_threshold(
            validation_probabilities, validation_indices, HAND_CLASSES[validation_indices],
            validation.labels, sample_weight=validation_weights,
            min_coverage=cfg.minimum_validation_coverage,
            min_class_coverage=cfg.minimum_class_coverage,
        )
        print(json.dumps({
            "event": "model_selected", "best_epoch": best_epoch,
            "validation_macro_f1": best_score, "temperature": temperature,
            "threshold": threshold,
        }), flush=True)

        configuration = {
            **asdict(cfg),
            "dilations": list(cfg.dilations),
            "run_id": run_id,
            "dataset_root": str(dataset),
            "model_kind": "unibo_four_channel_causal_tcn",
            "normalization": "train_only_global_p95_then_signed_log1p_v1",
            "global_scale": global_scale,
            "sample_rate_hz": 200,
            "window_ms": cfg.window_samples * 5,
            "hop_ms": cfg.hop_samples * 5,
            "split_protocol": "Day 1-5 train; Day 6 validation; Day 7-8 test",
            "weighting": "equal_subject_day_then_trial_then_truth_segment_then_window_v1",
            "device": selected_device,
            "test_evaluated": bool(evaluate_test),
        }
        artifact = {
            "artifact_format_version": 1,
            "model_kind": "unibo_four_channel_causal_tcn",
            "state_dict": best_state,
            "hidden_channels": cfg.hidden_channels,
            "dilations": cfg.dilations,
            "dropout": cfg.dropout,
            "classes": HAND_CLASSES,
            "global_scale": global_scale,
            "temperature": temperature,
            "threshold": threshold,
            "best_epoch": best_epoch,
            "config": configuration,
        }
        model_path = model_stage / "model.pt"
        torch.save(artifact, model_path)
        model_sha256 = _sha256(model_path)
        validation_metrics, validation_arrays = evaluate_unibo_probabilities(
            "validation", validation, validation_probabilities, threshold,
        )
        metrics: dict[str, Any] = {"validation": validation_metrics}
        prediction_rows: list[tuple[str, Any, dict[str, np.ndarray]]] = [
            ("validation", validation, validation_arrays),
        ]

        if evaluate_test:
            print(json.dumps({
                "event": "model_frozen_opening_test", "model_sha256": model_sha256,
            }), flush=True)
            test = load_unibo_raw_windows(
                dataset, ("test",), window_samples=cfg.window_samples,
                hop_samples=cfg.hop_samples, max_windows_per_trial_label={"test": None},
            )["test"]
            test_logits = _collect_logits(
                model, test.emg, global_scale=global_scale,
                batch_size=cfg.batch_size, device=selected_device,
            )
            test_probabilities = _softmax(test_logits, temperature)
            test_metrics, test_arrays = evaluate_unibo_probabilities(
                "test", test, test_probabilities, threshold,
            )
            metrics["test"] = test_metrics
            prediction_rows.append(("test", test, test_arrays))

        _json_dump(result_stage / "config.json", configuration)
        _json_dump(result_stage / "metrics.json", metrics)
        with (result_stage / "training_history.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        _write_predictions(result_stage / "predictions.csv", prediction_rows)
        _write_confusion_matrices(result_stage / "confusion_matrix.csv", metrics)
        _write_risk_coverage(result_stage / "risk_coverage.csv", metrics)
        source_file = Path(__file__).resolve()
        repository_root = source_file.parents[4]
        run_manifest = {
            "run_id": run_id,
            "status": "complete",
            "model_sha256": model_sha256,
            "benchmark_manifest_sha256": _sha256(dataset / "manifest.json"),
            "splits_sha256": _sha256(dataset / "splits.json"),
            "runner_sha256": _sha256(source_file),
            "elapsed_seconds": time.time() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "dependencies": {
                "numpy": np.__version__, "torch": torch.__version__,
            },
            "git": _git_metadata(repository_root),
            "integrity": integrity,
            "parameter_count": _parameter_count(model),
            "best_epoch": best_epoch,
            "test_opened_after_model_hash": bool(evaluate_test),
        }
        _json_dump(result_stage / "run_manifest.json", run_manifest)
        final_model.parent.mkdir(parents=True, exist_ok=True)
        final_result.parent.mkdir(parents=True, exist_ok=True)
        model_stage.replace(final_model)
        result_stage.replace(final_result)
        shutil.rmtree(stage, ignore_errors=True)
        result = {
            "run_id": run_id,
            "model": str(final_model / "model.pt"),
            "results": str(final_result),
            "model_sha256": model_sha256,
            "best_epoch": best_epoch,
            "validation_macro_f1": best_score,
            "temperature": temperature,
            "threshold": threshold,
            "metrics": metrics,
        }
        print(json.dumps({
            "event": "complete", "run_id": run_id, "results": str(final_result),
        }), flush=True)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the strict UniBo four-class H causal TCN baseline")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-samples", type=int, default=40)
    parser.add_argument("--hop-samples", type=int, default=40)
    parser.add_argument("--max-train-windows-per-trial-label", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = UniBoTcnConfig(
        window_samples=arguments.window_samples,
        hop_samples=arguments.hop_samples,
        max_train_windows_per_trial_label=arguments.max_train_windows_per_trial_label,
        hidden_channels=arguments.hidden_channels,
        batch_size=arguments.batch_size,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
    )
    result = run_tcn_baseline(
        arguments.dataset, arguments.output_root, arguments.run_id,
        config=config, evaluate_test=arguments.evaluate_test, device=arguments.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
