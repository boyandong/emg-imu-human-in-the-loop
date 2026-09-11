from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import LABELS, load_split
from .features import MPFConfig
from .metrics import evaluate_events, evaluate_paper_fnr
from .model import MPFTDSConfig, MPFTDSNetwork


def paper_learning_rate(epoch: int, peak_learning_rate: float = 1e-3) -> float:
    if epoch < 1:
        raise ValueError("epoch 必须从 1 开始")
    if epoch <= 5:
        return peak_learning_rate * epoch / 5.0
    return peak_learning_rate if epoch <= 25 else 5e-4


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []; targets = []
    model.eval()
    with torch.inference_mode():
        for features, target in loader:
            probabilities.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def train(data_root: Path, output_root: Path, epochs: int, batch_size: int,
          learning_rate: float, device_name: str,
          sample_rate_hz: int = 200) -> Path:
    torch.manual_seed(7); np.random.seed(7)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    feature_cfg = MPFConfig.for_sample_rate(sample_rate_hz)
    train_data = load_split(data_root, "train", feature_config=feature_cfg)
    val_data = load_split(data_root, "val", feature_config=feature_cfg)
    test_data = load_split(data_root, "test", feature_config=feature_cfg)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    cfg = MPFTDSConfig(input_dim=feature_cfg.output_dim)
    model = MPFTDSNetwork(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    run_dir = output_root / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_fnr = float("inf"); best_event_f1 = 0.0; paper_threshold = 0.35
    for epoch in range(1, epochs + 1):
        current_lr = paper_learning_rate(epoch, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        model.train(); losses = []
        for features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device)), target.to(device)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss))
        val_probs, val_targets = _predict(model, val_loader, device)
        paper_metrics = evaluate_paper_fnr(val_probs, val_targets, LABELS, paper_threshold)
        flat_probs = val_probs.reshape(-1, len(LABELS)); flat_targets = val_targets.reshape(-1, len(LABELS))
        event_metrics = evaluate_events(
            flat_probs, flat_targets, LABELS, (paper_threshold,) * len(LABELS))
        print(f"epoch={epoch} lr={current_lr:.7f} loss={np.mean(losses):.6f} "
              f"val_mean_fnr={paper_metrics.mean_fnr:.6f} "
              f"val_event_macro_f1={event_metrics.macro_f1:.6f}", flush=True)
        if paper_metrics.mean_fnr < best_fnr:
            best_fnr = paper_metrics.mean_fnr; best_event_f1 = event_metrics.macro_f1
            torch.save({"state_dict": model.state_dict(), "feature_config": feature_cfg.to_dict(),
                        "network_config": cfg.to_dict(), "implementation_revision": 3,
                        "epoch": epoch}, run_dir / "best.pt")
    payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    test_probs, test_targets = _predict(model, test_loader, device)
    test_paper = evaluate_paper_fnr(test_probs, test_targets, LABELS, paper_threshold)
    test_metrics = evaluate_events(
        test_probs.reshape(-1, len(LABELS)), test_targets.reshape(-1, len(LABELS)),
        LABELS, (paper_threshold,) * len(LABELS))
    bundle = run_dir / "bundle"; bundle.mkdir()
    artifact = bundle / "model.pt"; torch.save(payload, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    display_names = {"index_press": "食指按下", "index_release": "食指释放",
                     "middle_press": "中指按下", "middle_release": "中指释放"}
    labels = {"model_outputs": list(LABELS), "display_names": display_names}
    (bundle / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    # Compute the causal context from the configured sample rate instead of
    # assuming the legacy 2 kHz / 40-sample stride preset.
    tds_receptive_frames = 1 + cfg.blocks_per_scale * (cfg.kernel_size - 1) * sum(
        2 ** scale for scale in range(cfg.num_scales))
    receptive_field_samples = (
        feature_cfg.left_context_samples + 1
        + (tds_receptive_frames - 1) * feature_cfg.output_stride_samples
    )
    preprocessing = {"preprocessing_version": "meta_8ch_v1", "realtime_buffer_seconds": 8.0,
                     "realtime_model_window_seconds": 6.0, "realtime_fixed_lag_seconds": 0.25,
                     "online_event_threshold": paper_threshold,
                     "class_thresholds": {name: paper_threshold for name in LABELS},
                     "debounce_seconds": 0.05, "minimum_hold_seconds": 0.5}
    (bundle / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    manifest = {"bundle_version": 2, "model_id": f"mpf_tds_{run_dir.parent.name.replace('-', '')}_{run_dir.name.replace('-', '')}",
                "algorithm_id": "personal_mpf_tds_v1", "runtime_backend": "mpf_tds",
                "artifact": {"filename": "model.pt", "format": "mpf_tds_state_dict", "sha256": digest,
                             "size_bytes": artifact.stat().st_size},
                "signal": {"sample_rate_hz": feature_cfg.sample_rate_hz,
                           "input_channels": feature_cfg.channels},
                "preprocessing": preprocessing,
                 "network": {**cfg.to_dict(), "input_channels": feature_cfg.channels,
                             "output_channels": len(LABELS),
                             "sample_rate_hz": feature_cfg.sample_rate_hz,
                             "left_context_samples": feature_cfg.left_context_samples,
                             "receptive_field_samples": receptive_field_samples,
                             "receptive_field_seconds": receptive_field_samples / feature_cfg.sample_rate_hz,
                             "stride": feature_cfg.output_stride_samples},
                "features": {**feature_cfg.to_dict(), "implementation_revision": 3},
                "decoder": {"threshold": paper_threshold, "matching": "needleman_wunsch",
                            "metric": "mean_per_class_fnr"},
                "training": {"checkpoint_metric": "validation_mean_fnr", "val_mean_fnr": best_fnr,
                             "val_event_macro_f1": best_event_f1,
                             "best_epoch": int(payload.get("epoch", 0)),
                             "test_mean_fnr": test_paper.mean_fnr,
                             "test_per_class_fnr": test_paper.per_class_fnr,
                             "test_event_macro_f1": test_metrics.macro_f1, "test_event_micro_f1": test_metrics.micro_f1,
                             "test_false_positives_per_minute": test_metrics.false_positives_per_minute,
                             "test_hold_success_rate": test_metrics.hold_success_rate,
                             "per_class": test_metrics.per_class},
                "labels_file": "labels.json", "preprocessing_file": "preprocessing.json"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EMGFORCE_BUNDLE={bundle}", flush=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path); parser.add_argument("--output-root", default="logs", type=Path)
    parser.add_argument("--epochs", type=int, default=300); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate-hz", type=int, choices=(200, 2000), default=200)
    args = parser.parse_args(); train(
        args.data_root, args.output_root, args.epochs, args.batch_size,
        args.learning_rate, args.device, args.sample_rate_hz)


if __name__ == "__main__":
    main()
