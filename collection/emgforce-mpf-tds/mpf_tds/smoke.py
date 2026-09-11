from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import torch

from .features import MPFConfig, MultiBandMatrixPowerFeatures
from .model import MPFTDSConfig, MPFTDSNetwork


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data MPF+TDS forward/backward smoke test")
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.data_root / "discrete_gestures_corpus.csv").open(encoding="utf-8")))
    row = next(item for item in rows if item["split"] == "train")
    config = MPFConfig()
    extractor = MultiBandMatrixPowerFeatures(config)
    required_samples = config.left_context_samples + 1 + 399 * config.output_stride_samples
    with h5py.File(args.data_root / row["dataset"], "r") as handle:
        signal = torch.from_numpy(handle["data"]["emg"][:required_samples].T).float()[None]
    features = extractor(signal)
    model = MPFTDSNetwork(MPFTDSConfig(input_dim=config.output_dim))
    logits = model(features)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    gradients = sum(parameter.grad is not None for parameter in model.parameters())
    print(f"EMGFORCE_SMOKE_OK raw={tuple(signal.shape)} features={tuple(features.shape)} "
          f"logits={tuple(logits.shape)} loss={float(loss):.6f} gradients={gradients}")


if __name__ == "__main__":
    main()
