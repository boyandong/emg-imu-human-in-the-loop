"""Emit a deterministic inference fingerprint for a Lightning gesture CKPT."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from generic_neuromotor_interface.lightning import DiscreteGesturesModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--output", help="Optional .npy path for numerical comparison")
    args = parser.parse_args()

    module = DiscreteGesturesModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu").eval()
    signal = torch.zeros(1, 8, args.samples, dtype=torch.float32)
    with torch.inference_mode():
        probabilities = torch.sigmoid(module(signal))[0].numpy().astype(
            np.float32, copy=False)
    if args.output:
        np.save(args.output, probabilities)
    print(json.dumps({
        "shape": list(probabilities.shape),
        "sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
        "minimum": float(probabilities.min()),
        "maximum": float(probabilities.max()),
        "head": probabilities.ravel()[:8].tolist(),
    }))


if __name__ == "__main__":
    main()
