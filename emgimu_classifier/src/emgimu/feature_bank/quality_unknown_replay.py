"""Audit explicit Unknown decisions on frozen quality-corruption predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import late_fusion_decision
from .force_full_fusion import IDS
from .quality_corruption_study import SCENARIOS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(validation: Path, final: Path) -> dict:
    rows: list[dict] = []
    sources: dict[str, dict] = {}
    for phase, directory in (("validation", validation), ("final", final)):
        prediction_path = directory / "corruption_predictions.npz"
        manifest_path = directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["phase"] != phase or tuple(manifest["families"]) != IDS:
            raise ValueError(f"{phase} source manifest does not match frozen families")
        sources[phase] = {
            "run_id": directory.name,
            "predictions_sha256": _sha256(prediction_path),
            "manifest_sha256": _sha256(manifest_path),
        }
        with np.load(prediction_path, allow_pickle=False) as saved:
            users = saved["users"]
            trials = saved["trials"]
            labels = saved["labels"]
            if len(users) != len(trials) or len(labels) != len(trials):
                raise ValueError("source labels and trial identifiers are not aligned")
            for scenario in SCENARIOS:
                providers = {family: saved[f"{scenario}::{family}"] for family in IDS}
                q_min = saved[f"{scenario}::quality_min"]
                q_mean = saved[f"{scenario}::quality_mean"]
                quality = {
                    family: q_min if family in ("F0", "F2b_CSP") else q_mean
                    for family in IDS
                }
                decision = late_fusion_decision(
                    providers, IDS, np.ones(len(IDS)) / len(IDS),
                    tuple(str(index) for index in range(providers[IDS[0]].shape[1])),
                    quality,
                )
                np.testing.assert_allclose(
                    decision.probabilities, saved[f"{scenario}::full_quality_routing"],
                    rtol=1e-9, atol=1e-10,
                )
                rejected = decision.rejected
                rows.append({
                    "phase": phase,
                    "scenario": scenario,
                    "windows": len(trials),
                    "unknown_count": int(rejected.sum()),
                    "unknown_rate": float(rejected.mean()),
                    "unknown_users": sorted(set(map(str, users[rejected]))),
                    "unknown_trial_ids": sorted(set(map(str, trials[rejected]))),
                    "fallback_correct_count": int(np.sum(
                        np.argmax(decision.probabilities[rejected], axis=1)
                        == labels[rejected])),
                })
    return {
        "status": "frozen_probability_replay",
        "rule": "Unknown only when all available quality-weighted providers have zero weight; confidence threshold 0",
        "sources": sources,
        "rows": rows,
        "boundary": "Synthetic force corruptions and saved model probabilities only; no source refit, threshold selection, live-device accuracy, or precise quality calibration claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.validation, args.final)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"replayed {len(result['rows'])} frozen phase/scenario cells", flush=True)


if __name__ == "__main__":
    main()
