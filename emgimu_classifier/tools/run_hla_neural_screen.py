from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAMultiDatasetModel, hla_parameter_count
from emgimu.datasets.hla_neural_training import (
    HLANeuralRunConfig,
    prepare_hla_screen_examples,
    run_neural_subject_fold,
)


REPRESENTATIONS = ("R0", "R1-core", "R2", "R3", "R2-wide")


def _complete(path: Path) -> bool:
    manifest = path / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError):
        return False


def _bootstrap(
    values: np.ndarray, seed: int, iterations: int = 2000,
) -> list[float | None]:
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.mean(values[rng.integers(0, len(values), len(values))])
        for _ in range(iterations)
    ])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a matched-subject HLA representation screen")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sensor-view", required=True)
    parser.add_argument("--targets", required=True, help="comma-separated held-out subjects")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum-subjects", type=int)
    parser.add_argument("--maximum-windows-per-trial", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    representations = tuple(value.strip() for value in args.representations.split(",") if value.strip())
    unknown = set(representations) - set(REPRESENTATIONS)
    if not targets or unknown or len(set(targets)) != len(targets):
        raise ValueError("targets must be unique and representations must be known")
    args.output_root.mkdir(parents=True, exist_ok=True)
    parameter_counts = {
        name: hla_parameter_count(HLAMultiDatasetModel(
            {"screen": 2}, HLAEncoderConfig(representation=name),
        ))
        for name in representations
    }
    if "R3" in parameter_counts and "R2-wide" in parameter_counts:
        ratio = parameter_counts["R2-wide"] / parameter_counts["R3"]
        if not 0.8 <= ratio <= 1.25:
            raise ValueError(f"R2-wide/R3 parameter ratio {ratio:.3f} is not a valid capacity control")
    for target in targets:
        pending = [
            representation for representation in representations
            if not _complete(args.output_root / f"{representation}_{target}_s{args.seed}")
        ]
        if not pending:
            print(f"SKIP_ALL_COMPLETE {target}", flush=True)
            continue
        preparation_config = HLANeuralRunConfig(
            representation="R3", sensor_view=args.sensor_view, target_subject=target,
            seed=args.seed, maximum_subjects=args.maximum_subjects,
            maximum_windows_per_trial=args.maximum_windows_per_trial,
            batch_size=args.batch_size, maximum_epochs=args.epochs,
            patience=args.patience, device=args.device,
        )
        prepared = prepare_hla_screen_examples(args.dataset_root, preparation_config)
        for representation in representations:
            destination = args.output_root / f"{representation}_{target}_s{args.seed}"
            if _complete(destination):
                print(f"SKIP_COMPLETE {destination.name}", flush=True)
                continue
            run_neural_subject_fold(
                args.dataset_root,
                destination,
                HLANeuralRunConfig(
                    representation=representation,
                    sensor_view=args.sensor_view,
                    target_subject=target,
                    seed=args.seed,
                    maximum_subjects=args.maximum_subjects,
                    maximum_windows_per_trial=args.maximum_windows_per_trial,
                    batch_size=args.batch_size,
                    maximum_epochs=args.epochs,
                    patience=args.patience,
                    device=args.device,
                ),
                prepared=prepared,
            )
            print(f"COMPLETE {destination.name}", flush=True)
    rows: dict[str, dict[str, float]] = {}
    for representation in representations:
        rows[representation] = {}
        for target in targets:
            path = args.output_root / f"{representation}_{target}_s{args.seed}" / "run_manifest.json"
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            rows[representation][target] = float(value["metrics"]["active_macro_f1"])
    common_targets = sorted(set.intersection(*(
        set(values) for values in rows.values()
    ))) if rows else []
    summary: dict[str, object] = {
        "status": "complete" if len(common_targets) == len(targets) else "partial",
        "seed": args.seed,
        "sensor_view": args.sensor_view,
        "targets": targets,
        "common_completed_targets": common_targets,
        "parameter_counts": parameter_counts,
        "representations": {},
        "paired_differences_vs_R0": {},
    }
    for representation, values in rows.items():
        observed = np.asarray([values[target] for target in common_targets])
        summary["representations"][representation] = {
            "mean_active_macro_f1": float(observed.mean()) if len(observed) else None,
            "subject_values": values,
        }
        if representation != "R0" and "R0" in rows and common_targets:
            difference = observed - np.asarray([rows["R0"][target] for target in common_targets])
            summary["paired_differences_vs_R0"][representation] = {
                "mean": float(difference.mean()),
                "median": float(np.median(difference)),
                "improved_subjects": int(np.sum(difference > 0)),
                "ci95": _bootstrap(difference, args.seed),
            }
    (args.output_root / "screen_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
