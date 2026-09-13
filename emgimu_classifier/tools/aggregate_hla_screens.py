from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _evaluation_signature(path: Path) -> tuple[tuple[str, str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return tuple(
            (row["trial_id"], row["start_timestamp_ms"], row["actual_label"])
            for row in rows
        )


def _bootstrap(values: np.ndarray, seed: int, iterations: int) -> list[float | None]:
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    sampled = np.asarray([
        np.mean(values[rng.integers(0, len(values), len(values))])
        for _ in range(iterations)
    ])
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def aggregate(
    roots: list[Path],
    *,
    baseline: str,
    challenger: str,
    bootstrap_seed: int = 42,
    bootstrap_iterations: int = 2000,
) -> dict[str, object]:
    runs: dict[tuple[int, str, str], dict[str, object]] = {}
    signatures: dict[str, tuple[tuple[str, str, str], ...]] = {}
    sensor_views: set[str] = set()
    commits: set[str] = set()
    for root in roots:
        summary = json.loads((root / "screen_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"screen is not complete: {root}")
        seed = int(summary["seed"])
        sensor_views.add(str(summary["sensor_view"]))
        for manifest_path in root.glob("*_s*/run_manifest.json"):
            run = json.loads(manifest_path.read_text(encoding="utf-8"))
            key = (seed, str(run["representation"]), str(run["target_subject"]))
            if key in runs:
                raise ValueError(f"duplicate run {key}")
            runs[key] = run
            commit = run.get("source_commit")
            if commit:
                commits.add(str(commit))
            target = str(run["target_subject"])
            signature = _evaluation_signature(manifest_path.parent / "predictions.csv")
            previous = signatures.setdefault(target, signature)
            if previous != signature:
                raise ValueError(f"evaluation examples differ across matched runs for {target}")
    if len(sensor_views) != 1:
        raise ValueError(f"screens mix sensor views: {sorted(sensor_views)}")
    seeds = sorted({key[0] for key in runs})
    subjects = sorted(
        {key[2] for key in runs if key[1] == baseline}
        & {key[2] for key in runs if key[1] == challenger}
    )
    missing = [
        (seed, representation, subject)
        for seed in seeds for representation in (baseline, challenger) for subject in subjects
        if (seed, representation, subject) not in runs
    ]
    if missing:
        raise ValueError(f"matched comparison is incomplete: {missing[:5]}")
    subject_scores: dict[str, dict[str, float]] = defaultdict(dict)
    per_class: dict[str, list[dict[str, float]]] = defaultdict(list)
    for representation in (baseline, challenger):
        for subject in subjects:
            values = [
                float(runs[(seed, representation, subject)]["metrics"]["active_macro_f1"])
                for seed in seeds
            ]
            subject_scores[representation][subject] = float(np.mean(values))
        class_count = len(runs[(seeds[0], representation, subjects[0])]["per_class"])
        for label in range(class_count):
            subject_values = []
            for subject in subjects:
                seed_values = [
                    float(runs[(seed, representation, subject)]["per_class"][label]["recall"])
                    for seed in seeds
                ]
                subject_values.append(float(np.mean(seed_values)))
            per_class[representation].append({
                "task_label": label,
                "mean_recall": float(np.mean(subject_values)),
                "subject_standard_deviation": float(np.std(subject_values, ddof=1)),
            })
    baseline_values = np.asarray([subject_scores[baseline][subject] for subject in subjects])
    challenger_values = np.asarray([subject_scores[challenger][subject] for subject in subjects])
    difference = challenger_values - baseline_values
    return {
        "status": "complete",
        "sensor_view": next(iter(sensor_views)),
        "seeds": seeds,
        "subjects": subjects,
        "source_commits": sorted(commits),
        "evaluation_pairing_verified": True,
        "representations": {
            representation: {
                "subject_seed_mean_active_macro_f1": subject_scores[representation],
                "mean_active_macro_f1": float(np.mean([
                    subject_scores[representation][subject] for subject in subjects
                ])),
                "per_class": per_class[representation],
            }
            for representation in (baseline, challenger)
        },
        "paired_challenger_minus_baseline": {
            "baseline": baseline,
            "challenger": challenger,
            "mean": float(np.mean(difference)),
            "median": float(np.median(difference)),
            "improved_subjects": int(np.sum(difference > 0)),
            "subject_count": len(subjects),
            "ci95": _bootstrap(difference, bootstrap_seed, bootstrap_iterations),
            "bootstrap_unit": "subject_after_averaging_seeds",
            "bootstrap_iterations": bootstrap_iterations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate matched HLA screens across seeds")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    result = aggregate(
        args.roots, baseline=args.baseline, challenger=args.challenger,
        bootstrap_seed=args.bootstrap_seed, bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
