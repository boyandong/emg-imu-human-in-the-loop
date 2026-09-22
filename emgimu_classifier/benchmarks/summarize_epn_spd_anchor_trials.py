"""Verify frozen SPD trial-study outputs and deliver a compact score record."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path, output: Path) -> None:
    phases = {}
    for phase in ("validation", "final"):
        directory = root / f"feature_bank_epn_spd_anchor_{phase}_20260922"
        manifest_path = directory / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["phase"] != phase or tuple(manifest["users"]) != (
            (16, 17, 18) if phase == "validation" else (19, 20, 21)
        ):
            raise ValueError("phase or target users do not match frozen protocol")
        for name, expected in manifest["outputs_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"saved output digest mismatch: {phase}/{name}")
        if digest(Path(manifest["selection_file"])) != manifest["selection_file_sha256"]:
            raise ValueError("frozen calibration selection digest mismatch")
        with (directory / "pooled_scores.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["shots_per_class"]) for row in rows] != [1, 2, 5]:
            raise ValueError("pooled scores have unexpected calibration budgets")
        pooled = []
        for row in rows:
            pooled.append({key: (
                int(value) if key in {"shots_per_class", "subjects", "evaluation_trials"}
                else float(value) if key in {"macro_f1", "accuracy", "log_loss", "brier", "ece"}
                else json.loads(value) if key == "per_class_f1_json" else value
            ) for key, value in row.items()})
        phases[phase] = {
            "source_reference_sha256": manifest["source_reference_sha256"],
            "selection_file_sha256": manifest["selection_file_sha256"],
            "manifest_sha256": digest(manifest_path),
            "output_sha256": manifest["outputs_sha256"],
            "pooled": pooled,
        }
    if phases["validation"]["source_reference_sha256"] != phases["final"]["source_reference_sha256"]:
        raise ValueError("source reference changed between phases")
    summary = {
        "completion_proven": False,
        "study": "EPN frozen-source SPD tangent personal trial prototypes",
        "source_users": "1-15",
        "target_users": {"validation": "16-18", "final": "19-21"},
        "zero_shot": "N/A: no personal class prototypes",
        "phases": phases,
        "boundary": "Offline six-class EPN candidate; not historical RLCS, fixed multi-family Core incremental value, own-device accuracy or deployed recognition.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)
