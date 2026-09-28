"""Independently rescore every saved public DS2 v9 trial prediction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from public_ds2_force_v9 import sha256


def score(truth: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
    pred = probs.argmax(axis=1)
    target = np.eye(4)[truth]
    return (float(accuracy_score(truth, pred)),
            float(f1_score(truth, pred, labels=np.arange(4), average="macro", zero_division=0)),
            float(log_loss(truth, probs, labels=np.arange(4))),
            float(np.mean(np.sum((probs - target) ** 2, axis=1))))


def verify(directory: Path) -> dict:
    result = json.loads((directory / "RESULTS.json").read_text(encoding="utf-8"))
    path = directory / result["trial_predictions"]
    if sha256(path) != result["trial_predictions_sha256"]:
        raise ValueError("prediction CSV hash differs from result manifest")
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != 2 * 3 * (471 + 438):
        raise ValueError("unexpected number of held-out trial/arm rows")
    grouped = {}
    for row in rows:
        key = (row["training_mode"], row["arm"], row["split"])
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 12:
        raise ValueError("missing mode/arm/split group")
    fingerprints = {}
    for (mode, arm, split), group in grouped.items():
        ids = np.array([int(r["raw_trial_index_zero_based"]) for r in group])
        subject = np.array([int(r["subject_folder"]) for r in group])
        truth = np.array([int(r["gesture_code"]) for r in group])
        force = np.array([int(r["force_code"]) for r in group])
        probs = np.array([[float(r[f"p{k}"]) for k in range(4)] for r in group])
        if (not np.array_equal(ids, np.sort(ids)) or len(np.unique(ids)) != len(ids) or
                not np.isfinite(probs).all() or np.any(probs < 0) or
                not np.allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-9) or
                set(subject) - set(result["split_subject_folders"][split]) or
                set(force) != {0, 1, 2}):
            raise ValueError(f"invalid saved prediction rows: {mode}/{arm}/{split}")
        reference = result["scores"][mode][arm][split]
        values = score(truth, probs)
        for name, actual in zip(("accuracy", "macro_f1", "log_loss", "brier"), values):
            if not np.isclose(actual, reference[name], rtol=0, atol=1e-10):
                raise ValueError(f"saved score differs: {mode}/{arm}/{split}/{name}")
        for code in range(3):
            selected = force == code
            reference = result["by_force_code"][mode][arm][split][str(code)]
            if selected.sum() != reference["trials"]:
                raise ValueError("force-condition trial count differs")
            for name, actual in zip(("accuracy", "macro_f1", "log_loss", "brier"),
                                    score(truth[selected], probs[selected])):
                if not np.isclose(actual, reference[name], rtol=0, atol=1e-10):
                    raise ValueError(f"force-condition score differs: {mode}/{arm}/{split}/{code}/{name}")
        fingerprints[(mode, split, arm)] = (ids, truth, force)
    for mode in result["training_modes"]:
        for split in ("validation", "final_descriptive"):
            base = fingerprints[(mode, split, "F0")]
            for arm in ("F1", "F0_plus_F1"):
                if any(not np.array_equal(a, b) for a, b in
                       zip(base, fingerprints[(mode, split, arm)])):
                    raise ValueError("paired arms score different trials")
    return {"status": "all_saved_trial_and_force_metrics_verified",
            "prediction_rows": len(rows),
            "groups": len(grouped),
            "per_force_counts": dict(Counter(row["force_code"] for row in rows)),
            "trial_predictions_sha256": result["trial_predictions_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    (args.directory / "VERIFICATION.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
