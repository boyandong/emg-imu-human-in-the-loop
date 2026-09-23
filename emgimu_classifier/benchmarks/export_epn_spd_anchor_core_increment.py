"""Export the frozen EPN SPD/Core replay into the required delivery tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from emgimu.datasets.epn612 import GESTURES


ARMS = ("F0", "Core", "SPD_Anchor", "Core_plus_SPD_Anchor")
BUDGETS = (1, 2, 5)
PHASE_USERS = {"validation": (16, 17, 18), "final": (19, 20, 21)}
CORE_NAME = "F0+F3_Ring+F2b_CSP+F6_IMU"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(source: Path, output: Path) -> dict:
    replay = json.loads(source.read_text(encoding="utf-8"))
    phase = replay["phase"]
    if phase not in PHASE_USERS or tuple(replay["users"]) != PHASE_USERS[phase]:
        raise ValueError("replay phase/users mismatch")
    if tuple(replay["budgets"]) != BUDGETS or tuple(replay["arms"]) != ARMS:
        raise ValueError("replay budget/arms mismatch")
    scores = replay["scores"]
    by_key = {(str(row["subject"]), int(row["shots_per_class"]), row["model"]): row
              for row in scores}
    expected = {(str(subject), budget, arm)
                for subject in ("ALL", *PHASE_USERS[phase]) for budget in BUDGETS for arm in ARMS}
    if len(scores) != len(expected) or set(by_key) != expected:
        raise ValueError("replay score cells are missing or duplicated")
    for row in scores:
        if row["phase"] != phase or int(row["evaluation_trials"]) <= 0:
            raise ValueError("replay score phase/trial count mismatch")
        per_class = json.loads(row["per_class_f1_json"])
        if set(per_class) != set(GESTURES):
            raise ValueError("native per-class scores are incomplete")
    for subject in ("ALL", *PHASE_USERS[phase]):
        for budget in BUDGETS:
            counts = {by_key[str(subject), budget, arm]["evaluation_trials"] for arm in ARMS}
            if len(counts) != 1:
                raise ValueError("comparison arms use different evaluation trials")

    family_rows, curve_rows, increment_rows = [], [], []
    for subject in ("ALL", *PHASE_USERS[phase]):
        for budget in BUDGETS:
            base = by_key[str(subject), budget, "Core"]
            added = by_key[str(subject), budget, "Core_plus_SPD_Anchor"]
            common = {"dataset": "epn612", "phase": phase, "subject": subject,
                      "session/domain": "N/A: no validated repeated-session key",
                      "condition": "cross_user", "shots_per_class": budget,
                      "calibration_budget": budget,
                      "evaluation_unit": "whole_native_trial",
                      "evaluation_trials": base["evaluation_trials"]}
            increment_rows.append({**common, "core_bank": CORE_NAME,
                "added_family": "F7_SPD_tangent_personal_anchor_fixed_late_fusion",
                "delta_logloss": base["log_loss"] - added["log_loss"],
                "delta_macro_f1": added["macro_f1"] - base["macro_f1"],
                "delta_brier": base["brier"] - added["brier"],
                "comparison": "frozen Core versus fixed 1/2(Core + SPD Anchor) on identical noncalibration trials"})
            for arm in ARMS:
                score = by_key[str(subject), budget, arm]
                family = {"F0": "F0", "Core": CORE_NAME,
                          "SPD_Anchor": "F7_SPD_tangent_personal_anchor",
                          "Core_plus_SPD_Anchor": CORE_NAME + "+F7_SPD_tangent_personal_anchor"}[arm]
                metrics = {key: score[key] for key in ("macro_f1", "accuracy", "log_loss", "brier", "ece")}
                family_rows.append({**common, "feature_family": family, "model": arm,
                                    **metrics, "per_class_f1_json": score["per_class_f1_json"],
                                    "protocol": "fixed late fusion; source-frozen Core and source-SPD reference"})
                curve_common = {key: value for key, value in common.items()
                                if key not in ("calibration_budget", "evaluation_unit")}
                curve_rows.append({**curve_common, "feature_bank": family, "method": arm,
                                   **metrics, "per_class_f1_json": score["per_class_f1_json"]})
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    names = ("feature_family_results.csv", "conditional_incremental.csv", "calibration_curve.csv")
    for name, rows in zip(names, (family_rows, increment_rows, curve_rows)):
        save_csv(output / name, rows)
    manifest = {"phase": phase, "dataset": "epn612", "source_users": list(range(1, 16)),
                "target_users": list(PHASE_USERS[phase]), "calibration_budgets": list(BUDGETS),
                "families": ["F0", "F3_Ring", "F2b_CSP", "F6_IMU", "F7_SPD_tangent_personal_anchor"],
                "methods": list(ARMS), "model_hyperparameters": {"fixed_mixture_core_weight": 0.5},
                "preprocessing_config": "frozen native-trial prediction inputs; see source manifests",
                "split_ids": "exact source calibration/evaluation trial IDs verified by replay; source hashes below",
                "random_seeds": {"reused_calibration_selection": 20260915},
                "source_result": str(source), "source_result_sha256": sha(source),
                "source_inputs_sha256": replay["inputs_sha256"],
                "export_script_sha256": sha(Path(__file__)),
                "output_sha256": {name: sha(output / name) for name in names},
                "boundary": replay["boundary"]}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": phase, "family_rows": len(family_rows),
                      "increment_rows": len(increment_rows), "curve_rows": len(curve_rows)}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
