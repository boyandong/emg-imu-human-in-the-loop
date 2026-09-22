"""Replay fixed Quality × reference-family interactions without fitting models.

The four probability compositions use the same frozen force trial predictions.
All families here are current references, not recovered historical algorithms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from emgimu.feature_bank.quality_corruption_study import SCENARIOS
from emgimu.feature_bank.screening import metrics


PAIRS = ("F1_X1H", "F3_Ring", "F2b_CSP", "F4_Spectral", "F5_Temporal")
BASE, QUALITY = "F0", "F9_Quality"
PHASE_USERS = {"validation": (7, 8), "final": (9, 10)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compositions(providers: dict[str, np.ndarray], family: str) -> dict[str, np.ndarray]:
    """Equal-provider late fusion; no quality routing or target fitting."""
    if family not in PAIRS:
        raise ValueError("pair was not frozen before the final replay")
    arms = {"B": (BASE,), "B_plus_family": (BASE, family),
            "B_plus_quality": (BASE, QUALITY),
            "B_plus_family_plus_quality": (BASE, family, QUALITY)}
    return {arm: np.mean([providers[name] for name in names], axis=0)
            for arm, names in arms.items()}


def interaction(arm_metrics: dict[str, dict]) -> dict[str, float]:
    b, f, q, fq = (arm_metrics[key] for key in
                    ("B", "B_plus_family", "B_plus_quality", "B_plus_family_plus_quality"))
    return {
        "S_negative_logloss": -fq["log_loss"] + f["log_loss"] + q["log_loss"] - b["log_loss"],
        "S_macro_f1": fq["macro_f1"] - f["macro_f1"] - q["macro_f1"] + b["macro_f1"],
        "S_negative_brier": -fq["brier"] + f["brier"] + q["brier"] - b["brier"],
    }


def pair_complementarity(truth: np.ndarray, a_prob: np.ndarray,
                         b_prob: np.ndarray) -> dict[str, float | None]:
    """Distinguish prediction disagreement from correctness disagreement."""
    a_pred, b_pred = a_prob.argmax(axis=1), b_prob.argmax(axis=1)
    a_wrong, b_wrong = a_pred != truth, b_pred != truth
    correlation = (float(np.corrcoef(a_wrong, b_wrong)[0, 1])
                   if np.any(a_wrong) and not np.all(a_wrong)
                   and np.any(b_wrong) and not np.all(b_wrong) else None)
    return {"error_correlation": correlation,
            "disagreement_rate": float(np.mean(a_pred != b_pred)),
            "correctness_disagreement_rate": float(np.mean(a_wrong != b_wrong)),
            "a_correct_b_wrong": float(np.mean(~a_wrong & b_wrong)),
            "a_wrong_b_correct": float(np.mean(a_wrong & ~b_wrong))}


def run(source_root: Path, output_root: Path, compact_output: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    arm_rows, interaction_rows, complementarity_rows = [], [], []
    sources = {}
    for phase, expected_users in PHASE_USERS.items():
        source = source_root / f"feature_bank_force_quality_{phase}_20260915"
        manifest_path, audit_path = source / "run_manifest.json", source / "replay_audit.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (manifest["phase"] != phase or manifest["classifier_or_family_fit"]
                or manifest["target_calibration"] or tuple(manifest["scenarios"]) != SCENARIOS
                or not {BASE, QUALITY, *PAIRS} <= set(manifest["families"])
                or audit["status"] != "ok" or not audit["source_hashes_checked"]
                or audit["max_absolute_probability_error"] > 1e-10):
            raise ValueError("source replay protocol is not eligible")
        source_run = source_root / manifest["source_run"]
        for name, expected in manifest["source_artifact_sha256"].items():
            if sha(source_run / name) != expected:
                raise ValueError(f"frozen source artifact changed: {phase}/{name}")
        npz_path = source / "corruption_predictions.npz"
        sources[phase] = {name: sha(source / name) for name in
                          ("corruption_predictions.npz", "run_manifest.json", "replay_audit.json")}
        with np.load(npz_path, allow_pickle=False) as saved:
            truth, users, trials = saved["labels"], saved["users"], saved["trials"]
            if set(users.tolist()) != set(expected_users) or len(set(trials.tolist())) != len(trials):
                raise ValueError("held-out subjects or native trial IDs are inconsistent")
            with np.load(source_run / "heldout_predictions.npz", allow_pickle=False) as original:
                for key, value in (("labels", truth), ("users", users), ("trials", trials)):
                    np.testing.assert_array_equal(original[key], value)
                for user in expected_users:
                    np.testing.assert_allclose(
                        original[f"{user}_0_baseline_F0"],
                        saved[f"clean::{BASE}"][users == user], rtol=1e-9, atol=1e-10)
            for scenario in SCENARIOS:
                providers = {name: saved[f"{scenario}::{name}"] for name in
                             (BASE, QUALITY, *PAIRS)}
                if any(p.shape != (len(truth), 7) or not np.all(np.isfinite(p)) or
                       not np.allclose(p.sum(axis=1), 1, atol=1e-8)
                       for p in providers.values()):
                    raise ValueError("provider probabilities are not aligned")
                for family in PAIRS:
                    arms = compositions(providers, family)
                    for user in (*expected_users, "ALL"):
                        mask = np.ones(len(truth), bool) if user == "ALL" else users == user
                        scored = {name: metrics(truth[mask], p[mask], np.ones(mask.sum()))
                                  for name, p in arms.items()}
                        for name, score in scored.items():
                            arm_rows.append({"phase": phase, "scenario": scenario, "subject": user,
                                             "base": BASE, "family": family, "quality": QUALITY,
                                             "arm": name, "evaluation_trials": int(mask.sum()), **score})
                        interaction_rows.append({"phase": phase, "scenario": scenario, "subject": user,
                                                 "base": BASE, "family_a": family, "family_b": QUALITY,
                                                 "evaluation_trials": int(mask.sum()),
                                                 **interaction(scored)})
                        complementarity_rows.append({
                            "phase": phase, "scenario": scenario, "subject": user,
                            "family_a": family, "family_b": QUALITY,
                            "evaluation_trials": int(mask.sum()),
                            **pair_complementarity(truth[mask], providers[family][mask],
                                                   providers[QUALITY][mask]),
                        })
        print(f"{phase}: replayed {len(SCENARIOS)} fixed scenarios, {len(PAIRS)} pairs", flush=True)
    outputs = {"interaction_results.csv": interaction_rows,
               "arm_scores.csv": arm_rows,
               "error_complementarity.csv": complementarity_rows}
    for filename, rows in outputs.items():
        save_csv(output_root / filename, rows)
    synthetic_summary = []
    for phase in PHASE_USERS:
        for family in PAIRS:
            matched = [row for row in interaction_rows if row["phase"] == phase
                       and row["family_a"] == family and row["subject"] == "ALL"
                       and row["scenario"] != "clean"]
            base_scores = [row["macro_f1"] for row in arm_rows if row["phase"] == phase
                           and row["family"] == family and row["subject"] == "ALL"
                           and row["scenario"] != "clean" and row["arm"] == "B"]
            full_scores = [row["macro_f1"] for row in arm_rows if row["phase"] == phase
                           and row["family"] == family and row["subject"] == "ALL"
                           and row["scenario"] != "clean"
                           and row["arm"] == "B_plus_family_plus_quality"]
            if len(matched) != 7 or len(base_scores) != 7 or len(full_scores) != 7:
                raise ValueError("synthetic scenario summary is incomplete")
            synthetic_summary.append({"phase": phase, "family": family,
                                      "scenarios": 7,
                                      "mean_S_negative_logloss": float(np.mean([r["S_negative_logloss"] for r in matched])),
                                      "positive_S_negative_logloss_scenarios": sum(r["S_negative_logloss"] > 0 for r in matched),
                                      "mean_baseline_macro_f1": float(np.mean(base_scores)),
                                      "mean_full_arm_macro_f1": float(np.mean(full_scores))})
    compact = {
        "completion_proven": False,
        "pairs": list(PAIRS),
        "protocol": "Frozen source provider probabilities; fixed equal-weight B/F/Q/FQ compositions; no refit or target calibration",
        "analysis_source_sha256": sha(Path(__file__)),
        "source_runs": sources,
        "output_sha256": {name: sha(output_root / name) for name in outputs},
        "rows": {name: len(rows) for name, rows in outputs.items()},
        "pooled_clean": [row for row in interaction_rows if row["subject"] == "ALL" and row["scenario"] == "clean"],
        "pooled_clean_complementarity": [row for row in complementarity_rows if row["subject"] == "ALL" and row["scenario"] == "clean"],
        "pooled_synthetic_summary": synthetic_summary,
        "boundary": "Quality × current reference F1/Ring/CSP/Spectral/Temporal probability-composition interactions under synthetic perturbations, not historical-family equivalence, physiological synergy, real-device noise, or complete Stage 4 coverage.",
    }
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("compact_output", type=Path)
    args = parser.parse_args()
    run(args.source_root, args.output_root, args.compact_output)
