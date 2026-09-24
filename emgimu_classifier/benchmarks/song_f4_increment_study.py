"""Exploratory F4 conditional increment on the user's Song 8-channel sessions.

The same S01/S02 source, S03 validation and S04 final split is reused. S04 has
already been inspected by other studies, so this does not reset the test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import _join_batches, load_session
from benchmarks.song_spd_increment_study import metrics, trial_probabilities
from emgimu.feature_bank.families import (
    LocalDetailFamily, SpdTangentFamily, SpectralStateFamily,
)


ARMS = {
    "F0": ("F0",),
    "F0_plus_SPD": ("F0", "SPD"),
    "F0_plus_F4": ("F0", "F4"),
    "F0_plus_SPD_plus_F4": ("F0", "SPD", "F4"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(source: Path, baseline_path: Path, output: Path) -> dict:
    print("[1/4] load hash-checked Song S01/S02/S03 and fit source-only features", flush=True)
    data = {sid: load_session(source / f"2026-09-18_{sid}", sid, "causal")
            for sid in ("S01", "S02", "S03")}
    source_batch = _join_batches([data["S01"], data["S02"]])
    source_labels = np.concatenate([data[sid]["hand"] for sid in ("S01", "S02")])
    families = {"F0": LocalDetailFamily().fit(source_batch),
                "SPD": SpdTangentFamily().fit(source_batch),
                "F4": SpectralStateFamily().fit(source_batch)}
    features = {name: {sid: family.transform(item["batch"])
                       for sid, item in data.items()}
                for name, family in families.items()}

    print("[2/4] train fixed four-arm logistic comparison on S01/S02", flush=True)
    models = {}
    for arm, names in ARMS.items():
        train = np.concatenate([
            np.concatenate([features[name][sid] for name in names], axis=1)
            for sid in ("S01", "S02")])
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(train, source_labels)
        models[arm] = model
    classes = models["F0"][-1].classes_
    if classes.tolist() != ["fist", "index_pinch", "neutral", "open_hand"] or any(
            not np.array_equal(model[-1].classes_, classes) for model in models.values()):
        raise ValueError("Song hand class order changed")

    def score_session(sid: str):
        scores, probabilities, rows, expected_ids, expected_truth = {}, {}, [], None, None
        for arm, names in ARMS.items():
            matrix = np.concatenate([features[name][sid] for name in names], axis=1)
            ids, truth, probability = trial_probabilities(
                data[sid]["hand"], models[arm].predict_proba(matrix),
                data[sid]["trial"], classes)
            if expected_ids is None:
                expected_ids, expected_truth = ids, truth
            elif not np.array_equal(ids, expected_ids) or not np.array_equal(truth, expected_truth):
                raise ValueError("Comparison arms do not cover identical formal trials")
            scores[arm] = metrics(truth, probability, classes)
            probabilities[arm] = probability
            for trial, label, prob in zip(ids, truth, probability):
                rows.append({"session": sid, "arm": arm, "trial_id": str(trial),
                             "truth": str(label),
                             **{f"p_{name}": float(prob[i])
                                for i, name in enumerate(classes)}})
        return scores, probabilities, rows, expected_truth

    print("[3/4] evaluate S03; verify frozen F0/SPD baseline", flush=True)
    validation, validation_p, validation_rows, validation_truth = score_session("S03")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["source_hdf5_sha256"]["S03"] != data["S03"]["audit"]["sha256"]:
        raise ValueError("Baseline S03 source differs")
    for arm in ("F0", "F0_plus_SPD"):
        frozen = baseline["source_zero_shot"]["validation"][arm]
        if any(abs(validation[arm][name] - frozen[name]) > 1e-5
               for name in ("accuracy", "macro_f1", "log_loss", "brier")):
            raise ValueError(f"S03 {arm} fails saved source baseline replay")

    print("[4/4] load S04 only after fixed models and S03 checks", flush=True)
    data["S04"] = load_session(source / "2026-09-18_S04", "S04", "causal")
    for name, family in families.items():
        features[name]["S04"] = family.transform(data["S04"]["batch"])
    final, final_p, final_rows, final_truth = score_session("S04")
    if baseline["source_hdf5_sha256"]["S04"] != data["S04"]["audit"]["sha256"]:
        raise ValueError("Baseline S04 source differs")
    for arm in ("F0", "F0_plus_SPD"):
        frozen = baseline["source_zero_shot"]["final"][arm]
        if any(abs(final[arm][name] - frozen[name]) > 1e-5
               for name in ("accuracy", "macro_f1", "log_loss", "brier")):
            raise ValueError(f"S04 {arm} fails saved source baseline replay")

    scores = {"validation": validation, "final": final}
    deltas, paired = {}, {}
    for split, truth, probabilities in (("validation", validation_truth, validation_p),
                                        ("final", final_truth, final_p)):
        deltas[split] = {}
        paired[split] = {}
        for base, candidate in (("F0", "F0_plus_F4"),
                                ("F0_plus_SPD", "F0_plus_SPD_plus_F4")):
            b, c = scores[split][base], scores[split][candidate]
            deltas[split][f"{candidate}_vs_{base}"] = {
                "delta_log_loss": b["log_loss"] - c["log_loss"],
                "delta_brier": b["brier"] - c["brier"],
                "delta_macro_f1": c["macro_f1"] - b["macro_f1"],
            }
            base_correct = classes[np.argmax(probabilities[base], axis=1)] == truth
            candidate_correct = classes[np.argmax(probabilities[candidate], axis=1)] == truth
            paired[split][f"{candidate}_vs_{base}"] = {
                "corrected_trials": int(np.sum(~base_correct & candidate_correct)),
                "new_errors": int(np.sum(base_correct & ~candidate_correct)),
                "both_correct": int(np.sum(base_correct & candidate_correct)),
                "both_wrong": int(np.sum(~base_correct & ~candidate_correct)),
            }

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "F4_INCREMENT_TRIAL_PREDICTIONS.csv"
    rows = validation_rows + final_rows
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    increment_path = output / "F4_INCREMENT_CONDITIONAL.csv"
    with increment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "base_arm", "candidate_arm", "trials",
            "delta_log_loss", "delta_brier", "delta_macro_f1"),
            lineterminator="\n")
        writer.writeheader()
        writer.writerows({"split": split, "base_arm": base,
                          "candidate_arm": candidate,
                          "trials": scores[split][candidate]["trials"],
                          **deltas[split][f"{candidate}_vs_{base}"]}
                         for split in ("validation", "final")
                         for base, candidate in (("F0", "F0_plus_F4"),
                                                 ("F0_plus_SPD", "F0_plus_SPD_plus_F4")))
    paired_path = output / "F4_INCREMENT_PAIRED_ERRORS.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "base_arm", "candidate_arm", "corrected_trials",
            "new_errors", "both_correct", "both_wrong"), lineterminator="\n")
        writer.writeheader()
        writer.writerows({"split": split, "base_arm": base,
                          "candidate_arm": candidate,
                          **paired[split][f"{candidate}_vs_{base}"]}
                         for split in ("validation", "final")
                         for base, candidate in (("F0", "F0_plus_F4"),
                                                 ("F0_plus_SPD", "F0_plus_SPD_plus_F4")))
    classifier_root = Path(__file__).resolve().parents[1]
    result = {
        "status": "exploratory_song_one_person_one_day_f4_conditional_increment",
        "source_sessions": ["S01", "S02"], "validation_session": "S03",
        "final_session": "S04", "filter_mode": "causal_continuous",
        "source_hdf5_sha256": {sid: item["audit"]["sha256"]
                               for sid, item in data.items()},
        "source_code_sha256": {
            "study": sha256(Path(__file__)),
            "song_loader": sha256(classifier_root / "benchmarks/song_real8_study.py"),
            "song_trial_metrics": sha256(classifier_root / "benchmarks/song_spd_increment_study.py"),
            "feature_families": sha256(classifier_root / "src/emgimu/feature_bank/families.py"),
            "feature_batch": sha256(classifier_root / "src/emgimu/feature_bank/core.py"),
        },
        "frozen_baseline_sha256": sha256(baseline_path),
        "frozen_baseline_replayed": True,
        "classes": classes.tolist(),
        "arms": {arm: list(names) for arm, names in ARMS.items()},
        "feature_counts": {name: len(family.feature_names)
                           for name, family in families.items()},
        "scores": scores,
        "conditional_increments": deltas,
        "paired_errors": paired,
        "trial_predictions": prediction_path.name,
        "trial_predictions_sha256": sha256(prediction_path),
        "conditional_incremental_csv": increment_path.name,
        "conditional_incremental_sha256": sha256(increment_path),
        "paired_errors_csv": paired_path.name,
        "paired_errors_sha256": sha256(paired_path),
        "boundary": "Only Song one-person/one-day cue-labelled stable formal trials. S01/S02 source-only family/scaler/model fits, S03 validation, S04 final loaded after fixed-model S03 checks; S04 was previously inspected. S01-S03 failed whole-session collection readiness. No live USB, new-wearing, multiuser/day or deployable accuracy claim. F4 is a current reference spectral family, not historical Frequency equivalence.",
    }
    (output / "F4_INCREMENT_RESULTS.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"S03_F0_SPD_F1": validation["F0_plus_SPD"]["macro_f1"],
                      "S03_F0_SPD_F4_F1": validation["F0_plus_SPD_plus_F4"]["macro_f1"],
                      "S04_F0_SPD_F1": final["F0_plus_SPD"]["macro_f1"],
                      "S04_F0_SPD_F4_F1": final["F0_plus_SPD_plus_F4"]["macro_f1"]}),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.baseline, args.output)
