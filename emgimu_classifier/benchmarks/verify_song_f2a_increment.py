"""Recompute Song F2a held-out trial scores from exported probabilities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, atol=1e-10, rtol=1e-10))


def verify(source: Path, directory: Path, baseline_path: Path, runner: Path) -> dict:
    result_path = directory / "F2a_INCREMENT_RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (result["status"] != "exploratory_song_one_person_one_day_f2a_conditional_increment"
            or result["source_sessions"] != ["S01", "S02"]
            or result["validation_session"] != "S03"
            or result["final_session"] != "S04"
            or result["filter_mode"] != "causal_continuous"
            or result["frozen_baseline_sha256"] != sha256(baseline_path)
            or not result["frozen_baseline_replayed"]
            or result["source_code_sha256"]["study"] != sha256(runner)):
        raise ValueError("Song F2a study provenance changed")
    classifier_root = runner.resolve().parents[1]
    for key, path in {
            "song_loader": classifier_root / "benchmarks/song_real8_study.py",
            "song_trial_metrics": classifier_root / "benchmarks/song_spd_increment_study.py",
            "feature_families": classifier_root / "src/emgimu/feature_bank/families.py",
            "feature_batch": classifier_root / "src/emgimu/feature_bank/core.py",
    }.items():
        if result["source_code_sha256"][key] != sha256(path):
            raise ValueError(f"Source implementation changed: {key}")
    for sid in ("S01", "S02", "S03", "S04"):
        readiness = json.loads((source / f"2026-09-18_{sid}" /
                                "SESSION_COLLECTION_READINESS.json").read_text(encoding="utf-8"))
        if (result["source_hdf5_sha256"][sid] != readiness["hdf5_sha256"] or
                result["source_hdf5_sha256"][sid] != baseline["source_hdf5_sha256"][sid]):
            raise ValueError(f"Song HDF5 source identity changed: {sid}")
    classes = result["classes"]
    if classes != ["fist", "index_pinch", "neutral", "open_hand"] or \
            set(result["arms"]) != {"F0", "F0_plus_SPD", "F0_plus_F2a",
                                    "F0_plus_SPD_plus_F2a"}:
        raise ValueError("Song classes/arms changed")
    prediction_path = directory / result["trial_predictions"]
    if result["trial_predictions_sha256"] != sha256(prediction_path):
        raise ValueError("Trial prediction file changed")
    for name, digest in (("conditional_incremental_csv", "conditional_incremental_sha256"),
                         ("paired_errors_csv", "paired_errors_sha256")):
        if result[digest] != sha256(directory / result[name]):
            raise ValueError(f"Song F2a delivery table changed: {name}")
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8", newline="")))
    grouped = {}
    for row in rows:
        sid, arm = row["session"], row["arm"]
        if sid not in ("S03", "S04") or arm not in result["arms"] or \
                row["truth"] not in classes:
            raise ValueError("Invalid Song trial identity")
        p = np.array([float(row[f"p_{name}"]) for name in classes])
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or \
                not close(float(p.sum()), 1.0):
            raise ValueError("Invalid Song trial probability")
        key = (sid, arm)
        if row["trial_id"] in grouped.setdefault(key, {}):
            raise ValueError("Duplicate Song trial prediction")
        grouped[key][row["trial_id"]] = (row["truth"], p)
    if len(grouped) != 8 or len(rows) != 1136:
        raise ValueError("Song F2a prediction coverage changed")
    for sid, split, count in (("S03", "validation", 140), ("S04", "final", 144)):
        baseline_ids = set(grouped[(sid, "F0")])
        for arm in result["arms"]:
            group = grouped[(sid, arm)]
            if len(group) != count or set(group) != baseline_ids or any(
                    group[trial][0] != grouped[(sid, "F0")][trial][0]
                    for trial in baseline_ids):
                raise ValueError("Song comparison arms do not cover identical trials")
            ids = sorted(group)
            truth = np.array([group[i][0] for i in ids])
            probability = np.stack([group[i][1] for i in ids])
            prediction = np.asarray(classes)[np.argmax(probability, axis=1)]
            reference = result["scores"][split][arm]
            one_hot = truth[:, None] == np.asarray(classes)[None, :]
            values = {"accuracy": accuracy_score(truth, prediction),
                      "macro_f1": f1_score(truth, prediction, labels=classes,
                                            average="macro", zero_division=0),
                      "log_loss": log_loss(truth, probability, labels=classes),
                      "brier": np.mean((probability - one_hot) ** 2)}
            if (reference["trials"] != count or
                    any(not close(float(value), reference[name])
                        for name, value in values.items()) or
                    reference["confusion_matrix"] != confusion_matrix(
                        truth, prediction, labels=classes).tolist() or
                    any(not close(reference["recall"][name], float(recall_score(
                        truth, prediction, labels=[name], average="macro",
                        zero_division=0))) for name in classes)):
                raise ValueError(f"Song {sid}/{arm} score does not replay")
            if arm in ("F0", "F0_plus_SPD"):
                frozen = baseline["source_zero_shot"][split][arm]
                if any(abs(reference[name] - frozen[name]) > 1e-5
                       for name in values):
                    raise ValueError("Saved F0/SPD baseline changed")
        for base, candidate in (("F0", "F0_plus_F2a"),
                                ("F0_plus_SPD", "F0_plus_SPD_plus_F2a")):
            comparison = f"{candidate}_vs_{base}"
            b, c = result["scores"][split][base], result["scores"][split][candidate]
            expected = {"delta_log_loss": b["log_loss"] - c["log_loss"],
                        "delta_brier": b["brier"] - c["brier"],
                        "delta_macro_f1": c["macro_f1"] - b["macro_f1"]}
            if any(not close(result["conditional_increments"][split][comparison][name], value)
                   for name, value in expected.items()):
                raise ValueError("Conditional increment does not replay")
            bgroup, cgroup = grouped[(sid, base)], grouped[(sid, candidate)]
            corrected = new_errors = both_correct = both_wrong = 0
            for trial in baseline_ids:
                bcorrect = classes[int(np.argmax(bgroup[trial][1]))] == bgroup[trial][0]
                ccorrect = classes[int(np.argmax(cgroup[trial][1]))] == cgroup[trial][0]
                corrected += (not bcorrect) and ccorrect
                new_errors += bcorrect and (not ccorrect)
                both_correct += bcorrect and ccorrect
                both_wrong += (not bcorrect) and (not ccorrect)
            paired = result["paired_errors"][split][comparison]
            if paired != {"corrected_trials": corrected, "new_errors": new_errors,
                          "both_correct": both_correct, "both_wrong": both_wrong}:
                raise ValueError("Paired Song errors do not replay")
    conditional_rows = list(csv.DictReader((directory / result["conditional_incremental_csv"]).open(
        encoding="utf-8", newline="")))
    paired_rows = list(csv.DictReader((directory / result["paired_errors_csv"]).open(
        encoding="utf-8", newline="")))
    wanted = {(split, base, candidate)
              for split in ("validation", "final")
              for base, candidate in (("F0", "F0_plus_F2a"),
                                      ("F0_plus_SPD", "F0_plus_SPD_plus_F2a"))}
    for table in (conditional_rows, paired_rows):
        if len(table) != 4 or {(row["split"], row["base_arm"], row["candidate_arm"])
                               for row in table} != wanted:
            raise ValueError("Song F2a delivery table coverage changed")
    for row in conditional_rows:
        split, base, candidate = row["split"], row["base_arm"], row["candidate_arm"]
        expected = result["conditional_increments"][split][f"{candidate}_vs_{base}"]
        if int(row["trials"]) != result["scores"][split][candidate]["trials"] or any(
                not close(float(row[key]), value) for key, value in expected.items()):
            raise ValueError("Song F2a conditional CSV does not replay")
    for row in paired_rows:
        split, base, candidate = row["split"], row["base_arm"], row["candidate_arm"]
        expected = result["paired_errors"][split][f"{candidate}_vs_{base}"]
        if any(int(row[key]) != value for key, value in expected.items()):
            raise ValueError("Song F2a paired-error CSV does not replay")
    audit = {
        "status": "all_song_f2a_trial_scores_and_pairs_recomputed",
        "result_sha256": sha256(result_path),
        "prediction_rows": len(rows),
        "source_sessions_verified_against_readiness_and_frozen_baseline": 4,
        "arm_count": len(result["arms"]),
        "boundary": "This read-back checks source/implementation hashes, 1136 trial probabilities, four-arm held-out scores, class recalls, confusion matrices and paired increments. It cannot turn one person/day and previously inspected S04 into live-device or new-cohort validation.",
    }
    (directory / "F2a_INCREMENT_VERIFICATION.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "prediction_rows", "arm_count")}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.study_dir, args.baseline, args.runner)
