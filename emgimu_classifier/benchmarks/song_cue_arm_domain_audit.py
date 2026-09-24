"""Stratify saved Song family predictions by native prompted arm condition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from benchmarks.song_real8_study import ARMS, _text, parse_label
from benchmarks.song_spd_increment_study import metrics


SESSIONS = ("S01", "S02", "S03", "S04")
CLASSES = ("fist", "index_pinch", "neutral", "open_hand")
CORE = "F0_F2c"
ADDITIONS = ("F1", "F2a", "F2b", "F4", "F5", "F6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_cues(source: Path, expected_hashes: dict[str, str]) -> dict[str, tuple[str, str]]:
    cues = {}
    for sid in SESSIONS:
        path = source / f"2026-09-18_{sid}" / "session.h5"
        if sha256(path) != expected_hashes[sid]:
            raise ValueError(f"Song source recording changed: {sid}")
        with h5py.File(path) as handle:
            trials = handle["trials"][:]
        for row in trials:
            if (_text(row["trial_kind"]) != "formal" or not bool(row["valid"]) or
                    _text(row["completion_status"]) != "completed"):
                continue
            cue_arm, hand = parse_label(_text(row["label"]))
            identity = f"{sid}:{int(row['trial_id'])}"
            if identity in cues:
                raise ValueError(f"duplicate Song formal trial: {identity}")
            cues[identity] = (cue_arm, hand)
    return cues


def score(rows: list[dict]) -> dict:
    truth = np.asarray([row["true_label"] for row in rows])
    probability = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES]
                              for row in rows], dtype=np.float64)
    value = metrics(truth, probability, np.asarray(CLASSES))
    return {key: value[key] for key in ("trials", "accuracy", "macro_f1",
                                         "log_loss", "brier", "recall", "confusion_matrix")}


def run(source: Path, screen_path: Path, predictions: Path, output: Path) -> dict:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if (screen["status"] != "exploratory_song_fixed_family_held_out_screen" or
            sha256(predictions) != screen["predictions_sha256"]):
        raise ValueError("Song family screen or prediction digest changed")
    cues = source_cues(source, screen["source_hdf5_sha256"])
    with predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7966 or set(screen["arms"]) != {row["arm"] for row in rows}:
        raise ValueError("Song family prediction inventory changed")
    expected = set()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["arm"], row["trial_id"])
        if key in expected:
            raise ValueError(f"duplicate model/trial prediction: {key}")
        expected.add(key)
        if row["trial_id"] not in cues:
            raise ValueError(f"predicted trial not in source: {row['trial_id']}")
        cue_arm, hand = cues[row["trial_id"]]
        if row["true_label"] != hand or row["held_out_session"] != row["trial_id"].split(":")[0]:
            raise ValueError(f"prediction/source label mismatch: {row['trial_id']}")
        grouped[(row["arm"], cue_arm)].append(row)
    trial_ids = {row["trial_id"] for row in rows}
    if len(trial_ids) != 569 or len(expected) != len(screen["arms"]) * len(trial_ids):
        raise ValueError("Song family screen has unmatched trial membership")
    domain_counts = Counter(cues[trial][0] for trial in trial_ids)
    domain_scores = {arm: {domain: score(grouped[(arm, domain)]) for domain in ARMS}
                     for arm in screen["arms"]}
    for arm in screen["arms"]:
        for domain in ARMS:
            if domain_scores[arm][domain]["trials"] != domain_counts[domain]:
                raise ValueError(f"arm/domain trial count mismatch: {arm}/{domain}")
    conditional = {}
    for added in ADDITIONS:
        arm = f"Core_{added}"
        conditional[added] = {}
        for domain in ARMS:
            base = domain_scores[CORE][domain]
            candidate = domain_scores[arm][domain]
            conditional[added][domain] = {
                "trials": domain_counts[domain],
                "delta_log_loss": base["log_loss"] - candidate["log_loss"],
                "delta_brier": base["brier"] - candidate["brier"],
                "delta_macro_f1": candidate["macro_f1"] - base["macro_f1"],
            }
    result = {
        "status": "song_native_cue_arm_domain_screen_audited",
        "source_hdf5_sha256": screen["source_hdf5_sha256"],
        "family_screen_sha256": sha256(screen_path),
        "family_predictions_sha256": sha256(predictions),
        "prediction_rows": len(rows), "unique_trials": len(trial_ids),
        "cue_arm_trial_counts": dict(domain_counts),
        "domain_scores": domain_scores,
        "conditional_vs_core": conditional,
        "boundary": "Seven native prompted arm conditions, not independently measured postures. Same participant/date and cued stable trials; arm counts are uneven. Domain scores are post-hoc descriptive checks on frozen held-out probabilities. No new training, cross-day/person conclusion, or live model selection.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "trials": len(trial_ids),
                      "prediction_rows": len(rows), "cue_arm_counts": dict(domain_counts)}),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.screen, args.predictions, args.output)
