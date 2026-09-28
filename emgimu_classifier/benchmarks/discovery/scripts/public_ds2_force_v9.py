"""Fixed three-arm public DS2 v9 force-condition cross-user screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import LocalDetailFamily, ScalePatternFamily
from public_ds2_subject_gesture_incremental import metrics, trial_probabilities


WINDOW_STARTS = (3000, 6000, 9000)
WINDOW_SAMPLES = 375
SPLITS = {"train": tuple(range(1, 13)), "validation": tuple(range(13, 17)),
          "final_descriptive": tuple(range(17, 21))}
MODES = {"unseen_high": (0, 1), "product_all": (0, 1, 2)}
ARMS = {"F0": ("F0",), "F1": ("F1",), "F0_plus_F1": ("F0", "F1")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(raw_path: Path, join_path: Path, label_audit_path: Path,
        protocol_path: Path, output: Path) -> dict:
    classifier_root = Path(__file__).resolve().parents[3]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
    if (sha256(join_path) != audit["trial_join_csv_sha256"] or
            sha256(raw_path) != audit["source_sha256"]["v8_raw_mat"] or
            protocol["source_version"] != 9):
        raise ValueError("public DS2 source or frozen protocol changed")
    for name, subjects in SPLITS.items():
        protocol_key = "final_descriptive" if name == "final_descriptive" else name
        if protocol["split_subject_folders"][protocol_key] != list(subjects):
            raise ValueError("subject split differs from protocol")
    print("[1/4] load hash-joined public trial labels and 3-channel raw windows", flush=True)
    rows = list(csv.DictReader(join_path.open(encoding="utf-8", newline="")))
    eligible = [row for row in rows if row["subject_folder"] != "N/A"
                and row["gesture_code_if_uniform"] != "N/A"
                and int(row["gesture_code_if_uniform"]) in range(4)]
    if len(rows) != 2863 or len(eligible) != 2297:
        raise ValueError(f"eligible active-trial count changed: {len(eligible)}")
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    if raw.shape != (2863, 3, 15000):
        raise ValueError("public raw-MAT shape changed")
    trial_ids = np.array([int(row["raw_trial_index_zero_based"]) for row in eligible], dtype=np.int32)
    subjects = np.array([int(row["subject_folder"]) for row in eligible], dtype=np.int8)
    gestures = np.array([int(row["gesture_code_if_uniform"]) for row in eligible], dtype=np.int8)
    forces = np.array([int(row["force_code"]) for row in eligible], dtype=np.int8)
    windows = np.empty((len(eligible) * 3, WINDOW_SAMPLES, 3), dtype=np.float32)
    for index, trial in enumerate(trial_ids):
        for window, start in enumerate(WINDOW_STARTS):
            windows[index * 3 + window] = raw[trial, :, start:start + WINDOW_SAMPLES].T
    del raw
    window_trials = np.repeat(trial_ids, 3)
    window_subjects = np.repeat(subjects, 3)
    window_gestures = np.repeat(gestures, 3)
    window_forces = np.repeat(forces, 3)
    masks = {name: np.isin(window_subjects, ids) for name, ids in SPLITS.items()}
    if not np.all(sum(mask.astype(np.int8) for mask in masks.values()) == 1):
        raise ValueError("subject partitions overlap or miss trials")
    for name, mask in masks.items():
        if (set(np.unique(window_gestures[mask])) != set(range(4)) or
                set(np.unique(window_forces[mask])) != set(range(3))):
            raise ValueError(f"{name} lacks gesture or force condition")

    print("[2/4] train only on source subjects and permitted source force levels", flush=True)
    classes = np.arange(4)
    all_scores, by_condition, prediction_rows = {}, {}, []
    for mode, allowed_forces in MODES.items():
        train_mask = masks["train"] & np.isin(window_forces, allowed_forces)
        train_batch = FeatureBatch(windows[train_mask], 1500.0)
        families = {"F0": LocalDetailFamily(), "F1": ScalePatternFamily()}
        features = {}
        for family_name, family in families.items():
            family.fit(train_batch, window_gestures[train_mask])
            features[family_name] = {
                "train": family.transform(train_batch),
                **{split: family.transform(FeatureBatch(windows[masks[split]], 1500.0))
                   for split in ("validation", "final_descriptive")},
            }
        all_scores[mode], by_condition[mode] = {}, {}
        for arm, names in ARMS.items():
            train_x = np.concatenate([features[name]["train"] for name in names], axis=1)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=20260928))
            model.fit(train_x, window_gestures[train_mask])
            if not np.array_equal(model[-1].classes_, classes):
                raise ValueError("unexpected trained gesture class order")
            all_scores[mode][arm], by_condition[mode][arm] = {}, {}
            for split in ("validation", "final_descriptive"):
                mask = masks[split]
                x = np.concatenate([features[name][split] for name in names], axis=1)
                ids, truth, probability = trial_probabilities(
                    model.predict_proba(x), window_trials[mask], window_gestures[mask])
                expected_ids = trial_ids[np.isin(subjects, SPLITS[split])]
                if not np.array_equal(ids, expected_ids):
                    raise ValueError("trial identity or order changed")
                subject_by_id = dict(zip(trial_ids, subjects))
                force_by_id = dict(zip(trial_ids, forces))
                force_at_trial = np.array([force_by_id[int(trial)] for trial in ids])
                all_scores[mode][arm][split] = metrics(truth, probability, classes)
                by_condition[mode][arm][split] = {
                    str(force): metrics(truth[force_at_trial == force],
                                        probability[force_at_trial == force], classes)
                    for force in range(3)}
                for trial, gesture, force, probs in zip(ids, truth, force_at_trial, probability):
                    prediction_rows.append({
                        "training_mode": mode, "arm": arm, "split": split,
                        "raw_trial_index_zero_based": int(trial),
                        "subject_folder": int(subject_by_id[int(trial)]),
                        "gesture_code": int(gesture), "force_code": int(force),
                        **{f"p{code}": float(probs[code]) for code in classes},
                    })
            print(f"  {mode}/{arm}: held-out trials scored", flush=True)

    print("[3/4] save all paired trial probabilities and condition metrics", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "TRIAL_PREDICTIONS.csv"
    with predictions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    result = {
        "status": "public_ds2_v9_subjective_force_descriptive_cross_user_screen",
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": {"raw_mat": sha256(raw_path), "trial_join": sha256(join_path),
                          "force_label_audit": sha256(label_audit_path),
                          "runner": sha256(Path(__file__)),
                          "feature_families": sha256(classifier_root / "src/emgimu/feature_bank/families.py"),
                          "feature_batch": sha256(classifier_root / "src/emgimu/feature_bank/core.py")},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "scikit-learn")},
        "eligible_active_trials": len(eligible),
        "split_subject_folders": {name: list(ids) for name, ids in SPLITS.items()},
        "training_modes": {name: list(levels) for name, levels in MODES.items()},
        "arms": {name: list(families) for name, families in ARMS.items()},
        "window_starts": list(WINDOW_STARTS), "window_samples": WINDOW_SAMPLES,
        "scores": all_scores, "by_force_code": by_condition,
        "trial_predictions": predictions.name,
        "trial_predictions_sha256": sha256(predictions),
        "boundary": "Publisher v9 subjective force codes, only four active gesture codes, 3-channel public raw data. Models fit on source subjects only. No target calibration. Final subjects had previously been inspected in gesture-only work, so final results are descriptive. No mechanical-force measurement, historical X1-H equivalence, own-device effect, or deployment claim.",
    }
    (output / "RESULTS.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print("[4/4] fixed study complete", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mat", required=True, type=Path)
    parser.add_argument("--trial-join", required=True, type=Path)
    parser.add_argument("--force-audit", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "public_ds2_force_v9/PROTOCOL.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "public_ds2_force_v9")
    args = parser.parse_args()
    run(args.raw_mat, args.trial_join, args.force_audit, args.protocol, args.output)


if __name__ == "__main__":
    main()
