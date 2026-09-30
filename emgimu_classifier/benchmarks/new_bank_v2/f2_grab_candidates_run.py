"""Source-only F2b/F2c on the frozen GRABMyo unseen-user Day1 trial split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2 import grab_user_run as parent_run
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_signal import DocumentCspFamily
from emgimu.feature_bank.families import SpdTangentFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2_GRAB_CANDIDATES_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT = parent_run.PROTOCOL
CLASSES = np.asarray(grab.GESTURES)


def evaluate(data_root: Path) -> dict:
    parent_run.check_protocol()
    for filename, key in (("GRAB_USER_PROTOCOL.json", "parent_protocol_sha256"),
                          ("GRAB_USER_RESULTS.json", "parent_result_sha256"),
                          ("GRAB_USER_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / filename) != PROTOCOL[key]:
            raise AssertionError(f"frozen GRAB parent changed: {filename}")
    if PROTOCOL["arms"] != ["F0v2", "F0v2+F2b_document", "F0v2+F2c_spd"]:
        raise ValueError("F2 candidate arms changed")
    parent = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
    records = [row for row in grab.records() if row["session"] == PARENT["day"]]
    if len(records) != 224:
        raise AssertionError("GRAB Day1 native inventory changed")
    manifest_sha = parent_run.check_files(data_root, records)
    if manifest_sha != parent["official_sha256_manifest_sha256"]:
        raise AssertionError("GRAB source hash manifest differs from frozen parent")
    windows = [grab.read_record(data_root, row) for row in records]
    subjects = np.asarray([row["subject"] for row in records])
    labels = np.asarray([row["gesture"] for row in records])
    source_mask = np.isin(subjects, PARENT["source_subjects"])
    source_windows = np.concatenate([window for window, selected in zip(windows, source_mask) if selected])
    source_labels = np.concatenate([np.full(len(window), label) for window, label, selected
                                    in zip(windows, labels, source_mask) if selected])
    rest_windows = np.concatenate([window for window, selected, label in zip(windows, source_mask, labels)
                                   if selected and label == PARENT["rest_code"]])
    if source_mask.sum() != 112 or len(rest_windows) != 560:
        raise AssertionError("GRAB source/Rest inventory changed")
    rest = FeatureBatch(rest_windows, grab.RATE)
    source = FeatureBatch(source_windows, grab.RATE)
    families = {
        "F0v2": RestNoiseDetailV2(rest_label=17).fit(rest, np.full(rest.windows, 17)),
        "F2b_document": DocumentCspFamily().fit(source, source_labels),
        "F2c_spd": SpdTangentFamily(shrinkage=0.05).fit(source),
    }
    vectors = {name: np.stack([grab.aggregate(family.transform(FeatureBatch(window, grab.RATE)))
                                for window in windows]) for name, family in families.items()}
    rows, scores = [], {}
    for arm in PROTOCOL["arms"]:
        feature = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(feature[source_mask], labels[source_mask])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"GRAB source classifier did not converge: {arm}")
        scores[arm] = {"train_trials": int(source_mask.sum()),
                       "feature_dimensions": int(feature.shape[1])}
        for phase in ("validation", "final"):
            target = np.isin(subjects, PARENT[f"{phase}_subjects"])
            if target.sum() != 56 or np.any(target & source_mask):
                raise AssertionError("GRAB unseen-user target split changed")
            probabilities = model.predict_proba(feature[target])
            scores[arm][phase] = {"trials": int(target.sum()),
                                  **grab.score(labels[target], probabilities, CLASSES, subjects[target])}
            for row, probability in zip(np.asarray(records, dtype=object)[target], probabilities):
                rows.append({"arm": arm, "phase": phase, "trial_id": row["stem"],
                             "subject": row["subject"], "gesture": row["gesture"],
                             **{f"p_{label}": float(value) for label, value in zip(CLASSES, probability)}})
        print(f"GRAB F2 {arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    with (ROOT / "GRAB_USER_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                    if row["arm"] == "F0v2"}
    baseline = [row for row in rows if row["arm"] == "F0v2"]
    if len(previous) != len(baseline) or len(baseline) != 112:
        raise AssertionError("GRAB parent/new baseline coverage changed")
    maximum = 0.
    for row in baseline:
        old = previous[(row["phase"], row["trial_id"])]
        if (row["subject"], row["gesture"]) != (int(old["subject"]), int(old["gesture"])):
            raise AssertionError("GRAB parent/new native trial identity changed")
        maximum = max(maximum, max(abs(row[f"p_{label}"] - float(old[f"p_{label}"]))
                                   for label in CLASSES))
    if maximum > 1e-8:
        raise AssertionError(f"GRAB F0v2 parent replay changed: {maximum}")
    path = ROOT / "F2_GRAB_CANDIDATES_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": PROTOCOL["parent_result_sha256"],
              "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
              "official_sha256_manifest_sha256": manifest_sha,
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "source_windows": int(source.windows), "rest_windows": int(rest.windows),
              "feature_dimensions": {name: len(family.feature_names) for name, family in families.items()},
              "source_trial_ids": parent["source_trial_ids"],
              "validation_trial_ids": parent["validation_trial_ids"],
              "final_trial_ids": parent["final_trial_ids"],
              "f0v2_parent_replay_max_abs_error": maximum, "scores": scores,
              "scope": PROTOCOL["boundary"]}
    (ROOT / "F2_GRAB_CANDIDATES_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                              encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    evaluate(parser.parse_args().data_root)
