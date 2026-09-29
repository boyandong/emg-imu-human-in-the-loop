"""Matched source-only document CSP increment under native wearing shifts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2 import wearing_v1_extension_run as parent_run
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.document_signal import DocumentCspFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2B_WEARING_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.asarray(PROTOCOL["classes"])


def evaluate() -> dict:
    parent_run.check_protocol()
    if (sha256(ROOT / "WEARING_V1_EXTENSION_RESULTS.json") != PROTOCOL["parent_result_sha256"]
            or sha256(ROOT / "WEARING_V1_EXTENSION_PREDICTIONS.csv") != PROTOCOL["parent_prediction_sha256"]):
        raise AssertionError("wearing new-v1 parent changed")
    parent = json.loads((ROOT / "WEARING_V1_EXTENSION_RESULTS.json").read_text(encoding="utf-8"))
    for key in ("source_domain", "target_domains", "validation_subjects", "final_subjects", "classes"):
        if PROTOCOL[key] != parent["protocol"][key]:
            raise AssertionError(f"document CSP split differs from frozen wearing parent: {key}")
    rows, splits = [], {}
    for phase in ("validation", "final"):
        for subject in PROTOCOL[f"{phase}_subjects"]:
            source = parent_run.load(parent_run.ARCHIVE, subject,
                                     (PROTOCOL["source_domain"],))
            target = parent_run.load(parent_run.ARCHIVE, subject,
                                     tuple(PROTOCOL["target_domains"]))
            if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
                    or source.batch.emg.shape[1] != 40 or set(source.labels) != set(CLASSES)):
                raise AssertionError("document CSP source contract changed")
            families = {"F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                        "F2b_document": DocumentCspFamily().fit(source.batch, source.labels)}
            xs, ys, source_users, source_trials = parent_run.vectors(source, families)
            xt, yt, target_users, target_trials = parent_run.vectors(target, families)
            key = f"{phase}_{subject}"
            if (set(source_trials) != set(parent["split_trial_ids"][key]["source"])
                    or set(target_trials) != set(parent["split_trial_ids"][key]["target"])
                    or set(source_trials) & set(target_trials)
                    or set(source_users) != {subject} or set(target_users) != {subject}):
                raise AssertionError("document CSP native trial split changed")
            splits[key] = {"source": source_trials.tolist(), "target": target_trials.tolist()}
            for arm in PROTOCOL["arms"]:
                names = arm.split("+")
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    random_state=20260924))
                model.fit(np.concatenate([xs[name] for name in names], axis=1), ys)
                np.testing.assert_array_equal(model[-1].classes_, CLASSES)
                if np.max(model[-1].n_iter_) >= 2000:
                    raise RuntimeError(f"document CSP classifier did not converge: {key}/{arm}")
                probabilities = model.predict_proba(
                    np.concatenate([xt[name] for name in names], axis=1))
                for trial, label, probability in zip(target_trials, yt, probabilities):
                    match = PATH_RE.fullmatch(str(trial))
                    if match is None or int(match["subject"]) != subject or int(match["label"]) != label:
                        raise AssertionError("document CSP native target identity changed")
                    rows.append({"phase": phase, "subject": subject,
                                 "domain": match["domain"], "arm": arm,
                                 "trial_id": str(trial), "label": int(label),
                                 **{f"p_{c}": float(value)
                                    for c, value in zip(CLASSES, probability)}})
            print(f"document CSP wearing {key}: {len(target_trials)} held trials", flush=True)
    with (ROOT / "WEARING_V1_EXTENSION_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                    if r["arm"] == "F0v2"}
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    if len(previous) != 240 or len(baseline) != 240:
        raise AssertionError("document CSP baseline coverage changed")
    maximum = 0.0
    for row in baseline:
        old = previous[(row["phase"], row["trial_id"])]
        if (row["subject"], row["domain"], row["label"]) != (
                int(old["subject"]), old["domain"], int(old["label"])):
            raise AssertionError("document CSP F0v2 identity changed")
        maximum = max(maximum, max(abs(row[f"p_{c}"] - float(old[f"p_{c}"]))
                                   for c in CLASSES))
    if maximum > 1e-8:
        raise AssertionError(f"document CSP F0v2 replay changed: {maximum}")
    scores = {phase: {arm: parent_run._score(rows, phase, arm)
                      for arm in PROTOCOL["arms"]} for phase in ("validation", "final")}
    path = ROOT / "F2B_WEARING_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": PROTOCOL["parent_result_sha256"],
              "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
              "archive_sha256": sha256(parent_run.ARCHIVE),
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "split_trial_ids": splits,
              "feature_dimensions": {name: int(xs[name].shape[1]) for name in families},
              "f0v2_parent_replay_max_abs_error": maximum,
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / "F2B_WEARING_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                       encoding="utf-8")
    print("document CSP wearing validation delta F1: "
          f"{scores['validation']['F0v2+F2b_document']['pooled']['macro_f1'] - scores['validation']['F0v2']['pooled']['macro_f1']:+.4f}",
          flush=True)
    return result


if __name__ == "__main__":
    evaluate()
