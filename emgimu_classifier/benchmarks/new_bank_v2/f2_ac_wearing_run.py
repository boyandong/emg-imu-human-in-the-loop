"""Source-only trace-covariance and SPD increments on frozen wearing trials."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.families import SpdTangentFamily
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, TraceCovarianceV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2_AC_WEARING_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def evaluate() -> dict:
    for suffix, key in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("RESULTS.json", "parent_result_sha256"),
                        ("PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / f"F2B_WEARING_{suffix}") != PROTOCOL[key]:
            raise AssertionError(f"frozen F2b wearing parent changed: {suffix}")
    wearing.check_protocol()
    parent = json.loads((ROOT / "F2B_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "F2B_WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    classes = np.asarray(original["classes"])
    rows, splits, dimensions = [], {}, {}
    for phase in ("validation", "final"):
        for subject in original[f"{phase}_subjects"]:
            source = wearing.load(wearing.ARCHIVE, subject, (original["source_domain"],))
            target = wearing.load(wearing.ARCHIVE, subject, tuple(original["target_domains"]))
            if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
                    or source.batch.emg.shape[1] != 40 or set(source.labels) != set(classes)):
                raise AssertionError("F2 wearing source contract changed")
            families = {"F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                        "F2a_trace": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
                        "F2c_spd": SpdTangentFamily(shrinkage=0.05).fit(source.batch)}
            xs, ys, source_users, source_trials = wearing.vectors(source, families)
            xt, yt, target_users, target_trials = wearing.vectors(target, families)
            key = f"{phase}_{subject}"
            if (set(source_trials) != set(parent["split_trial_ids"][key]["source"])
                    or set(target_trials) != set(parent["split_trial_ids"][key]["target"])
                    or set(source_trials) & set(target_trials)
                    or set(source_users) != {subject} or set(target_users) != {subject}):
                raise AssertionError("F2 wearing native split changed")
            splits[key] = {"source": source_trials.tolist(), "target": target_trials.tolist()}
            for name, values in xs.items():
                if name in dimensions and dimensions[name] != values.shape[1]:
                    raise AssertionError("F2 family dimension changed by subject")
                dimensions[name] = int(values.shape[1])
            for arm in PROTOCOL["arms"]:
                names = arm.split("+")
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    random_state=20260924))
                model.fit(np.concatenate([xs[name] for name in names], axis=1), ys)
                np.testing.assert_array_equal(model[-1].classes_, classes)
                if np.max(model[-1].n_iter_) >= 2000:
                    raise RuntimeError(f"F2 wearing classifier did not converge: {key}/{arm}")
                probabilities = model.predict_proba(
                    np.concatenate([xt[name] for name in names], axis=1))
                for trial, label, probability in zip(target_trials, yt, probabilities):
                    match = PATH_RE.fullmatch(str(trial))
                    if match is None or int(match["subject"]) != subject or int(match["label"]) != label:
                        raise AssertionError("F2 wearing target identity changed")
                    rows.append({"phase": phase, "subject": subject,
                                 "domain": match["domain"], "arm": arm,
                                 "trial_id": str(trial), "label": int(label),
                                 **{f"p_{c}": float(value)
                                    for c, value in zip(classes, probability)}})
            print(f"F2 wearing {key}: {len(target_trials)} held trials", flush=True)
    with (ROOT / "F2B_WEARING_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent_rows = list(csv.DictReader(stream))
    parent_keys = {(r["phase"], r["trial_id"], r["arm"]) for r in parent_rows}
    if len(parent_keys) != 480 or len(rows) != 480:
        raise AssertionError("F2 wearing parent/new target inventory changed")
    for phase in ("validation", "final"):
        expected = {r["trial_id"] for r in parent_rows
                    if r["phase"] == phase and r["arm"] == "F0v2"}
        for arm in PROTOCOL["arms"]:
            if {r["trial_id"] for r in rows if r["phase"] == phase and r["arm"] == arm} != expected:
                raise AssertionError("F2 wearing matched trial identities changed")
    scores = {phase: {arm: wearing._score(rows, phase, arm)
                      for arm in PROTOCOL["arms"]} for phase in ("validation", "final")}
    path = ROOT / "F2_AC_WEARING_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": PROTOCOL["parent_result_sha256"],
              "parent_prediction_sha256": PROTOCOL["parent_prediction_sha256"],
              "archive_sha256": sha256(wearing.ARCHIVE),
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "split_trial_ids": splits, "feature_dimensions": dimensions,
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / "F2_AC_WEARING_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                         encoding="utf-8")
    for arm in PROTOCOL["arms"]:
        print(f"F2 wearing {arm}: validation F1={scores['validation'][arm]['pooled']['macro_f1']:.4f}, "
              f"final F1={scores['final'][arm]['pooled']['macro_f1']:.4f}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
