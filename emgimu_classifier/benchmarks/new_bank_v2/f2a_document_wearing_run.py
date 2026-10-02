"""Frozen native wearing test of the historical uncentered F2a alternative."""
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
from emgimu.feature_bank.new_bank_v2 import DocumentTraceCovarianceV2, RestNoiseDetailV2


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2A_DOCUMENT_WEARING_PROTOCOL.json"
PREDICTIONS = ROOT / "F2A_DOCUMENT_WEARING_PREDICTIONS.csv"
RESULT = ROOT / "F2A_DOCUMENT_WEARING_RESULTS.json"


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (sha256(ROOT / "F2_AC_WEARING_RESULTS.json") != protocol["parent_result_sha256"] or
            sha256(ROOT / "F2_AC_WEARING_PREDICTIONS.csv") != protocol["parent_prediction_sha256"] or
            sha256(wearing.ARCHIVE) != protocol["archive_sha256"]):
        raise AssertionError("Frozen centered-F2a parent or native archive changed")
    wearing.check_protocol()
    parent = json.loads((ROOT / "F2_AC_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    parent_protocol = json.loads((ROOT / "F2B_WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    if (protocol["subjects"]["validation"] != parent_protocol["validation_subjects"] or
            protocol["subjects"]["final"] != parent_protocol["final_subjects"] or
            protocol["source_domain"] != parent_protocol["source_domain"] or
            protocol["target_domains"] != parent_protocol["target_domains"] or
            protocol["classes"] != parent_protocol["classes"]):
        raise AssertionError("Native wearing partition changed")
    classes = np.asarray(protocol["classes"])
    rows, partitions = [], {}
    for phase in ("validation", "final"):
        for subject in protocol["subjects"][phase]:
            source = wearing.load(wearing.ARCHIVE, subject, (protocol["source_domain"],))
            target = wearing.load(wearing.ARCHIVE, subject, tuple(protocol["target_domains"]))
            if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200 or
                    source.batch.emg.shape[1] != 40 or set(source.labels) != set(classes)):
                raise AssertionError("Native source contract changed")
            families = {
                "F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                "F2a_uncentered": DocumentTraceCovarianceV2(shrinkage=.05).fit(source.batch),
            }
            xs, ys, source_users, source_trials = wearing.vectors(source, families)
            xt, yt, target_users, target_trials = wearing.vectors(target, families)
            key = f"{phase}_{subject}"
            expected = parent["split_trial_ids"][key]
            if (set(source_trials) != set(expected["source"]) or
                    set(target_trials) != set(expected["target"]) or
                    set(source_trials) & set(target_trials) or
                    set(source_users) != {subject} or set(target_users) != {subject}):
                raise AssertionError("Source/target native trial identities changed")
            partitions[key] = {"source": source_trials.tolist(), "target": target_trials.tolist()}
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1., class_weight="balanced", max_iter=2000, random_state=20260924))
            model.fit(np.concatenate((xs["F0v2"], xs["F2a_uncentered"]), axis=1), ys)
            np.testing.assert_array_equal(model[-1].classes_, classes)
            if np.max(model[-1].n_iter_) >= 2000:
                raise RuntimeError(f"Uncentered F2a classifier did not converge: {key}")
            probabilities = model.predict_proba(
                np.concatenate((xt["F0v2"], xt["F2a_uncentered"]), axis=1))
            for trial, label, probability in zip(target_trials, yt, probabilities):
                identity = PATH_RE.fullmatch(str(trial))
                if identity is None or int(identity["subject"]) != subject or int(identity["label"]) != label:
                    raise AssertionError("Native trial label/identity changed")
                rows.append({"phase": phase, "subject": subject,
                             "domain": identity["domain"], "arm": protocol["candidate"],
                             "trial_id": str(trial), "label": int(label),
                             **{f"p_{c}": float(p) for c, p in zip(classes, probability)}})
            print(f"F2a document {key}: {len(target_trials)} held-out trials", flush=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    scores = {phase: wearing._score(rows, phase, protocol["candidate"])
              for phase in ("validation", "final")}
    result = {"status": "document_uncentered_f2a_native_wearing",
              "protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": protocol["parent_result_sha256"],
              "parent_prediction_sha256": protocol["parent_prediction_sha256"],
              "archive_sha256": protocol["archive_sha256"],
              "prediction_sha256": sha256(PREDICTIONS),
              "prediction_rows": len(rows),
              "feature_dimensions": {"F0v2": 48, "F2a_uncentered": 36},
              "split_trial_ids": partitions,
              "scores": scores,
              "scope": protocol["boundary"]}
    RESULT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for phase in ("validation", "final"):
        score = scores[phase]["pooled"]
        print(f"F2a document {phase}: F1={score['macro_f1']:.4f} loss={score['log_loss']:.4f}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
