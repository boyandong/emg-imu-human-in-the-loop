"""Frozen leave-one-family-out for the independent ring/spectral candidate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2.ring_freq_interaction_run import load_grab, load_roam
from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "RING_FREQ_LOFO_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARENT_PATH = ROOT / "RING_FREQ_INTERACTION_PREDICTIONS.csv"
FULL = "F0v2+ring_lag+frequency_direction"
ARMS = (FULL, "ring_lag+frequency_direction", "F0v2+frequency_direction",
        "F0v2+ring_lag")
STUDIES = ("roam_posture", "grab_user", "grab_day")


def check_protocol() -> None:
    if (PROTOCOL["candidate_full_bank"] != FULL
            or PROTOCOL["removals"] != {"F0v2": ARMS[1], "ring_lag": ARMS[2],
                                           "frequency_direction": ARMS[3]}
            or sha256(ROOT / "RING_FREQ_INTERACTION_PROTOCOL.json") !=
            PROTOCOL["parent_interaction_protocol_sha256"]
            or sha256(PARENT_PATH) != PROTOCOL["parent_interaction_predictions_sha256"]):
        raise ValueError("frozen ring/frequency LOFO parent changed")


def evaluate() -> dict:
    check_protocol()
    with PARENT_PATH.open(newline="", encoding="utf-8") as stream:
        old = list(csv.DictReader(stream))
    if len(old) != 3680:
        raise ValueError("parent prediction inventory changed")
    rows = [{**r, "arm": r["arm"]} for r in old if r["arm"] in (FULL, ARMS[2], ARMS[3])]
    if len(rows) != 2760:
        raise AssertionError("three copied LOFO arms incomplete")
    splits = {}
    for study in STUDIES:
        vectors, labels, identities, masks, rest_count = (load_roam() if study == "roam_posture"
                                                         else load_grab(study))
        x = np.concatenate([vectors["ring_lag"], vectors["frequency_direction"]], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced" if study == "roam_posture" else None,
            max_iter=2000, random_state=20260929 if study == "roam_posture" else 20260924))
        model.fit(x[masks["source"]], labels[masks["source"]])
        classes = np.asarray([0, 1, 2] if study == "roam_posture" else grab.GESTURES)
        np.testing.assert_array_equal(model[-1].classes_, classes)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"LOFO source classifier did not converge: {study}")
        splits[study] = {"source_trials": int(masks["source"].sum()),
                         "source_rest_windows": int(rest_count),
                         "feature_dimensions_without_F0v2": int(x.shape[1])}
        for phase in ("validation", "final"):
            mask = masks[phase]
            probability = model.predict_proba(x[mask])
            parent = {r["trial_id"]: r for r in rows if r["study"] == study and
                      r["phase"] == phase and r["arm"] == FULL}
            if len(parent) != int(mask.sum()):
                raise AssertionError("full-arm native target count changed")
            for identity, label, p in zip(np.asarray(identities, dtype=object)[mask],
                                          labels[mask], probability):
                trial_id = identity["trial_id"] if study == "roam_posture" else identity["stem"]
                if trial_id not in parent or int(parent[trial_id]["label"]) != int(label):
                    raise AssertionError("LOFO target identity/label changed")
                original = parent[trial_id]
                rows.append({"study": study, "arm": ARMS[1], "phase": phase,
                             "trial_id": trial_id, "subject": str(identity["subject"]),
                             "condition": original["condition"], "label": str(int(label)),
                             **{f"p_{c}": str(float(value)) for c, value in zip(classes, p)},
                             **{f"p_{c}": "" for c in ({0, 1, 2, 4, 15, 16, 17} - set(classes))}})
            print(f"LOFO {study} {phase}: fitted no-F0v2 arm on {mask.sum()} native targets",
                  flush=True)
    if len(rows) != 3680:
        raise AssertionError("LOFO four-arm target inventory changed")
    scores = {}
    increments = {}
    for study in STUDIES:
        classes = np.asarray([0, 1, 2] if study == "roam_posture" else grab.GESTURES)
        scores[study], increments[study] = {}, {}
        for phase in ("validation", "final"):
            arm_scores = {}
            for arm in ARMS:
                selected = [r for r in rows if r["study"] == study and
                            r["phase"] == phase and r["arm"] == arm]
                y = np.asarray([int(r["label"]) for r in selected])
                p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in selected])
                subjects = np.asarray([int(r["subject"]) for r in selected])
                arm_scores[arm] = grab.score(y, p, classes, subjects)
                arm_scores[arm]["trials"] = len(y)
                if study == "roam_posture":
                    arm_scores[arm]["by_posture"] = {
                        posture: grab.score(y[np.asarray([r["condition"] == posture
                                                            for r in selected])],
                                            p[np.asarray([r["condition"] == posture
                                                          for r in selected])],
                                            classes, subjects[np.asarray([r["condition"] == posture
                                                                          for r in selected])])
                        for posture in ("resting", "hanging", "unsupported", "reaching")}
            scores[study][phase] = arm_scores
            increments[study][phase] = {family: {
                "full_minus_removed_macro_f1": arm_scores[FULL]["macro_f1"] - arm_scores[arm]["macro_f1"],
                "removed_minus_full_log_loss": arm_scores[arm]["log_loss"] - arm_scores[FULL]["log_loss"],
                "removed_minus_full_brier": arm_scores[arm]["brier"] - arm_scores[FULL]["brier"]}
                for family, arm in PROTOCOL["removals"].items()}
    path = ROOT / "RING_FREQ_LOFO_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_prediction_sha256": sha256(PARENT_PATH),
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "source_splits": splits, "scores": scores, "full_minus_removed": increments,
              "scope": PROTOCOL["scope"]}
    (ROOT / "RING_FREQ_LOFO_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    evaluate()
