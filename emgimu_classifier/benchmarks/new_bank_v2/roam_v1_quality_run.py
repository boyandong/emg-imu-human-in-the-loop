"""Source-frozen new-v1 family screen on the ROAM synthetic-quality grid."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday.run import score
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from benchmarks.new_bank_v2.roam_quality_run import aggregate_families, fault, packed
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "ROAM_V1_QUALITY_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PREFIX = "ROAM_V1_QUALITY"
CLASSES = np.asarray(PROTOCOL["classes"])
ARMS = tuple(PROTOCOL["arms"])


def check_protocol() -> dict:
    for filename, key in (("ROAM_QUALITY_PROTOCOL.json", "parent_protocol_sha256"),
                          ("ROAM_QUALITY_RESULTS.json", "parent_result_sha256"),
                          ("ROAM_QUALITY_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / filename) != PROTOCOL[key]:
            raise AssertionError(f"frozen quality parent changed: {filename}")
    parent = json.loads((ROOT / "ROAM_QUALITY_PROTOCOL.json").read_text(encoding="utf-8"))
    if (PROTOCOL["source_subjects"] != parent["source_subjects"]
            or PROTOCOL["validation_subjects"] != parent["validation_subjects"]
            or PROTOCOL["final_subjects"] != parent["final_subjects"]
            or PROTOCOL["conditions"] != parent["conditions"]
            or PROTOCOL["sample_rate_hz"] != parent["sample_rate_hz"]
            or PROTOCOL["window_samples"] != parent["window_samples"]
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise AssertionError("new-v1 quality protocol does not match frozen parent")
    return parent


def replay_parent(rows: list[dict]) -> float:
    with (ROOT / "ROAM_QUALITY_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent = {(r["phase"], r["condition"], r["trial_id"]): r
                  for r in csv.DictReader(stream) if r["arm"] == "F0v2"}
    base = [r for r in rows if r["arm"] == "F0v2"]
    if len(base) != 1260 or len(parent) != 1260:
        raise AssertionError("quality F0v2 parent coverage changed")
    maximum = 0.0
    for row in base:
        prior = parent[(row["phase"], row["condition"], row["trial_id"])]
        if (row["subject"], row["label"]) != (int(prior["subject"]), int(prior["label"])):
            raise AssertionError("quality parent native identity changed")
        p = np.asarray([row[f"p_{c}"] for c in CLASSES])
        q = np.asarray([float(prior[f"p_{c}"]) for c in CLASSES])
        maximum = max(maximum, float(np.max(np.abs(p - q))))
    if maximum > 1e-10:
        raise AssertionError(f"quality baseline replay changed: {maximum}")
    return maximum


def summarize(cells: dict) -> dict:
    faults = PROTOCOL["conditions"][1:]
    coordinates = {"dropout_mean": float(np.mean([cells[f"dropout_ch{i}"]["macro_f1"]
                                                  for i in range(8)])),
                   **{name: cells[name]["macro_f1"] for name in faults[8:]}}
    return {"clean_macro_f1": cells["clean"]["macro_f1"],
            "family_macro_f1": coordinates,
            "synthetic_quality_mean_macro_f1": float(np.mean(list(coordinates.values()))),
            "synthetic_quality_min_macro_f1": float(min(coordinates.values())),
            "minimum_named_fault_macro_f1": float(min(cells[name]["macro_f1"] for name in faults)),
            "minimum_subject_fault_macro_f1": float(min(
                value for name in faults for value in cells[name]["per_subject_macro_f1"].values())),
            "mean_named_fault_log_loss": float(np.mean([cells[name]["log_loss"] for name in faults]))}


def evaluate() -> dict:
    parent = check_protocol()
    parent_result = json.loads((ROOT / "ROAM_QUALITY_RESULTS.json").read_text(encoding="utf-8"))
    archive = Path(parent["archive"])
    if sha256(archive).lower() != parent["archive_sha256"].lower():
        raise AssertionError("ROAM archive changed")
    batch, labels, identities, slices, counts = extract_archive(archive)
    users = np.asarray([r["subject"] for r in identities])
    postures = np.asarray([r["condition"] for r in identities])
    source_indices = np.flatnonzero(np.isin(users, PROTOCOL["source_subjects"])
                                    & (postures == PROTOCOL["posture"]))
    source_windows, source_slices = packed(batch, slices, source_indices)
    rest = [source_windows[a:b] for j, (a, b) in enumerate(source_slices)
            if labels[source_indices[j]] == 0]
    rest_batch = FeatureBatch(np.concatenate(rest), 200)
    if len(source_indices) != 162 or rest_batch.windows != 1828:
        raise AssertionError("ROAM quality source split changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                **{name: constructor().fit(rest_batch) for name, constructor in NEW_BANK_V1.items()}}
    source_vectors = aggregate_families(source_windows, source_slices, families)
    q95 = np.quantile(np.abs(source_windows.astype(np.float64)), 0.95, axis=(0, 1)).astype(np.float32)
    rms = np.sqrt(np.mean(source_windows.astype(np.float64) ** 2, axis=(0, 1))).astype(np.float32)
    np.testing.assert_allclose(q95, parent_result["source_reference_q95"], rtol=0, atol=0)
    np.testing.assert_allclose(rms, parent_result["source_reference_rms"], rtol=0, atol=0)
    models = {}
    for arm in ARMS:
        x = np.concatenate([source_vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260929))
        model.fit(x, labels[source_indices])
        np.testing.assert_array_equal(model[-1].classes_, CLASSES)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"source quality classifier did not converge: {arm}")
        models[arm] = model
    rows = []
    scores = {phase: {arm: {} for arm in ARMS} for phase in ("validation", "final")}
    split_ids = {}
    for phase in ("validation", "final"):
        indices = np.flatnonzero(np.isin(users, PROTOCOL[f"{phase}_subjects"])
                                     & (postures == PROTOCOL["posture"]))
        trial_ids = [identities[int(i)]["trial_id"] for i in indices]
        if (len(indices) != 45 or len(set(trial_ids)) != 45
                or trial_ids != parent_result[f"{phase}_trial_ids"]):
            raise AssertionError("ROAM held native bouts changed")
        split_ids[phase] = trial_ids
        target_windows, target_slices = packed(batch, slices, indices)
        for condition in PROTOCOL["conditions"]:
            transformed = aggregate_families(fault(target_windows, condition, q95, rms),
                                             target_slices, families)
            for arm in ARMS:
                xt = np.concatenate([transformed[name] for name in arm.split("+")], axis=1)
                probability = models[arm].predict_proba(xt)
                scores[phase][arm][condition] = score(labels[indices], probability, CLASSES,
                                                       users[indices])
                for i, p in zip(indices, probability):
                    identity = identities[int(i)]
                    rows.append({"phase": phase, "condition": condition, "arm": arm,
                                 "trial_id": identity["trial_id"],
                                 "subject": int(identity["subject"]),
                                 "label": int(identity["label"]),
                                 **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
            print(f"ROAM new-v1 quality {phase} {condition}", flush=True)
    replay_error = replay_parent(rows)
    summary = {phase: {arm: summarize(scores[phase][arm]) for arm in ARMS}
               for phase in ("validation", "final")}
    chosen = min(ARMS, key=lambda arm: (
        -summary["validation"][arm]["synthetic_quality_mean_macro_f1"],
        -summary["validation"][arm]["minimum_named_fault_macro_f1"],
        summary["validation"][arm]["mean_named_fault_log_loss"]))
    path = ROOT / f"{PREFIX}_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(archive), "prediction_sha256": sha256(path),
              "source_trial_ids": [identities[int(i)]["trial_id"] for i in source_indices],
              "split_trial_ids": split_ids, "source_rest_windows": rest_batch.windows,
              "source_reference_q95": q95.tolist(), "source_reference_rms": rms.tolist(),
              "native_static_files_verified": len(counts),
              "feature_dimensions": {name: int(v.shape[1]) for name, v in source_vectors.items()},
              "baseline_replay_max_abs_error": replay_error,
              "validation_selected_arm": chosen, "scores": scores, "summary": summary,
              "scope": PROTOCOL["scope"]}
    (ROOT / f"{PREFIX}_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"ROAM new-v1 quality validation-selected arm: {chosen}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
