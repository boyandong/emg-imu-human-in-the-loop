"""Falsify an anatomical reading of F3c with fixed cyclic-index counterfactuals."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import DocumentRingRelativeCovarianceV2, RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F3C_CHANNEL_ORDER_PROTOCOL.json"
PREDICTIONS = ROOT / "F3C_CHANNEL_ORDER_PREDICTIONS.csv"
RESULT = ROOT / "F3C_CHANNEL_ORDER_RESULTS.json"


def orders(seed: int, count: int) -> dict[str, list[int]]:
    identity = np.arange(8)
    dihedral = {tuple(np.roll(identity, k)) for k in range(8)}
    dihedral |= {tuple(np.roll(identity[::-1], k)) for k in range(8)}
    rng = np.random.default_rng(seed)
    chosen = {"native_order": identity.tolist()}
    used = set(dihedral)
    while len(chosen) <= count:
        order = tuple(int(v) for v in rng.permutation(8))
        if order in used:
            continue
        order_array = np.asarray(order)
        used.update(tuple(np.roll(order_array, k)) for k in range(8))
        used.update(tuple(np.roll(order_array[::-1], k)) for k in range(8))
        chosen[f"shuffle_{len(chosen):02d}"] = list(order)
    return chosen


def reordered(data, order: list[int]):
    batch = FeatureBatch(data.batch.emg[:, :, order], data.batch.sample_rate_hz)
    return replace(data, batch=batch)


def evaluate() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (sha256(ROOT / "DOCUMENT_SPATIAL_WEARING_RESULTS.json") != protocol["parent_result_sha256"]
            or sha256(ROOT / "DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv") != protocol["parent_prediction_sha256"]
            or sha256(wearing.ARCHIVE) != protocol["archive_sha256"]):
        raise AssertionError("frozen document F3c wearing parent changed")
    wearing.check_protocol()
    parent = json.loads((ROOT / "DOCUMENT_SPATIAL_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    root_protocol = json.loads((ROOT / "F2A_DOCUMENT_WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    variants = orders(protocol["random_seed"], protocol["random_permutations"])
    classes = np.asarray(root_protocol["classes"])
    with (ROOT / "DOCUMENT_SPATIAL_WEARING_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        original = {(row["phase"], row["trial_id"]): row for row in csv.DictReader(stream)
                    if row["arm"] == "F0v2+F3c_document"}
    rows = []
    maximum_identity_replay_error = 0.0
    for phase in ("validation", "final"):
        for subject in root_protocol["subjects"][phase]:
            source = wearing.load(wearing.ARCHIVE, subject, (root_protocol["source_domain"],))
            target = wearing.load(wearing.ARCHIVE, subject, tuple(root_protocol["target_domains"]))
            baseline = RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels)
            xs_f0, ys, _, source_trials = aggregate(baseline.transform(source.batch), source)
            xt_f0, yt, _, target_trials = aggregate(baseline.transform(target.batch), target)
            frozen = parent["split_trial_ids"][f"{phase}_{subject}"]
            if (set(source_trials) != set(frozen["source"]) or
                    set(target_trials) != set(frozen["target"]) or
                    set(source_trials) & set(target_trials)):
                raise AssertionError("wearing source/target trial split changed")
            for name, order in variants.items():
                source_ordered, target_ordered = reordered(source, order), reordered(target, order)
                # Nonphysical orders deliberately evaluate the same cyclic-index statistic.
                family = DocumentRingRelativeCovarianceV2(ring_topology=True, shrinkage=.05).fit(
                    source_ordered.batch)
                xs, source_y, _, f3_source_trials = aggregate(
                    family.transform(source_ordered.batch), source_ordered)
                xt, target_y, _, f3_target_trials = aggregate(
                    family.transform(target_ordered.batch), target_ordered)
                np.testing.assert_array_equal(source_y, ys)
                np.testing.assert_array_equal(target_y, yt)
                np.testing.assert_array_equal(f3_source_trials, source_trials)
                np.testing.assert_array_equal(f3_target_trials, target_trials)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1., class_weight="balanced", max_iter=2000, random_state=20260924))
                model.fit(np.concatenate((xs_f0, xs), axis=1), ys)
                np.testing.assert_array_equal(model[-1].classes_, classes)
                if np.max(model[-1].n_iter_) >= 2000:
                    raise RuntimeError(f"F3c order classifier failed to converge: {phase}/{subject}/{name}")
                probability = model.predict_proba(np.concatenate((xt_f0, xt), axis=1))
                for trial, label, p in zip(target_trials, yt, probability):
                    domain = str(trial).split("/")[-2]
                    rows.append({"phase": phase, "subject": subject, "domain": domain,
                                 "arm": name, "trial_id": str(trial), "label": int(label),
                                 **{f"p_{c}": float(value) for c, value in zip(classes, p)}})
                    if name == "native_order":
                        prior = original[(phase, str(trial))]
                        maximum_identity_replay_error = max(maximum_identity_replay_error,
                            float(np.max(np.abs(p - np.asarray(
                                [float(prior[f"p_{c}"]) for c in classes])))))
            print(f"F3c order {phase} user {subject}: {len(target_trials)} trials x {len(variants)} orders",
                  flush=True)
    if maximum_identity_replay_error > 1e-8:
        raise AssertionError("native-order F3c does not replay the frozen parent")
    scores = {phase: {name: wearing._score(rows, phase, name) for name in variants}
              for phase in ("validation", "final")}
    ranks = {}
    for phase in ("validation", "final"):
        native = scores[phase]["native_order"]["pooled"]
        shuffled = [scores[phase][name]["pooled"] for name in variants if name != "native_order"]
        ranks[phase] = {
            "identity_f1_rank_high_is_better": 1 + sum(
                item["macro_f1"] > native["macro_f1"] + 1e-12 for item in shuffled),
            "identity_loss_rank_low_is_better": 1 + sum(
                item["log_loss"] < native["log_loss"] - 1e-12 for item in shuffled),
            "random_f1_min": min(item["macro_f1"] for item in shuffled),
            "random_f1_max": max(item["macro_f1"] for item in shuffled),
            "random_loss_min": min(item["log_loss"] for item in shuffled),
            "random_loss_max": max(item["log_loss"] for item in shuffled),
        }
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": protocol["parent_result_sha256"],
              "parent_prediction_sha256": protocol["parent_prediction_sha256"],
              "archive_sha256": protocol["archive_sha256"],
              "prediction_sha256": sha256(PREDICTIONS), "prediction_rows": len(rows),
              "orders": variants, "maximum_identity_replay_error": maximum_identity_replay_error,
              "scores": scores, "ranks": ranks, "scope": protocol["boundary"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("validation", "final"):
        print(f"F3c {phase}: native F1={scores[phase]['native_order']['pooled']['macro_f1']:.4f}; "
              f"F1 rank={ranks[phase]['identity_f1_rank_high_is_better']}/{len(variants)}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
