"""Frozen independent F0v2/F2a/F3c trial-level electrode-shift experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, metrics, sha256
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)
from emgimu.feature_bank.wearing_full_fusion import load

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "WEARING_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARCHIVE = Path(PROTOCOL["archive"])
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.arange(5)


def _score(rows: list[dict], phase: str, arm: str) -> dict:
    part = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
    y = np.asarray([r["label"] for r in part])
    p = np.asarray([[r[f"p_{c}"] for c in CLASSES] for r in part])
    subjects = np.asarray([r["subject"] for r in part])
    domains = np.asarray([r["domain"] for r in part])
    by_subject = {str(s): metrics(y[subjects == s], p[subjects == s], CLASSES)
                  for s in sorted(set(subjects))}
    by_domain = {str(d): metrics(y[domains == d], p[domains == d], CLASSES)
                 for d in sorted(set(domains))}
    return {"pooled": metrics(y, p, CLASSES), "by_subject": by_subject,
            "by_domain": by_domain,
            "minimum_subject_macro_f1": min(v["macro_f1"] for v in by_subject.values()),
            "worst_domain_macro_f1": min(v["macro_f1"] for v in by_domain.values())}


def run() -> None:
    if (ARMS != ("F0", "F0v2", "F0v2+F2a", "F0v2+F3c", "F0v2+F2a+F3c")
            or PROTOCOL["rest_label"] != 2
            or PROTOCOL["source_domain"] != "training"
            or PROTOCOL["target_domains"] != ["trial_1", "trial_2", "trial_3", "trial_4"]):
        raise ValueError("wearing protocol changed")
    rows: list[dict] = []
    split_ids: dict[str, dict] = {}
    dimensions: dict[str, int] = {}
    for phase in ("validation", "final"):
        for subject in PROTOCOL[f"{phase}_subjects"]:
            source = load(ARCHIVE, subject, ("training",))
            target = load(ARCHIVE, subject, tuple(PROTOCOL["target_domains"]))
            families = {
                "F0": LocalDetailFamily().fit(source.batch, source.labels),
                "F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source.batch),
            }
            source_vectors, target_vectors = {}, {}
            source_y = target_y = source_trials = target_trials = None
            for name, family in families.items():
                a, ay, _, at = aggregate(family.transform(source.batch), source)
                b, by, _, bt = aggregate(family.transform(target.batch), target)
                if source_y is not None:
                    np.testing.assert_array_equal(source_y, ay)
                    np.testing.assert_array_equal(target_y, by)
                    np.testing.assert_array_equal(source_trials, at)
                    np.testing.assert_array_equal(target_trials, bt)
                source_vectors[name], target_vectors[name] = a, b
                source_y, target_y, source_trials, target_trials = ay, by, at, bt
                dimensions[name] = int(a.shape[1])
            if set(source_trials) & set(target_trials):
                raise ValueError("native source/target trial overlap")
            split_ids[f"{phase}_{subject}"] = {
                "source": source_trials.tolist(), "target": target_trials.tolist()}
            for arm in ARMS:
                names = arm.split("+")
                x = np.concatenate([source_vectors[name] for name in names], axis=1)
                xt = np.concatenate([target_vectors[name] for name in names], axis=1)
                classes, p = fit_predict(x, source_y, xt)
                np.testing.assert_array_equal(classes, CLASSES)
                for trial, label, probability in zip(target_trials, target_y, p):
                    match = PATH_RE.fullmatch(str(trial))
                    if match is None or int(match["subject"]) != subject:
                        raise ValueError("invalid native trial identity")
                    rows.append({"phase": phase, "subject": subject, "domain": match["domain"],
                                 "arm": arm, "trial_id": str(trial), "label": int(label),
                                 **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
            print(f"wearing {phase} subject {subject}: {len(target_trials)} target trials", flush=True)
    scores = {phase: {arm: _score(rows, phase, arm) for arm in ARMS}
              for phase in ("validation", "final")}
    selected = min(ARMS, key=lambda arm: (-scores["validation"][arm]["pooled"]["macro_f1"],
                                           scores["validation"][arm]["pooled"]["log_loss"]))
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "archive_sha256": sha256(ARCHIVE), "feature_dimensions": dimensions,
              "split_trial_ids": split_ids, "validation_selected_arm": selected,
              "scores": scores, "boundary": PROTOCOL["boundary"]}
    (ROOT / "WEARING_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "WEARING_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"validation-selected arm: {selected}", flush=True)


if __name__ == "__main__":
    run()
