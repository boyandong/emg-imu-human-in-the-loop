"""Fixed 0/1-shot native-trial calibration on new v2 wearing features."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from benchmarks.new_bank_v1.run import metrics, sha256
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2
from emgimu.feature_bank.wearing_full_fusion import load

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "WEARING_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CLASSES = np.arange(5)
ARMS = ("F0v2", "F0v2+F2a+F3c")


def _score(rows: list[dict], phase: str, arm: str, method: str) -> dict:
    selected = [r for r in rows if r["phase"] == phase and r["arm"] == arm and r["method"] == method]
    y = np.asarray([r["label"] for r in selected])
    p = np.asarray([[r[f"p_{c}"] for c in CLASSES] for r in selected])
    users = np.asarray([r["subject"] for r in selected])
    domains = np.asarray([r["domain"] for r in selected])
    by_subject = {str(user): metrics(y[users == user], p[users == user], CLASSES)
                  for user in sorted(set(users))}
    by_domain = {str(domain): metrics(y[domains == domain], p[domains == domain], CLASSES)
                 for domain in sorted(set(domains))}
    return {"pooled": metrics(y, p, CLASSES), "by_subject": by_subject,
            "by_domain": by_domain,
            "minimum_subject_macro_f1": min(v["macro_f1"] for v in by_subject.values()),
            "worst_domain_macro_f1": min(v["macro_f1"] for v in by_domain.values())}


def run() -> None:
    parent = ROOT / "WEARING_PROTOCOL.json"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PROTOCOL["parent_protocol_sha256"]:
        raise ValueError("parent wearing protocol has changed")
    if (PROTOCOL["selected_parent_arm"] != ARMS[1] or PROTOCOL["comparison_arm"] != ARMS[0]
            or PROTOCOL["shots_per_class"] != 1
            or PROTOCOL["folds"] != ["rep0_cal_rep1_test", "rep1_cal_rep0_test"]):
        raise ValueError("one-shot calibration protocol changed")
    archive = Path(PROTOCOL["archive"])
    rows: list[dict] = []
    assignments: dict[str, dict] = {}
    temperatures: dict[str, float] = {}
    for phase in ("validation", "final"):
        for subject in PROTOCOL[f"{phase}_subjects"]:
            source = load(archive, subject, ("training",))
            target = load(archive, subject, tuple(PROTOCOL["target_domains"]))
            families = {
                "F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                "F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source.batch),
            }
            vectors = {}
            source_y = target_y = source_trials = target_trials = None
            for name, family in families.items():
                a, ay, _, at = aggregate(family.transform(source.batch), source)
                b, by, _, bt = aggregate(family.transform(target.batch), target)
                if source_y is not None:
                    np.testing.assert_array_equal(source_y, ay)
                    np.testing.assert_array_equal(target_y, by)
                    np.testing.assert_array_equal(source_trials, at)
                    np.testing.assert_array_equal(target_trials, bt)
                vectors[name] = (a, b)
                source_y, target_y, source_trials, target_trials = ay, by, at, bt
            if set(source_trials) & set(target_trials):
                raise ValueError("source and target native trials overlap")
            meta = [PATH_RE.fullmatch(str(trial)) for trial in target_trials]
            if any(match is None or int(match["subject"]) != subject for match in meta):
                raise ValueError("target trial identity invalid")
            for arm in ARMS:
                names = arm.split("+")
                x = np.concatenate([vectors[name][0] for name in names], axis=1)
                xt = np.concatenate([vectors[name][1] for name in names], axis=1)
                scaler = StandardScaler().fit(x)
                z, zt = scaler.transform(x), scaler.transform(xt)
                model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                           random_state=20260924).fit(z, source_y)
                np.testing.assert_array_equal(model.classes_, CLASSES)
                if np.max(model.n_iter_) >= 2000:
                    raise RuntimeError("source classifier failed to converge")
                source_probability = model.predict_proba(zt)
                source_centers = np.stack([z[source_y == label].mean(axis=0) for label in CLASSES])
                within = np.mean((z - source_centers[source_y]) ** 2, axis=1)
                temperature = max(float(np.median(within)), 1e-6)
                temperatures[f"{phase}_{subject}_{arm}"] = temperature
                for domain in PROTOCOL["target_domains"]:
                    for cal_rep, test_rep in ((0, 1), (1, 0)):
                        cal = np.asarray([match["domain"] == domain and int(match["rep"]) == cal_rep
                                          for match in meta])
                        test = np.asarray([match["domain"] == domain and int(match["rep"]) == test_rep
                                           for match in meta])
                        if (cal.sum() != 5 or test.sum() != 5 or np.any(cal & test)
                                or set(target_y[cal]) != set(CLASSES)
                                or set(target_y[test]) != set(CLASSES)):
                            raise ValueError("one-shot domain split invalid")
                        prototypes = np.stack([zt[cal][target_y[cal] == label][0] for label in CLASSES])
                        distance = np.mean((zt[test, None, :] - prototypes[None, :, :]) ** 2, axis=2)
                        personal_probability = softmax(-distance.astype(np.float64) / temperature, axis=1)
                        blended = 0.5 * source_probability[test] + 0.5 * personal_probability
                        blended /= blended.sum(axis=1, keepdims=True)
                        assignment = f"{phase}_{subject}_{domain}_cal{cal_rep}"
                        assignments[assignment] = {"calibration": target_trials[cal].tolist(),
                                                   "evaluation": target_trials[test].tolist(),
                                                   "source": source_trials.tolist()}
                        for method, probability in (("source_only", source_probability[test]),
                                                    ("one_shot", blended)):
                            for trial, label, values in zip(target_trials[test], target_y[test], probability):
                                rows.append({"phase": phase, "subject": subject, "domain": domain,
                                             "cal_rep": cal_rep, "arm": arm, "method": method,
                                             "trial_id": str(trial), "label": int(label),
                                             **{f"p_{c}": float(v) for c, v in zip(CLASSES, values)}})
            print(f"wearing calibration {phase} subject {subject}", flush=True)
    scores = {phase: {arm: {method: _score(rows, phase, arm, method)
                            for method in ("source_only", "one_shot")} for arm in ARMS}
              for phase in ("validation", "final")}
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "archive_sha256": sha256(archive), "assignments": assignments,
              "source_only_temperatures": temperatures, "scores": scores,
              "boundary": PROTOCOL["boundary"]}
    (ROOT / "WEARING_CAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "WEARING_CAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
