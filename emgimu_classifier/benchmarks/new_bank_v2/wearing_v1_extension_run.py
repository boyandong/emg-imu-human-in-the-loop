"""Same-user independent new-v1 family screen across wearing domains."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, sha256
from benchmarks.new_bank_v2.wearing_run import ARCHIVE, _score
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2
from emgimu.feature_bank.wearing_full_fusion import load

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "WEARING_V1_EXTENSION_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
CLASSES = np.asarray(PROTOCOL["classes"])


def check_protocol() -> dict:
    for suffix, key in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("RESULTS.json", "parent_result_sha256"),
                        ("TRIAL_PREDICTIONS.csv", "parent_prediction_sha256")):
        if sha256(ROOT / f"WEARING_{suffix}") != PROTOCOL[key]:
            raise AssertionError(f"frozen wearing parent changed: {suffix}")
    parent = json.loads((ROOT / "WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    if (PROTOCOL["source_domain"] != parent["source_domain"]
            or PROTOCOL["target_domains"] != parent["target_domains"]
            or PROTOCOL["validation_subjects"] != parent["validation_subjects"]
            or PROTOCOL["final_subjects"] != parent["final_subjects"]
            or PROTOCOL["rest_label"] != parent["rest_label"]
            or PROTOCOL["candidate_families"] != list(NEW_BANK_V1)
            or ARMS != ("F0v2", *[f"F0v2+{name}" for name in NEW_BANK_V1])):
        raise AssertionError("wearing extension disagrees with parent protocol")
    return parent


def vectors(data, families):
    output = {}
    aligned = None
    for name, family in families.items():
        x, y, users, trials = aggregate(family.transform(data.batch), data)
        if aligned is None:
            aligned = (y, users, trials)
        else:
            for got, expected in zip((y, users, trials), aligned):
                np.testing.assert_array_equal(got, expected)
        output[name] = x
    return output, *aligned


def baseline_replay(rows: list[dict]) -> float:
    with (ROOT / "WEARING_TRIAL_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        parent = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                  if r["arm"] == "F0v2"}
    base = [r for r in rows if r["arm"] == "F0v2"]
    if len(parent) != 240 or len(base) != 240:
        raise AssertionError("wearing baseline prediction inventory changed")
    maximum = 0.0
    for row in base:
        prior = parent[(row["phase"], row["trial_id"])]
        if (row["subject"], row["domain"], row["label"]) != (
                int(prior["subject"]), prior["domain"], int(prior["label"])):
            raise AssertionError("wearing baseline trial metadata changed")
        p = np.asarray([row[f"p_{c}"] for c in CLASSES])
        q = np.asarray([float(prior[f"p_{c}"]) for c in CLASSES])
        maximum = max(maximum, float(np.max(np.abs(p - q))))
    if maximum > 1e-8:
        raise AssertionError(f"wearing baseline numerical drift: {maximum}")
    return maximum


def evaluate() -> dict:
    check_protocol()
    parent_result = json.loads((ROOT / "WEARING_RESULTS.json").read_text(encoding="utf-8"))
    rows, splits, dimensions = [], {}, {}
    for phase in ("validation", "final"):
        for subject in PROTOCOL[f"{phase}_subjects"]:
            source = load(ARCHIVE, subject, (PROTOCOL["source_domain"],))
            target = load(ARCHIVE, subject, tuple(PROTOCOL["target_domains"]))
            if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
                    or source.batch.emg.shape[1] != 40 or set(source.labels) != set(CLASSES)):
                raise AssertionError("wearing source signal contract changed")
            families = {"F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                        **{name: factory().fit(source.batch, source.labels)
                           for name, factory in NEW_BANK_V1.items()}}
            xs, ys, source_users, source_trials = vectors(source, families)
            xt, yt, target_users, target_trials = vectors(target, families)
            if (set(source_trials) & set(target_trials)
                    or set(source_trials) != set(parent_result["split_trial_ids"][f"{phase}_{subject}"]["source"])
                    or set(target_trials) != set(parent_result["split_trial_ids"][f"{phase}_{subject}"]["target"])):
                raise AssertionError("frozen wearing trial partition changed")
            if set(source_users) != {subject} or set(target_users) != {subject}:
                raise AssertionError("wearing subject mismatch")
            splits[f"{phase}_{subject}"] = {"source": source_trials.tolist(),
                                             "target": target_trials.tolist()}
            for name, x in xs.items():
                if name in dimensions and dimensions[name] != x.shape[1]:
                    raise AssertionError("wearing feature dimension changed by subject")
                dimensions[name] = int(x.shape[1])
            for arm in ARMS:
                names = arm.split("+")
                x = np.concatenate([xs[name] for name in names], axis=1)
                x_target = np.concatenate([xt[name] for name in names], axis=1)
                classes, probability = fit_predict(x, ys, x_target)
                np.testing.assert_array_equal(classes, CLASSES)
                for trial_id, label, p in zip(target_trials, yt, probability):
                    match = PATH_RE.fullmatch(str(trial_id))
                    if match is None or int(match["subject"]) != subject or int(match["label"]) != label:
                        raise AssertionError("wearing native trial metadata changed")
                    rows.append({"phase": phase, "subject": subject,
                                 "domain": match["domain"], "arm": arm,
                                 "trial_id": str(trial_id), "label": int(label),
                                 **{f"p_{c}": float(value) for c, value in zip(CLASSES, p)}})
            print(f"wearing new-v1 {phase} subject {subject}: {len(target_trials)} target trials", flush=True)
    replay_error = baseline_replay(rows)
    scores = {phase: {arm: _score(rows, phase, arm) for arm in ARMS}
              for phase in ("validation", "final")}
    selected = min(ARMS, key=lambda arm: (-scores["validation"][arm]["pooled"]["macro_f1"],
                                           scores["validation"][arm]["pooled"]["log_loss"]))
    path = ROOT / "WEARING_V1_EXTENSION_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": PROTOCOL, "protocol_sha256": sha256(PROTOCOL_PATH),
              "archive_sha256": sha256(ARCHIVE), "prediction_sha256": sha256(path),
              "feature_dimensions": dimensions, "split_trial_ids": splits,
              "baseline_replay_max_abs_error": replay_error,
              "validation_selected_arm": selected, "scores": scores,
              "scope": PROTOCOL["scope"]}
    (ROOT / "WEARING_V1_EXTENSION_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wearing new-v1 validation-selected arm: {selected}", flush=True)
    return result


if __name__ == "__main__":
    evaluate()
