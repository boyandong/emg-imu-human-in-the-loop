"""Frozen new-v2 full-bank leave-one-family-out on native eight-channel trials."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v1.run import fit_predict, metrics
from benchmarks.new_bank_v2.force_run import RAW as FORCE_RAW
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.datasets.libemg_force import FILE_RE, load_libemg_force_windows
from emgimu.feature_bank.force_full_fusion import aggregate
from emgimu.feature_bank.new_bank_v2 import RingRelativeCovarianceV2, TraceCovarianceV2
from emgimu.feature_bank.wearing_full_fusion import load as load_wearing

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "LOFO_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
FULL = "F0v2+F2a+F3c"
ARMS = (FULL, "F2a+F3c", "F0v2+F3c", "F0v2+F2a")
PARENT_PATHS = {"wearing": ("WEARING_PROTOCOL.json", "WEARING_TRIAL_PREDICTIONS.csv"),
                "force": ("FORCE_PROTOCOL.json", "FORCE_TRIAL_PREDICTIONS.csv")}


def _parent_rows(dataset: str) -> list[dict]:
    protocol_name, predictions_name = PARENT_PATHS[dataset]
    if hashlib.sha256((ROOT / protocol_name).read_bytes()).hexdigest() != PROTOCOL[
            "parent_protocol_sha256"][dataset]:
        raise ValueError(f"{dataset} parent protocol changed")
    path = ROOT / predictions_name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL["parent_predictions_sha256"][dataset]:
        raise ValueError(f"{dataset} parent predictions changed")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    expected = 1200 if dataset == "wearing" else 5880
    if len(source) != expected:
        raise ValueError("parent prediction row count changed")
    rows = []
    for row in source:
        if row["arm"] not in (FULL, "F0v2+F3c", "F0v2+F2a"):
            continue
        classes = range(5 if dataset == "wearing" else 7)
        rows.append({"dataset": dataset, "phase": row["phase"],
                     "subject": int(row["subject"]),
                     "condition": row["domain" if dataset == "wearing" else "condition"],
                     "arm": row["arm"], "trial_id": row["trial_id"], "label": int(row["label"]),
                     **{f"p_{c}": float(row[f"p_{c}"]) for c in classes},
                     **{f"p_{c}": "" for c in range(len(classes), 7)}})
    if len(rows) != 3 * (240 if dataset == "wearing" else 1176):
        raise AssertionError("missing parent LOFO arms")
    return rows


def _append_new(rows: list[dict], dataset: str, phase: str, source, target,
                *, subject: int | None = None) -> dict:
    families = {"F2a": TraceCovarianceV2(shrinkage=0.05).fit(source.batch),
                "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(source.batch)}
    vectors = {}
    source_y = target_y = source_trials = target_trials = target_users = None
    for name, family in families.items():
        x, y, _, trials = aggregate(family.transform(source.batch), source)
        xt, yt, users, trials_t = aggregate(family.transform(target.batch), target)
        if source_y is not None:
            np.testing.assert_array_equal(source_y, y)
            np.testing.assert_array_equal(target_y, yt)
            np.testing.assert_array_equal(source_trials, trials)
            np.testing.assert_array_equal(target_trials, trials_t)
        vectors[name] = (x, xt)
        source_y, target_y = y, yt
        source_trials, target_trials, target_users = trials, trials_t, users
    if set(source_trials) & set(target_trials):
        raise ValueError("source/target native trial overlap")
    x = np.concatenate([vectors[name][0] for name in ("F2a", "F3c")], axis=1)
    xt = np.concatenate([vectors[name][1] for name in ("F2a", "F3c")], axis=1)
    classes, probabilities = fit_predict(x, source_y, xt)
    np.testing.assert_array_equal(classes, np.arange(5 if dataset == "wearing" else 7))
    for trial, user, label, p in zip(target_trials, target_users, target_y, probabilities):
        if dataset == "wearing":
            match = PATH_RE.fullmatch(str(trial))
            if match is None or int(match["subject"]) != user or user != subject:
                raise ValueError("wearing native trial identity mismatch")
            condition = match["domain"]
        else:
            match = FILE_RE.fullmatch(str(trial) + ".csv")
            if match is None or int(match["subject"]) != user or int(match["label"]) - 1 != label:
                raise ValueError("force native trial identity mismatch")
            condition = match["condition"]
        rows.append({"dataset": dataset, "phase": phase, "subject": int(user),
                     "condition": condition, "arm": "F2a+F3c",
                     "trial_id": str(trial), "label": int(label),
                     **{f"p_{c}": float(value) for c, value in zip(classes, p)},
                     **{f"p_{c}": "" for c in range(len(classes), 7)}})
    return {"source": source_trials.tolist(), "target": target_trials.tolist(),
            "source_users": sorted(set(map(int, source.subjects))),
            "target_users": sorted(set(map(int, target_users)))}


def _scores(rows: list[dict], dataset: str, phase: str, arm: str) -> dict:
    chosen = [r for r in rows if r["dataset"] == dataset and r["phase"] == phase and r["arm"] == arm]
    classes = np.arange(5 if dataset == "wearing" else 7)
    y = np.asarray([r["label"] for r in chosen])
    p = np.asarray([[r[f"p_{c}"] for c in classes] for r in chosen], dtype=np.float64)
    users = np.asarray([r["subject"] for r in chosen])
    conditions = np.asarray([r["condition"] for r in chosen])
    by_subject = {str(user): metrics(y[users == user], p[users == user], classes)
                  for user in sorted(set(users))}
    by_condition = {str(condition): metrics(y[conditions == condition], p[conditions == condition], classes)
                    for condition in sorted(set(conditions))}
    return {"pooled": metrics(y, p, classes), "by_subject": by_subject,
            "by_condition": by_condition,
            "minimum_subject_macro_f1": min(v["macro_f1"] for v in by_subject.values()),
            "worst_condition_macro_f1": min(v["macro_f1"] for v in by_condition.values())}


def run() -> None:
    if PROTOCOL["candidate_full_bank"] != FULL or PROTOCOL["removals"] != {
            "F0v2": "F2a+F3c", "F2a": "F0v2+F3c", "F3c": "F0v2+F2a"}:
        raise ValueError("frozen LOFO arm map changed")
    rows = _parent_rows("wearing") + _parent_rows("force")
    splits = {}
    wearing_protocol = json.loads((ROOT / "WEARING_PROTOCOL.json").read_text(encoding="utf-8"))
    for phase in ("validation", "final"):
        for subject in wearing_protocol[f"{phase}_subjects"]:
            source = load_wearing(Path(wearing_protocol["archive"]), subject, ("training",))
            target = load_wearing(Path(wearing_protocol["archive"]), subject,
                                  tuple(wearing_protocol["target_domains"]))
            splits[f"wearing_{phase}_{subject}"] = _append_new(
                rows, "wearing", phase, source, target, subject=subject)
            print(f"LOFO wearing {phase} subject {subject}", flush=True)
    force_protocol = json.loads((ROOT / "FORCE_PROTOCOL.json").read_text(encoding="utf-8"))
    source = load_libemg_force_windows(FORCE_RAW, subjects=force_protocol["source_subjects"],
                                       conditions=force_protocol["source_conditions"])
    for phase in ("validation", "final"):
        target = load_libemg_force_windows(FORCE_RAW, subjects=force_protocol[f"{phase}_subjects"],
                                           conditions=force_protocol["target_conditions"])
        splits[f"force_{phase}"] = _append_new(rows, "force", phase, source, target)
        print(f"LOFO force {phase}", flush=True)
    if len(rows) != 5664:
        raise AssertionError("LOFO native trial row count changed")
    scores = {dataset: {phase: {arm: _scores(rows, dataset, phase, arm) for arm in ARMS}
                        for phase in ("validation", "final")}
              for dataset in ("wearing", "force")}
    increments = {dataset: {phase: {family: {
        "delta_macro_f1": scores[dataset][phase][FULL]["pooled"]["macro_f1"] -
        scores[dataset][phase][arm]["pooled"]["macro_f1"],
        "delta_logloss_improvement": scores[dataset][phase][arm]["pooled"]["log_loss"] -
        scores[dataset][phase][FULL]["pooled"]["log_loss"],
        "delta_brier_improvement": scores[dataset][phase][arm]["pooled"]["brier"] -
        scores[dataset][phase][FULL]["pooled"]["brier"]}
        for family, arm in PROTOCOL["removals"].items()} for phase in ("validation", "final")}
        for dataset in ("wearing", "force")}
    result = {"protocol": PROTOCOL,
              "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
              "split_trial_ids": splits, "scores": scores,
              "full_minus_removed": increments, "boundary": PROTOCOL["boundary"]}
    (ROOT / "LOFO_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "LOFO_TRIAL_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "phase", "subject", "condition",
                                                      "arm", "trial_id", "label",
                                                      *[f"p_{c}" for c in range(7)]])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
