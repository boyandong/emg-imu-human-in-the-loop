"""Verify frozen native-trial v2 leave-one-family-out without refitting."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.datasets.libemg_force import FILE_RE

ROOT = Path(__file__).resolve().parent
FULL = "F0v2+F2a+F3c"
REMOVALS = {"F0v2": "F2a+F3c", "F2a": "F0v2+F3c", "F3c": "F0v2+F2a"}
PARENT = {"wearing": ("WEARING_TRIAL_PREDICTIONS.csv", "WEARING_RESULTS.json"),
          "force": ("FORCE_TRIAL_PREDICTIONS.csv", "FORCE_RESULTS.json")}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(rows: list[dict], classes: np.ndarray) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in rows])
    prediction = p.argmax(axis=1)
    return {"trials": len(rows), "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=classes)),
            "brier": float(np.mean(np.sum((p - np.eye(len(classes))[y]) ** 2, axis=1))),
            "per_class_recall": {str(c): float(v) for c, v in zip(
                classes, recall_score(y, prediction, labels=classes, average=None, zero_division=0))}}


def _compare(saved: dict, actual: dict, label: str) -> None:
    if set(saved) != set(actual):
        raise AssertionError(f"score keys changed: {label}")
    for field, value in actual.items():
        if isinstance(value, dict):
            _compare(saved[field], value, label + "/" + field)
        elif not np.isclose(saved[field], value, rtol=0, atol=1e-12):
            raise AssertionError(f"score changed: {label}/{field}")


def verify() -> dict:
    protocol_path = ROOT / "LOFO_PROTOCOL.json"
    result = json.loads((ROOT / "LOFO_RESULTS.json").read_text(encoding="utf-8"))
    if result["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise AssertionError("LOFO protocol hash changed")
    rows = _read(ROOT / "LOFO_TRIAL_PREDICTIONS.csv")
    if len(rows) != 5664:
        raise AssertionError("LOFO prediction row count changed")
    parent_lookup = {}
    parent_results = {}
    for dataset, (predictions, result_name) in PARENT.items():
        path = ROOT / predictions
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["protocol"][
                "parent_predictions_sha256"][dataset]:
            raise AssertionError("parent predictions changed")
        raw = _read(path)
        parent_results[dataset] = json.loads((ROOT / result_name).read_text(encoding="utf-8"))
        for item in raw:
            parent_lookup[(dataset, item["phase"], item["arm"], item["trial_id"])] = item
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_trial: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in PARENT or row["arm"] not in {FULL, *REMOVALS.values()}:
            raise AssertionError("unexpected dataset or arm")
        classes = np.arange(5 if dataset == "wearing" else 7)
        if dataset == "wearing":
            match = PATH_RE.fullmatch(row["trial_id"])
            condition = match["domain"] if match else None
        else:
            match = FILE_RE.fullmatch(row["trial_id"] + ".csv")
            condition = match["condition"] if match else None
        if (match is None or int(match["subject"]) != int(row["subject"]) or
                condition != row["condition"] or
                (dataset == "force" and int(match["label"]) - 1 != int(row["label"]))):
            raise AssertionError("native trial identity mismatch")
        p = np.asarray([float(row[f"p_{c}"]) for c in classes])
        if np.any(p < 0) or not np.all(np.isfinite(p)) or not np.isclose(p.sum(), 1, atol=1e-12):
            raise AssertionError("class probabilities invalid")
        if dataset == "wearing" and (row["p_5"] or row["p_6"]):
            raise AssertionError("wearing has five classes only")
        if row["arm"] != "F2a+F3c":
            prior = parent_lookup[(dataset, row["phase"], row["arm"], row["trial_id"])]
            if (int(prior["label"]) != int(row["label"]) or
                    not np.allclose(p, [float(prior[f"p_{c}"]) for c in classes], rtol=0, atol=0)):
                raise AssertionError("parent arm failed exact probability replay")
        key = (dataset, row["phase"], row["trial_id"])
        if row["arm"] in by_trial[key]:
            raise AssertionError("duplicate arm for native trial")
        by_trial[key][row["arm"]] = row
        grouped[(dataset, row["phase"], row["arm"])].append(row)
    if len(by_trial) != 1416 or any(set(arms) != {FULL, *REMOVALS.values()}
                                     for arms in by_trial.values()):
        raise AssertionError("LOFO arms are not matched on every native trial")
    for key, split in result["split_trial_ids"].items():
        dataset = key.split("_", 1)[0]
        if set(split["source"]) & set(split["target"]) or set(split["source_users"]) & set(split["target_users"]):
            if dataset == "force" or set(split["source"]) & set(split["target"]):
                raise AssertionError("source/target trial or user leakage")
        if dataset == "wearing":
            expected = parent_results[dataset]["split_trial_ids"][key.split("_", 1)[1]]
            if split["source"] != expected["source"] or split["target"] != expected["target"]:
                raise AssertionError("wearing trial split changed")
        else:
            phase = key.split("_", 1)[1]
            expected = parent_results[dataset]["split_trial_ids"][phase]
            if split["source"] != expected["source_trials"] or split["target"] != expected["target_trials"]:
                raise AssertionError("force trial split changed")
    paired = {}
    for dataset in PARENT:
        classes = np.arange(5 if dataset == "wearing" else 7)
        for phase in ("validation", "final"):
            for arm in (FULL, *REMOVALS.values()):
                subset = grouped[(dataset, phase, arm)]
                if len(subset) != (120 if dataset == "wearing" else 588):
                    raise AssertionError("phase-arm native trial count changed")
                score = result["scores"][dataset][phase][arm]
                _compare(score["pooled"], _score(subset, classes), f"{dataset}/{phase}/{arm}/pooled")
                for axis, column in (("by_subject", "subject"), ("by_condition", "condition")):
                    for cell, saved in score[axis].items():
                        _compare(saved, _score([r for r in subset if r[column] == cell], classes),
                                 f"{dataset}/{phase}/{arm}/{axis}/{cell}")
                if not np.isclose(score["minimum_subject_macro_f1"],
                                  min(v["macro_f1"] for v in score["by_subject"].values()), atol=1e-12):
                    raise AssertionError("minimum subject changed")
                if not np.isclose(score["worst_condition_macro_f1"],
                                  min(v["macro_f1"] for v in score["by_condition"].values()), atol=1e-12):
                    raise AssertionError("worst condition changed")
            full = {r["trial_id"]: r for r in grouped[(dataset, phase, FULL)]}
            for family, arm in REMOVALS.items():
                removed = {r["trial_id"]: r for r in grouped[(dataset, phase, arm)]}
                full_score = result["scores"][dataset][phase][FULL]["pooled"]
                removed_score = result["scores"][dataset][phase][arm]["pooled"]
                recorded = result["full_minus_removed"][dataset][phase][family]
                actual = {"delta_macro_f1": full_score["macro_f1"] - removed_score["macro_f1"],
                          "delta_logloss_improvement": removed_score["log_loss"] - full_score["log_loss"],
                          "delta_brier_improvement": removed_score["brier"] - full_score["brier"]}
                _compare(recorded, actual, f"{dataset}/{phase}/remove_{family}")
                corrected = created = 0
                for trial, f in full.items():
                    r = removed[trial]
                    f_ok = np.argmax([float(f[f"p_{c}"]) for c in classes]) == int(f["label"])
                    r_ok = np.argmax([float(r[f"p_{c}"]) for c in classes]) == int(r["label"])
                    corrected += int(f_ok and not r_ok)
                    created += int(not f_ok and r_ok)
                paired[f"{dataset}_{phase}_remove_{family}"] = {
                    "full_correct_removed_wrong": corrected,
                    "full_wrong_removed_correct": created}
    audit = {"status": "ok", "prediction_rows": len(rows), "native_trials": len(by_trial),
             "score_groups_recomputed": len(grouped),
             "parent_arms": "exact saved probability replay", "paired_vs_removal": paired,
             "boundary": "fixed candidate full bank; public trial-level descriptive final results"}
    (ROOT / "LOFO_VERIFICATION.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "prediction_rows": len(rows), "native_trials": len(by_trial)}))
    return audit


if __name__ == "__main__":
    verify()
