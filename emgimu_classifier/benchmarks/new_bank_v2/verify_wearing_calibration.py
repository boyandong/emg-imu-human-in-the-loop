"""Audit one-shot native-trial assignments and independently rescore predictions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from emgimu.datasets.electrode_shift import PATH_RE

ROOT = Path(__file__).resolve().parent
CLASSES = np.arange(5)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES] for r in rows])
    if not len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("invalid probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    pred = p.argmax(axis=1)
    return {"trials": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=CLASSES)),
            "brier": float(np.mean(np.sum((p - np.eye(5)[y]) ** 2, axis=1)))}


def _same(a: float, b: float) -> None:
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def verify() -> dict:
    result = json.loads((ROOT / "WEARING_CAL_RESULTS.json").read_text(encoding="utf-8"))
    protocol_file = ROOT / "WEARING_CAL_PROTOCOL.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (result["protocol"] != protocol or
            hashlib.sha256(protocol_file.read_bytes()).hexdigest() != result["protocol_sha256"] or
            hashlib.sha256((ROOT / "WEARING_PROTOCOL.json").read_bytes()).hexdigest()
            != protocol["parent_protocol_sha256"]):
        raise AssertionError("frozen protocol mismatch")
    rows = _read(ROOT / "WEARING_CAL_PREDICTIONS.csv")
    if len(rows) != 960 or len(result["assignments"]) != 48:
        raise AssertionError("expected two arms/treatments for 240 held-out trial evaluations")
    index = {}
    for row in rows:
        key = (row["phase"], int(row["subject"]), row["domain"], int(row["cal_rep"]),
               row["arm"], row["method"], row["trial_id"])
        if key in index:
            raise AssertionError("duplicate prediction")
        index[key] = row
        native = PATH_RE.fullmatch(row["trial_id"])
        if (native is None or int(native["subject"]) != key[1]
                or native["domain"] != key[2] or int(native["rep"]) == key[3]
                or int(native["label"]) != int(row["label"])):
            raise AssertionError("native target identity or fold mismatch")
    for name, assignment in result["assignments"].items():
        phase, subject, domain_rep = name.split("_", 2)
        domain, rep = domain_rep.rsplit("_cal", 1)
        subject, cal_rep = int(subject), int(rep)
        source, calibration, evaluation = (set(assignment[k]) for k in
                                           ("source", "calibration", "evaluation"))
        if (len(source) != 25 or len(calibration) != 5 or len(evaluation) != 5
                or source & calibration or source & evaluation or calibration & evaluation):
            raise AssertionError("source/calibration/evaluation trial leakage")
        if ({int(PATH_RE.fullmatch(t)["label"]) for t in calibration} != set(CLASSES)
                or {int(PATH_RE.fullmatch(t)["label"]) for t in evaluation} != set(CLASSES)):
            raise AssertionError("not exactly one native trial per class")
        for arm in ("F0v2", "F0v2+F2a+F3c"):
            for method in ("source_only", "one_shot"):
                found = {key[6] for key in index if key[:6] == (
                    phase, subject, domain, cal_rep, arm, method)}
                if found != evaluation:
                    raise AssertionError("evaluation identity changed across treatments")
    for phase in ("validation", "final"):
        for arm in ("F0v2", "F0v2+F2a+F3c"):
            for method in ("source_only", "one_shot"):
                part = [r for r in rows if r["phase"] == phase and r["arm"] == arm
                        and r["method"] == method]
                saved = result["scores"][phase][arm][method]
                for key, value in _score(part).items():
                    if key == "trials":
                        assert saved["pooled"][key] == value == 120
                    else:
                        _same(saved["pooled"][key], value)
                for field, column in (("by_subject", "subject"), ("by_domain", "domain")):
                    for group, score in saved[field].items():
                        actual = _score([r for r in part if str(r[column]) == group])
                        for metric, value in actual.items():
                            _same(score[metric], value)
                _same(saved["minimum_subject_macro_f1"],
                      min(s["macro_f1"] for s in saved["by_subject"].values()))
                _same(saved["worst_domain_macro_f1"],
                      min(s["macro_f1"] for s in saved["by_domain"].values()))
    parent = {(r["phase"], int(r["subject"]), r["arm"], r["trial_id"]): r for r in
              _read(ROOT / "WEARING_TRIAL_PREDICTIONS.csv")}
    zero = {(r["phase"], int(r["subject"]), r["arm"], r["trial_id"]): r for r in rows
            if r["method"] == "source_only"}
    if set(zero) != {key for key in parent if key[2] in ("F0v2", "F0v2+F2a+F3c")}:
        raise AssertionError("zero-shot parent trial coverage differs")
    parent_error = max(abs(float(row[f"p_{c}"]) - float(parent[key][f"p_{c}"]))
                       for key, row in zero.items() for c in CLASSES)
    if parent_error > 1e-12:
        raise AssertionError("source model differs from frozen parent experiment")
    paired = {}
    for phase in ("validation", "final"):
        for arm in ("F0v2", "F0v2+F2a+F3c"):
            base = {(r["subject"], r["trial_id"]): r for r in rows if r["phase"] == phase
                    and r["arm"] == arm and r["method"] == "source_only"}
            adapted = {(r["subject"], r["trial_id"]): r for r in rows if r["phase"] == phase
                       and r["arm"] == arm and r["method"] == "one_shot"}
            corrected = harmed = 0
            for key, left in base.items():
                right = adapted[key]
                truth = int(left["label"])
                bp = int(np.argmax([float(left[f"p_{c}"]) for c in CLASSES]))
                ap = int(np.argmax([float(right[f"p_{c}"]) for c in CLASSES]))
                corrected += bp != truth and ap == truth
                harmed += bp == truth and ap != truth
            paired[f"{phase}_{arm}"] = {"corrected": int(corrected), "new_errors": int(harmed)}
    output = {"status": "ok", "prediction_rows": len(rows), "assignments": 48,
              "score_groups_recomputed": 8, "parent_zero_shot_max_error": parent_error,
              "paired": paired}
    (ROOT / "WEARING_CAL_VERIFICATION.json").write_text(json.dumps(output, indent=2) + "\n",
                                                        encoding="utf-8")
    print(json.dumps(output))
    return output


if __name__ == "__main__":
    verify()
