"""Deterministic Stage-1 screen from four frozen new-v2 prediction matrices."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "FAMILY_SCREEN_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = tuple(PROTOCOL["arms"])
SPECS = {
    "force": {"file": "FORCE_TRIAL_PREDICTIONS.csv", "result": "FORCE_RESULTS.json",
              "classes": [str(c) for c in range(7)], "phase": "phase", "label": "label",
              "subject": "subject", "condition": "condition", "dataset": "libemg_contraction_intensity",
              "aggregate": "eleven_intensity_conditions"},
    "wearing": {"file": "WEARING_TRIAL_PREDICTIONS.csv", "result": "WEARING_RESULTS.json",
                "classes": [str(c) for c in range(5)], "phase": "phase", "label": "label",
                "subject": "subject", "condition": "domain", "dataset": "libemg_electrode_shift",
                "aggregate": "four_shift_domains"},
    "day": {"file": "GRABMYO_TRIAL_PREDICTIONS.csv", "result": "GRABMYO_RESULTS.json",
            "classes": ["4", "15", "16", "17"], "phase": "split", "label": "gesture",
            "subject": "subject", "condition": None, "dataset": "grabmyo_forearm8",
            "aggregate": "session_by_phase"},
    "Song_same_day": {"file": "SONG_TRIAL_PREDICTIONS.csv", "result": "SONG_RESULTS.json",
                      "classes": ["fist", "index_pinch", "neutral", "open_hand"],
                      "phase": "phase", "label": "label", "subject": None,
                      "condition": "session", "dataset": "song_real8_one_user",
                      "aggregate": "session_by_phase"},
}


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score(rows: list[dict], classes: list[str]) -> dict:
    y = np.asarray([r["label"] for r in rows])
    p = np.asarray([[float(r[f"p_{c}"]) for c in classes] for r in rows])
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-12):
        raise ValueError("invalid saved class probability vector")
    labels = np.asarray(classes)
    pred = labels[p.argmax(axis=1)]
    one_hot = y[:, None] == labels[None, :]
    confidence = p.max(axis=1)
    correct = pred == y
    ece = 0.0
    for index in range(10):
        left, right = index / 10, (index + 1) / 10
        mask = (confidence >= left) & ((confidence <= right) if index == 9 else (confidence < right))
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    class_f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {"trials": len(rows), "accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=labels)),
            "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
            "ece": ece,
            "per_class_f1_json": json.dumps({c: float(v) for c, v in zip(classes, class_f1)}, sort_keys=True)}


def _saved_pooled(dataset: str, phase: str, raw_arm: str, result: dict) -> dict:
    if dataset in ("force", "wearing"):
        return result["scores"][phase][raw_arm]["pooled"]
    if dataset == "day":
        return result["arms"][raw_arm][phase]
    return result["sessions"][phase]["arms"][raw_arm]


def build(verify: bool = False) -> dict:
    scores, all_rows = {}, []
    input_hashes = {}
    for axis, spec in SPECS.items():
        path = ROOT / spec["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PROTOCOL["prediction_sha256"][axis]:
            raise ValueError(f"{axis} frozen prediction hash changed")
        input_hashes[axis] = digest
        with path.open(newline="", encoding="utf-8") as stream:
            raw = list(csv.DictReader(stream))
        result = json.loads((ROOT / spec["result"]).read_text(encoding="utf-8"))
        normalized = []
        for item in raw:
            arm = item["arm"].replace("F0+", "F0v2+") if axis == "day" else item["arm"]
            if axis == "day" and arm == "F0":
                arm = "F0v2"
            if arm not in ARMS:
                continue
            phase = item[spec["phase"]]
            if phase not in ("validation", "final"):
                raise ValueError("unknown frozen phase")
            normalized.append({"phase": phase, "arm": arm,
                               "raw_arm": item["arm"], "trial_id": item["trial_id"],
                               "subject": item[spec["subject"]] if spec["subject"] else "Song",
                               "condition": item[spec["condition"]] if spec["condition"] else
                               ("session_2" if phase == "validation" else "session_3"),
                               "label": item[spec["label"]],
                               **{f"p_{c}": item[f"p_{c}"] for c in spec["classes"]}})
        expected = {"force": 4704, "wearing": 960, "day": 1792, "Song_same_day": 1136}[axis]
        if len(normalized) != expected:
            raise AssertionError(f"{axis} frozen v2 family arm row count changed")
        for phase in ("validation", "final"):
            for arm in ARMS:
                chosen = [r for r in normalized if r["phase"] == phase and r["arm"] == arm]
                pooled = _score(chosen, spec["classes"])
                saved = _saved_pooled(axis, phase, chosen[0]["raw_arm"], result)
                for key in ("trials", "accuracy", "macro_f1", "log_loss", "brier"):
                    if not np.isclose(pooled[key], saved[key], rtol=0, atol=1e-10):
                        raise AssertionError(f"{axis}/{phase}/{arm} differs from saved metrics: {key}")
                scores[(axis, phase, arm)] = pooled
                aggregate = (chosen[0]["condition"] if axis in ("day", "Song_same_day")
                             else spec["aggregate"])
                groups = [("pooled", "Song" if axis == "Song_same_day" else "ALL",
                           aggregate, chosen)]
                if axis != "Song_same_day":
                    groups += [("subject", subject, aggregate,
                                [r for r in chosen if r["subject"] == subject])
                               for subject in sorted({r["subject"] for r in chosen}, key=int)]
                if axis in ("force", "wearing"):
                    groups += [("condition", "ALL", condition,
                                [r for r in chosen if r["condition"] == condition])
                               for condition in sorted({r["condition"] for r in chosen})]
                for scope, subject, condition, group in groups:
                    value = _score(group, spec["classes"])
                    all_rows.append({"dataset": spec["dataset"], "axis": axis,
                                     "phase": phase, "scope": scope, "subject": subject,
                                     "condition": condition, "feature_family": arm,
                                     "calibration_budget": 0, "evaluation_unit": "whole_native_trial",
                                     "evaluation_trials": value["trials"],
                                     "macro_f1": value["macro_f1"], "accuracy": value["accuracy"],
                                     "log_loss": value["log_loss"], "brier": value["brier"],
                                     "ece": value["ece"],
                                     "per_class_f1_json": value["per_class_f1_json"]})
    if len(all_rows) != 256:
        raise AssertionError("expected 256 saved-prediction Stage-1 summary cells")
    vector = []
    for family in PROTOCOL["candidate_families"]:
        for axis in PROTOCOL["robustness_axes"]:
            if axis in ("force", "wearing", "day"):
                for phase in ("validation", "final"):
                    base = scores[(axis, phase, "F0v2")]
                    added = scores[(axis, phase, "F0v2+" + family)]
                    vector.append({"family": family, "axis": axis, "phase": phase,
                                   "dataset": SPECS[axis]["dataset"], "status": "observed",
                                   "delta_macro_f1": added["macro_f1"] - base["macro_f1"],
                                   "delta_logloss_improvement": base["log_loss"] - added["log_loss"],
                                   "delta_brier_improvement": base["brier"] - added["brier"],
                                   "boundary": PROTOCOL["axis_evidence"][axis]})
            else:
                vector.append({"family": family, "axis": axis, "phase": "N/A",
                               "dataset": "N/A", "status": "N/A", "delta_macro_f1": "N/A",
                               "delta_logloss_improvement": "N/A",
                               "delta_brier_improvement": "N/A",
                               "boundary": PROTOCOL["axis_evidence"][axis]})
        for phase in ("validation", "final"):
            base = scores[("Song_same_day", phase, "F0v2")]
            added = scores[("Song_same_day", phase, "F0v2+" + family)]
            vector.append({"family": family, "axis": "Song_same_day_supplement", "phase": phase,
                           "dataset": SPECS["Song_same_day"]["dataset"], "status": "exploratory",
                           "delta_macro_f1": added["macro_f1"] - base["macro_f1"],
                           "delta_logloss_improvement": base["log_loss"] - added["log_loss"],
                           "delta_brier_improvement": base["brier"] - added["brier"],
                           "boundary": PROTOCOL["supplemental"]})
    if len(vector) != 24:
        raise AssertionError("expected 24 seven-axis and Song supplemental vector cells")
    outputs = {"FAMILY_SCREEN.csv": all_rows, "ROBUSTNESS_VECTOR.csv": vector}
    for name, rows in outputs.items():
        path = ROOT / name
        if verify:
            with path.open(newline="", encoding="utf-8") as stream:
                if list(csv.DictReader(stream)) != [
                        {key: str(value) for key, value in row.items()} for row in rows]:
                    raise AssertionError(f"derived table changed: {name}")
        else:
            _write(path, rows)
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "source_predictions_sha256": input_hashes,
             "rows": {name: len(table) for name, table in outputs.items()},
             "saved_pooled_metric_groups_replayed": len(scores),
             "boundary": "three observed axes, four explicit N/A axes; Song one-user/day supplemental"}
    path = ROOT / "FAMILY_SCREEN_AUDIT.json"
    content = json.dumps(audit, indent=2) + "\n"
    if verify:
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError("family-screen audit changed")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": audit["rows"]}))
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
