"""Frozen same-user probability-anchor calibration on GRABMyo native trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_SCORE_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv"
PATTERN = re.compile(r"session(?P<day>[23])_participant(?P<subject>\d+)_gesture(?P<gesture>\d+)_trial(?P<repetition>[1-7])\Z")
CLASSES = tuple(int(value) for value in PROTOCOL["classes"])
ARMS = tuple(PROTOCOL["arms"])


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def score(rows: list[dict]) -> dict:
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    probability = np.asarray([[row[f"p_{c}"] for c in CLASSES] for row in rows], dtype=float)
    predicted = np.asarray(CLASSES)[probability.argmax(axis=1)]
    positions = np.asarray([CLASSES.index(value) for value in labels])
    one_hot = labels[:, None] == np.asarray(CLASSES)[None, :]
    return {"trials": len(rows), "macro_f1": float(f1_score(labels, predicted, labels=CLASSES,
                                                           average="macro", zero_division=0)),
            "accuracy": float(np.mean(labels == predicted)),
            "log_loss": float(np.mean(-np.log(np.clip(
                probability[np.arange(len(rows)), positions], np.finfo(float).eps, 1.0)))),
            "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "per_class_f1": {str(c): float(v) for c, v in zip(CLASSES, f1_score(
                labels, predicted, labels=CLASSES, average=None, zero_division=0))}}


def anchor_probability(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    squared = np.sum((prototypes - query[None, :]) ** 2, axis=1)
    logits = -squared / float(PROTOCOL["temperature"])
    unnormalized = np.exp(logits - logits.max())
    return unnormalized / unnormalized.sum()


def build(verify: bool = False) -> dict:
    if hashlib.sha256((ROOT / "GRABMYO_PROTOCOL.json").read_bytes()).hexdigest() != PROTOCOL["parent_protocol_sha256"]:
        raise AssertionError("parent GRABMyo protocol changed")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != PROTOCOL["prediction_sha256"]:
        raise AssertionError("frozen source probabilities changed")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    indexed = {}
    for row in source:
        arm, phase = row["arm"], row["split"]
        if arm not in ARMS or phase not in PROTOCOL["phases"]:
            raise AssertionError("unexpected arm or phase")
        match = PATTERN.fullmatch(row["trial_id"])
        if match is None:
            raise AssertionError("native recording identity cannot be parsed")
        day = 2 if phase == "validation" else 3
        subject, gesture, repetition = (int(match[name]) for name in
                                        ("subject", "gesture", "repetition"))
        if (int(match["day"]) != day or subject != int(row["subject"]) or
                gesture != int(row["gesture"]) or subject not in PROTOCOL["subjects"] or
                gesture not in CLASSES):
            raise AssertionError("native ID disagrees with frozen row metadata")
        probability = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        if not np.all(np.isfinite(probability)) or np.any(probability < 0) or not np.isclose(probability.sum(), 1, atol=1e-12):
            raise AssertionError("invalid frozen probability vector")
        key = (arm, phase, subject, gesture, repetition)
        if key in indexed:
            raise AssertionError("duplicate native recording")
        indexed[key] = {"trial_id": row["trial_id"], "p": probability}
    expected = len(ARMS) * len(PROTOCOL["phases"]) * len(PROTOCOL["subjects"]) * len(CLASSES) * 7
    if len(indexed) != expected:
        raise AssertionError(f"expected {expected} frozen native arm-recordings")
    rows, groups, assignments = [], [], []
    for arm in ARMS:
        for phase in PROTOCOL["phases"]:
            for budget in PROTOCOL["budgets"]:
                phase_rows = []
                for subject in PROTOCOL["subjects"]:
                    prototypes = None
                    calibration_ids = []
                    if budget:
                        prototypes = np.stack([np.mean([
                            indexed[(arm, phase, subject, label, repetition)]["p"]
                            for repetition in range(1, budget + 1)], axis=0)
                            for label in CLASSES])
                        calibration_ids = [indexed[(arm, phase, subject, label, repetition)]["trial_id"]
                                           for label in CLASSES for repetition in range(1, budget + 1)]
                    eval_ids = []
                    for label in CLASSES:
                        for repetition in PROTOCOL["evaluation_repetitions"]:
                            record = indexed[(arm, phase, subject, label, repetition)]
                            eval_ids.append(record["trial_id"])
                            population = record["p"]
                            probability = population if budget == 0 else (
                                (1 - PROTOCOL["anchor_weight"]) * population +
                                PROTOCOL["anchor_weight"] * anchor_probability(population, prototypes))
                            if not np.isclose(probability.sum(), 1, atol=1e-12):
                                raise AssertionError("adapted probabilities do not sum to one")
                            phase_rows.append({"arm": arm, "phase": phase, "subject": subject,
                                               "shots_per_class": budget, "trial_id": record["trial_id"],
                                               "label": label, "repetition": repetition,
                                               **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
                    if set(calibration_ids) & set(eval_ids):
                        raise AssertionError("calibration and evaluation recording overlap")
                    assignments.append({"arm": arm, "phase": phase, "subject": subject,
                                        "shots_per_class": budget, "calibration_ids": calibration_ids,
                                        "evaluation_ids": eval_ids})
                if len(phase_rows) != 64:
                    raise AssertionError("expected 64 fixed evaluation recordings per arm/phase/budget")
                rows.extend(phase_rows)
                values = score(phase_rows)
                groups.append({"arm": arm, "phase": phase, "subject": "ALL",
                               "shots_per_class": budget, **values})
                for subject in PROTOCOL["subjects"]:
                    subset = [row for row in phase_rows if row["subject"] == subject]
                    groups.append({"arm": arm, "phase": phase, "subject": subject,
                                   "shots_per_class": budget, **score(subset)})
    if len(rows) != 2048 or len(groups) != 288 or len(assignments) != 256:
        raise AssertionError("unexpected calibration output size")
    zero = [row for row in rows if row["shots_per_class"] == 0]
    for row in zero:
        native = PATTERN.fullmatch(row["trial_id"])
        original = indexed[(row["arm"], row["phase"], int(native["subject"]),
                            int(native["gesture"]), int(native["repetition"]))]["p"]
        if not np.array_equal(original, [row[f"p_{c}"] for c in CLASSES]):
            raise AssertionError("zero-shot probabilities must exactly replay source model")
    curve = [{**{key: value for key, value in row.items() if key != "per_class_f1"},
              "per_class_f1_json": json.dumps(row["per_class_f1"], sort_keys=True)}
             for row in groups]
    outputs = {"GRAB_SCORE_CAL_PREDICTIONS.csv": rows,
               "GRAB_SCORE_CAL_CURVE.csv": curve}
    for name, output in outputs.items():
        content = csv_text(output)
        path = ROOT / name
        if verify:
            if path.read_bytes() != content.encode("utf-8"):
                raise AssertionError(f"derived output changed: {name}")
        else:
            path.write_bytes(content.encode("utf-8"))
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "source_predictions_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "rows": {name: len(output) for name, output in outputs.items()},
             "native_source_arm_recordings": len(indexed), "zero_shot_exact_replays": len(zero),
             "assignments": assignments, "boundary": PROTOCOL["boundary"]}
    path = ROOT / "GRAB_SCORE_CAL_AUDIT.json"
    content = json.dumps(audit, indent=2) + "\n"
    if verify:
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError("calibration assignment audit changed")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify,
                      "predictions": len(rows), "metric_rows": len(groups)}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
