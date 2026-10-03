"""Frozen, trial-disjoint public GRAB replay of the document-exact F7 anchor."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss, recall_score

from emgimu.feature_bank.calibration import DocumentPersonalAnchorV2


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "benchmarks" / "new_bank_v2"
PROTOCOL = HERE / "F7_DOCUMENT_ANCHOR_PROTOCOL.json"
OUTPUT_CSV = HERE / "F7_DOCUMENT_ANCHOR_PREDICTIONS.csv"
OUTPUT_JSON = HERE / "F7_DOCUMENT_ANCHOR_RESULTS.json"
PATTERN = re.compile(r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial_parts(trial_id: str) -> tuple[int, int, int, int]:
    match = PATTERN.fullmatch(trial_id)
    if match is None:
        raise ValueError(f"invalid trial id: {trial_id}")
    return tuple(map(int, match.groups()))


def score(rows: list[dict], classes: list[int]) -> dict:
    y = np.array([int(row["gesture"]) for row in rows])
    p = np.array([[float(row[f"p_{label}"]) for label in classes] for row in rows])
    if len(y) == 0 or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid prediction probabilities")
    pred = np.array(classes)[np.argmax(p, axis=1)]
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=classes)),
        "brier": float(np.mean(np.sum((p - (y[:, None] == np.array(classes))) ** 2, axis=1))),
        "recall": {str(label): float(value) for label, value in zip(
            classes, recall_score(y, pred, labels=classes, average=None, zero_division=0)
        )},
    }


def run() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assets = {
        "parent_feature_protocol_sha256": SOURCE / "V1_FEATURE_ANCHOR_PROTOCOL.json",
        "parent_feature_result_sha256": SOURCE / "V1_FEATURE_ANCHOR_RESULTS.json",
        "source_fitted_f0_feature_sha256": SOURCE / "V1_FEATURE_ANCHOR_F0v2.npy",
        "frozen_f0_prediction_sha256": SOURCE / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv",
    }
    for key, path in assets.items():
        if digest(path) != protocol[key]:
            raise ValueError(f"source hash mismatch: {path}")
    parent = json.loads(assets["parent_feature_result_sha256"].read_text(encoding="utf-8"))
    ids = parent["target_trial_ids"]
    features = np.load(assets["source_fitted_f0_feature_sha256"], allow_pickle=False)
    if features.shape != (448, 96) or len(ids) != 448 or len(set(ids)) != 448 or not np.isfinite(features).all():
        raise ValueError("parent target feature identity or shape mismatch")
    by_id = {trial_id: features[index] for index, trial_id in enumerate(ids)}
    with assets["frozen_f0_prediction_sha256"].open(newline="", encoding="utf-8") as handle:
        frozen = list(csv.DictReader(handle))
    frozen = {row["trial_id"]: row for row in frozen if row["arm"] == "F0v2"}
    if set(frozen) != set(ids):
        raise ValueError("frozen F0 prediction identities mismatch")

    classes = protocol["classes"]
    rows: list[dict] = []
    cells: list[dict] = []
    for phase, day in protocol["target_days"].items():
        for budget in protocol["shots_per_class"]:
            for subject in protocol["subjects"]:
                cal_ids = [trial_id for trial_id in ids if (
                    (parts := trial_parts(trial_id))[0] == day and parts[1] == subject
                    and parts[2] in classes and 1 <= parts[3] <= budget
                )]
                eval_ids = [trial_id for trial_id in ids if (
                    (parts := trial_parts(trial_id))[0] == day and parts[1] == subject
                    and parts[2] in classes and parts[3] in protocol["evaluation_repetitions"]
                )]
                if len(cal_ids) != 4 * budget or len(eval_ids) != 8 or set(cal_ids) & set(eval_ids):
                    raise ValueError("calibration/evaluation split mismatch")
                anchor = DocumentPersonalAnchorV2(metric="standardized_euclidean").fit(
                    np.stack([by_id[i] for i in cal_ids]),
                    np.array([trial_parts(i)[2] for i in cal_ids]),
                )
                if anchor.classes_.tolist() != classes:
                    raise ValueError("anchor class order mismatch")
                distances = anchor._distances(np.stack([by_id[i] for i in eval_ids]))
                logits = -distances / anchor.similarity_scale_
                logits -= np.max(logits, axis=1, keepdims=True)
                unnormalized = np.exp(logits)
                anchor_prob = unnormalized / unnormalized.sum(axis=1, keepdims=True)
                for index, trial_id in enumerate(eval_ids):
                    label = trial_parts(trial_id)[2]
                    parent_row = frozen[trial_id]
                    if parent_row["phase"] != phase or int(parent_row["subject"]) != subject or int(parent_row["gesture"]) != label:
                        raise ValueError("frozen F0 row metadata mismatch")
                    base_prob = np.array([float(parent_row[f"p_{k}"]) for k in classes])
                    for arm, prob in (("F0v2", base_prob), ("F7_document", anchor_prob[index]),
                                      ("F0v2+F7_document", 0.5 * (base_prob + anchor_prob[index]))):
                        rows.append({"phase": phase, "day": day, "subject": subject, "budget": budget,
                                     "arm": arm, "trial_id": trial_id, "gesture": label,
                                     **{f"p_{k}": float(v) for k, v in zip(classes, prob)}})
                cells.append({"phase": phase, "day": day, "subject": subject, "budget": budget,
                              "calibration_trial_ids": cal_ids, "evaluation_trial_ids": eval_ids,
                              "calibration_fitted_tau": float(anchor.similarity_scale_)})
            print(f"F7 replay complete: {phase}, {budget} shots/class", flush=True)

    fieldnames = ["phase", "day", "subject", "budget", "arm", "trial_id", "gesture"] + [f"p_{k}" for k in classes]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for phase in protocol["target_days"]:
        metrics[phase] = {}
        for budget in protocol["shots_per_class"]:
            metrics[phase][str(budget)] = {}
            for arm in ("F0v2", "F7_document", "F0v2+F7_document"):
                selected = [row for row in rows if row["phase"] == phase and row["budget"] == budget and row["arm"] == arm]
                scores = score(selected, classes)
                per_subject = [score([row for row in selected if row["subject"] == subject], classes)["macro_f1"]
                               for subject in protocol["subjects"]]
                scores["minimum_subject_macro_f1"] = float(min(per_subject))
                metrics[phase][str(budget)][arm] = scores
    result = {"status": "ok", "protocol_sha256": digest(PROTOCOL),
              "prediction_sha256": digest(OUTPUT_CSV), "source_hashes": {key: digest(path) for key, path in assets.items()},
              "row_count": len(rows), "calibration_cells": cells, "metrics": metrics,
              "boundary": protocol["boundary"]}
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
