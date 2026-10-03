"""Frozen Day2 source-CV and descriptive Day3 replay of exact D/E reliability."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss

from emgimu.feature_bank.calibration import ReliabilityWeights
from emgimu.feature_bank.document_reliability_v2 import DocumentReliabilityWeightsV2


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "benchmarks" / "new_bank_v2"
PROTOCOL = HERE / "DOCUMENT_RELIABILITY_GRAB_PROTOCOL.json"
OUTPUT_JSON = HERE / "DOCUMENT_RELIABILITY_GRAB_RESULTS.json"
OUTPUT_CSV = HERE / "DOCUMENT_RELIABILITY_GRAB_PREDICTIONS.csv"
PATTERN = re.compile(r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(trial_id: str) -> tuple[int, int, int, int]:
    match = PATTERN.fullmatch(trial_id)
    if match is None:
        raise ValueError(f"invalid trial identity: {trial_id}")
    return tuple(map(int, match.groups()))


def score(rows: list[dict], classes: list[int]) -> dict:
    y = np.array([row["gesture"] for row in rows])
    p = np.array([[row[f"p_{label}"] for label in classes] for row in rows])
    if not len(y) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid provider probabilities")
    return {"n": len(y), "macro_f1": float(f1_score(y, np.asarray(classes)[p.argmax(axis=1)],
                                                  labels=classes, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=classes)),
            "brier": float(np.mean(np.sum((p - (y[:, None] == np.asarray(classes))) ** 2, axis=1)))}


def build() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assets = {
        "parent_result_sha256": OLD / "V1_FEATURE_ANCHOR_RESULTS.json",
        "parent_prediction_sha256": OLD / "GRAB_DAY_V1_EXTENSION_PREDICTIONS.csv",
        "source_fitted_f0_features_sha256": HERE / "F7_DOCUMENT_ANCHOR_F0v2.npy",
        "source_fitted_f0_scale_features_sha256": HERE / "DOCUMENT_RELIABILITY_F0_SCALE.npy",
    }
    for key, path in assets.items():
        if sha(path) != protocol[key]:
            raise ValueError(f"frozen source hash changed: {path}")
    parent = json.loads(assets["parent_result_sha256"].read_text(encoding="utf-8"))
    ids = parent["target_trial_ids"]
    if len(ids) != 448 or len(set(ids)) != 448:
        raise ValueError("target feature trial inventory changed")
    features = {
        "F0v2": np.load(assets["source_fitted_f0_features_sha256"], allow_pickle=False),
        "F0v2+scale_pattern": np.load(assets["source_fitted_f0_scale_features_sha256"], allow_pickle=False),
    }
    if (features["F0v2"].shape != (448, 96)
            or features["F0v2+scale_pattern"].shape != (448, 112)
            or not all(np.isfinite(x).all() for x in features.values())):
        raise ValueError("source-fitted target feature shape changed")
    by_feature = {arm: dict(zip(ids, x)) for arm, x in features.items()}
    with assets["parent_prediction_sha256"].open(newline="", encoding="utf-8") as handle:
        frozen = list(csv.DictReader(handle))
    providers = protocol["providers"]
    classes = protocol["classes"]
    prob = {}
    for row in frozen:
        if row["arm"] not in providers:
            continue
        key = (row["arm"], row["trial_id"])
        if key in prob or int(row["gesture"]) != parse(row["trial_id"])[2]:
            raise ValueError("frozen provider identity changed")
        p = np.array([float(row[f"p_{label}"]) for label in classes])
        if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-8):
            raise ValueError("frozen provider probability changed")
        prob[key] = p
    if len(prob) != 2 * len(ids) or any((arm, trial_id) not in prob for arm in providers for trial_id in ids):
        raise ValueError("provider probability inventory changed")

    def trial_ids(day: int, subject: int, repetitions: set[int]) -> list[str]:
        return [trial_id for trial_id in ids if (parts := parse(trial_id))[0] == day
                and parts[1] == subject and parts[2] in classes and parts[3] in repetitions]

    def population(source_ids: list[str]) -> tuple[float, ...]:
        y_index = np.array([classes.index(parse(trial_id)[2]) for trial_id in source_ids])
        losses = np.array([-np.mean(np.log(np.clip(np.array([prob[arm, trial_id][y_index[index]]
                         for index, trial_id in enumerate(source_ids)]), 1e-15, 1))) for arm in providers])
        unnormalized = np.exp(-losses + losses.min())
        return tuple((unnormalized / unnormalized.sum()).tolist())

    def weights(cal_ids: list[str], prior: tuple[float, ...], n0: float, tau: float, source_id: str):
        labels = np.array([parse(trial_id)[2] for trial_id in cal_ids])
        cal = {arm: (np.stack([by_feature[arm][trial_id] for trial_id in cal_ids]), labels, cal_ids)
               for arm in providers}
        exact = DocumentReliabilityWeightsV2(tuple(classes), tuple(providers), prior, n0, tau, source_id)
        detail = exact.calculate(cal)
        legacy = ReliabilityWeights(tuple(classes), tuple(providers), np.asarray(prior), n0=n0,
                                    temperature=tau).personal({arm: (x, y) for arm, (x, y, _) in cal.items()})
        if detail["n_cal_trials"] != len(cal_ids):
            raise ValueError("calibration trial count changed")
        return detail["final"], legacy, detail

    def predictions(eval_ids: list[str], w: np.ndarray, day: int, subject: int, budget: int, arm: str) -> list[dict]:
        result = []
        for trial_id in eval_ids:
            p = sum(weight * prob[provider, trial_id] for weight, provider in zip(w, providers))
            result.append({"day": day, "subject": subject, "budget": budget, "arm": arm,
                           "trial_id": trial_id, "gesture": parse(trial_id)[2],
                           **{f"p_{label}": float(value) for label, value in zip(classes, p)}})
        return result

    source_day = protocol["source_selection_day"]
    grid = []
    source_repetitions = set(protocol["evaluation_repetitions"])
    for n0 in protocol["n0_grid"]:
        for tau in protocol["temperature_grid"]:
            losses, f1s = [], []
            for subject in protocol["subjects"]:
                prior_ids = [trial_id for other in protocol["subjects"] if other != subject
                             for trial_id in trial_ids(source_day, other, source_repetitions)]
                if len(prior_ids) != 7 * 8:
                    raise ValueError("source-CV prior trial count changed")
                prior = population(prior_ids)
                for budget in protocol["shots_per_class"]:
                    cal_ids = trial_ids(source_day, subject, set(range(1, budget + 1)))
                    eval_ids = trial_ids(source_day, subject, source_repetitions)
                    if len(cal_ids) != budget * 4 or len(eval_ids) != 8 or set(cal_ids) & set(eval_ids):
                        raise ValueError("source-CV calibration/evaluation split changed")
                    exact, _, _ = weights(cal_ids, prior, n0, tau, f"day2-leave-subject-{subject}-out")
                    cell = score(predictions(eval_ids, exact, source_day, subject, budget, "exact"), classes)
                    losses.append(cell["log_loss"])
                    f1s.append(cell["macro_f1"])
            grid.append({"n0": n0, "temperature": tau, "cells": len(losses),
                         "mean_log_loss": float(np.mean(losses)), "mean_subject_macro_f1": float(np.mean(f1s))})
        print(f"D/E source CV complete: n0={n0}", flush=True)
    selected = min(grid, key=lambda item: (item["mean_log_loss"], item["n0"], item["temperature"]))
    source_ids = [trial_id for subject in protocol["subjects"]
                  for trial_id in trial_ids(source_day, subject, source_repetitions)]
    if len(source_ids) != 64:
        raise ValueError("full source-population trial count changed")
    prior = population(source_ids)
    print(f"D/E source policy frozen: n0={selected['n0']}, tau={selected['temperature']}", flush=True)

    rows, cells = [], []
    target_day = protocol["descriptive_target_day"]
    for subject in protocol["subjects"]:
        for budget in protocol["shots_per_class"]:
            cal_ids = trial_ids(target_day, subject, set(range(1, budget + 1)))
            eval_ids = trial_ids(target_day, subject, source_repetitions)
            if len(cal_ids) != budget * 4 or len(eval_ids) != 8 or set(cal_ids) & set(eval_ids):
                raise ValueError("descriptive target split changed")
            exact, legacy, detail = weights(cal_ids, prior, selected["n0"], selected["temperature"],
                                             f"day2-source-cv:{sha(PROTOCOL)}")
            for arm, w in (("population_prior", np.asarray(prior)),
                           ("document_exact_personal_reliability", exact),
                           ("legacy_formula_same_policy", legacy)):
                rows.extend(predictions(eval_ids, w, target_day, subject, budget, arm))
            cells.append({"subject": subject, "budget": budget, "calibration_trial_ids": cal_ids,
                          "evaluation_trial_ids": eval_ids, "population": list(prior),
                          "exact_weights": exact.tolist(), "legacy_weights": legacy.tolist(),
                          "between": detail["between"].tolist(), "within": detail["within"].tolist(),
                          "alpha": float(detail["alpha"])})
        print(f"D/E descriptive Day3 complete: subject={subject}", flush=True)
    fields = ["day", "subject", "budget", "arm", "trial_id", "gesture"] + [f"p_{label}" for label in classes]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    scores = {str(budget): {arm: score([row for row in rows if row["budget"] == budget and row["arm"] == arm], classes)
                            for arm in protocol["arms"]} for budget in protocol["shots_per_class"]}
    result = {"status": "descriptive_target_only", "protocol_sha256": sha(PROTOCOL),
              "source_hashes": {key: sha(path) for key, path in assets.items()},
              "prediction_sha256": sha(OUTPUT_CSV), "prediction_rows": len(rows),
              "source_cv_grid": grid, "selected_source_policy": selected,
              "source_population_trial_count": len(source_ids), "source_population": list(prior),
              "descriptive_cells": cells, "descriptive_scores": scores,
              "boundary": protocol["boundary"]}
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    build()
