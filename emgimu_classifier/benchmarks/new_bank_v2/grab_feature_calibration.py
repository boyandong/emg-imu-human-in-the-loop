"""Cross-day personal prototypes in source-standardized new-v2 feature space."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v1.grabmyo_run import aggregate
from benchmarks.new_bank_v2.grab_score_calibration import csv_text, score
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import (
    RestNoiseDetailV2, RingRelativeCovarianceV2, TraceCovarianceV2,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "GRAB_FEATURE_CAL_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE_PREDICTIONS = ROOT / "GRABMYO_TRIAL_PREDICTIONS.csv"
ARMS = tuple(PROTOCOL["arms"])
CLASSES = tuple(PROTOCOL["classes"])


def read_frozen_probabilities() -> dict:
    if hashlib.sha256(SOURCE_PREDICTIONS.read_bytes()).hexdigest() != PROTOCOL["parent_prediction_sha256"]:
        raise AssertionError("parent frozen probabilities changed")
    output = {}
    with SOURCE_PREDICTIONS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["arm"], row["trial_id"])
            if key in output:
                raise AssertionError("duplicate frozen native prediction")
            output[key] = np.asarray([float(row[f"p_{label}"]) for label in CLASSES])
    if len(output) != 1792:
        raise AssertionError("expected 4 arms by 448 target native recordings")
    return output


def extract(data_root: Path) -> tuple[list[dict], dict[str, np.ndarray], dict[str, int]]:
    grab.check_all_files(data_root)
    records = list(grab.records())
    if len(records) != 672:
        raise AssertionError("unexpected GRABMyo recording set")
    rest_windows = np.concatenate([grab.read_record(data_root, record) for record in records
                                   if record["session"] == 1 and record["gesture"] == 17])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    families = {
        "F0": RestNoiseDetailV2(rest_label=17).fit(rest_batch, np.full(rest_batch.windows, 17)),
        "F2a": TraceCovarianceV2(shrinkage=0.05).fit(rest_batch),
        "F3c": RingRelativeCovarianceV2(shrinkage=0.05).fit(rest_batch),
    }
    vectors = {name: [] for name in families}
    for count, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(aggregate(family.transform(batch)))
        if count % 100 == 0 or count == len(records):
            print(f"feature-space calibration extracted {count}/{len(records)} recordings", flush=True)
    arrays = {name: np.stack(values) for name, values in vectors.items()}
    dims = {name: int(value.shape[1]) for name, value in arrays.items()}
    return records, arrays, dims


def anchor_probability(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    count = query.shape[0]
    distances = np.sum((prototypes - query[None, :]) ** 2, axis=1) / count
    pair_distances = np.asarray([np.sum((prototypes[a] - prototypes[b]) ** 2) / count
                                 for a in range(len(CLASSES)) for b in range(a + 1, len(CLASSES))])
    positive = pair_distances[pair_distances > 1e-12]
    temperature = float(np.median(positive)) if len(positive) else 1.0
    logits = -distances / max(temperature, 1e-12)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def build(data_root: Path, verify: bool = False) -> dict:
    if hashlib.sha256((ROOT / "GRABMYO_PROTOCOL.json").read_bytes()).hexdigest() != PROTOCOL["parent_grabmyo_protocol_sha256"]:
        raise AssertionError("parent GRABMyo protocol changed")
    if hashlib.sha256((ROOT / "GRAB_SCORE_CAL_PROTOCOL.json").read_bytes()).hexdigest() != PROTOCOL["score_space_comparator_protocol_sha256"]:
        raise AssertionError("score-space comparison protocol changed")
    if (tuple(PROTOCOL["subjects"]) != grab.SUBJECTS or CLASSES != grab.GESTURES or
            PROTOCOL["source_day"] != 1 or PROTOCOL["target_phases"] != {"validation": 2, "final": 3} or
            PROTOCOL["evaluation_repetitions"] != [6, 7]):
        raise AssertionError("protocol disagrees with official GRABMyo subset")
    frozen = read_frozen_probabilities()
    records, vectors, dimensions = extract(data_root)
    sessions = np.asarray([r["session"] for r in records])
    labels = np.asarray([r["gesture"] for r in records])
    source = sessions == 1
    if source.sum() != 224:
        raise AssertionError("wrong source recording count")
    if {int(s): int((sessions == s).sum()) for s in (1, 2, 3)} != {1: 224, 2: 224, 3: 224}:
        raise AssertionError("day split changed")
    all_rows, groups = [], []
    max_replay_error = 0.0
    for arm in ARMS:
        matrix = np.concatenate([vectors[name] for name in arm.split("+")], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=20260924))
        model.fit(matrix[source], labels[source])
        if not np.array_equal(model[-1].classes_, CLASSES) or model[-1].n_iter_.max() >= 2000:
            raise AssertionError("source classifier class order or convergence changed")
        standardized = model[0].transform(matrix)
        target_indices = np.flatnonzero(~source)
        recomputed = model.predict_proba(matrix[target_indices])
        for index, probability in zip(target_indices, recomputed):
            key = (arm, records[index]["stem"])
            difference = float(np.max(np.abs(probability - frozen[key])))
            max_replay_error = max(max_replay_error, difference)
            if difference > 1e-10:
                raise AssertionError(f"source model does not replay frozen prediction: {key}")
        record_index = {(r["session"], r["subject"], r["gesture"], r["trial"]): i
                        for i, r in enumerate(records)}
        if len(record_index) != len(records):
            raise AssertionError("duplicate native recording metadata")
        for phase, day in PROTOCOL["target_phases"].items():
            for budget in PROTOCOL["budgets"]:
                phase_rows = []
                for subject in PROTOCOL["subjects"]:
                    prototypes = None
                    if budget:
                        prototypes = np.stack([np.mean([
                            standardized[record_index[(day, subject, label, rep)]]
                            for rep in range(1, budget + 1)], axis=0)
                            for label in CLASSES])
                    for label in CLASSES:
                        for rep in PROTOCOL["evaluation_repetitions"]:
                            index = record_index[(day, subject, label, rep)]
                            trial_id = records[index]["stem"]
                            population = frozen[(arm, trial_id)]
                            probability = population if budget == 0 else (
                                (1 - PROTOCOL["anchor_weight"]) * population +
                                PROTOCOL["anchor_weight"] * anchor_probability(
                                    standardized[index], prototypes))
                            if not np.all(np.isfinite(probability)) or np.any(probability < 0) or not np.isclose(probability.sum(), 1, atol=1e-12):
                                raise AssertionError("invalid feature-anchor probability")
                            phase_rows.append({"arm": arm, "phase": phase, "subject": subject,
                                               "shots_per_class": budget, "trial_id": trial_id,
                                               "label": label, "repetition": rep,
                                               **{f"p_{c}": float(value) for c, value in zip(CLASSES, probability)}})
                if len(phase_rows) != 64:
                    raise AssertionError("expected 64 fixed evaluation recordings")
                all_rows.extend(phase_rows)
                groups.append({"arm": arm, "phase": phase, "subject": "ALL",
                               "shots_per_class": budget, **score(phase_rows)})
                for subject in PROTOCOL["subjects"]:
                    subset = [row for row in phase_rows if row["subject"] == subject]
                    groups.append({"arm": arm, "phase": phase, "subject": subject,
                                   "shots_per_class": budget, **score(subset)})
        print(f"feature-space calibration {arm}: source replay max error {max_replay_error:.2g}", flush=True)
    if len(all_rows) != 2048 or len(groups) != 288:
        raise AssertionError("unexpected output row counts")
    curve = [{**{key: value for key, value in row.items() if key != "per_class_f1"},
              "per_class_f1_json": json.dumps(row["per_class_f1"], sort_keys=True)}
             for row in groups]
    outputs = {"GRAB_FEATURE_CAL_PREDICTIONS.csv": all_rows,
               "GRAB_FEATURE_CAL_CURVE.csv": curve}
    for name, output in outputs.items():
        content = csv_text(output)
        path = ROOT / name
        if verify:
            if path.read_bytes() != content.encode("utf-8"):
                raise AssertionError(f"feature-space result changed: {name}")
        else:
            path.write_bytes(content.encode("utf-8"))
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "parent_prediction_sha256": hashlib.sha256(SOURCE_PREDICTIONS.read_bytes()).hexdigest(),
             "official_checksum_manifest_sha256": hashlib.sha256((data_root / "SHA256SUMS.txt").read_bytes()).hexdigest(),
             "source_recordings": 224, "target_recordings": 448,
             "feature_dimensions": dimensions, "source_model_replay_max_abs_error": max_replay_error,
             "zero_shot_exact_replays": 512,
             "rows": {name: len(output) for name, output in outputs.items()},
             "split": "Day1 source; each target day repetitions 1..N calibration, 6/7 evaluation",
             "boundary": PROTOCOL["boundary"]}
    path = ROOT / "GRAB_FEATURE_CAL_AUDIT.json"
    content = json.dumps(audit, indent=2) + "\n"
    if verify:
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError("feature-space audit changed")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify,
                      "rows": audit["rows"], "max_source_replay_error": max_replay_error}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    build(args.data_root, args.verify)
