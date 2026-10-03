"""Matched source-trained classifiers for Song F0 Rest-only threshold policy."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.document_signal import RestNoiseLocalDetailFamily
from emgimu.feature_bank.families import LocalDetailFamily


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "F0_REST_MODEL_PROTOCOL.json"
PARENT = HERE / "F0_REST_NOISE_PROTOCOL.json"
RESULT = HERE / "F0_REST_MODEL_RESULTS.json"
CSV = HERE / "F0_REST_MODEL_TRIAL_PREDICTIONS.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(rows: list[dict], classes: list[str]) -> dict:
    y = np.asarray([row["hand"] for row in rows])
    p = np.asarray([[row[f"p_{label}"] for label in classes] for row in rows])
    if not len(y) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid trial probabilities")
    true_index = np.asarray([classes.index(label) for label in y])
    pred = np.asarray(classes)[p.argmax(axis=1)]
    return {"trials": len(y), "macro_f1": float(f1_score(y, pred, labels=classes,
                                                         average="macro", zero_division=0)),
            "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), true_index], 1e-15, 1)).mean()),
            "brier": float(np.mean(np.sum((p - (y[:, None] == np.asarray(classes))) ** 2, axis=1))),
            "recall": {label: float(value) for label, value in zip(classes, recall_score(
                y, pred, labels=classes, average=None, zero_division=0))}}


def run(source: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if sha(PARENT) != protocol["parent_f0_protocol_sha256"]:
        raise ValueError("parent F0 protocol changed")
    source_sessions = protocol["source_sessions"]
    target_sessions = protocol["read_only_sessions"]
    if source_sessions != parent["source_sessions"] or target_sessions != parent["evaluation_sessions"]:
        raise ValueError("native session partition changed")
    f9 = json.loads((HERE / "F9_DOCUMENT_V3_PROTOCOL.json").read_text(encoding="utf-8"))
    data = {}
    for session in source_sessions + target_sessions:
        item = load_session(source / f"2026-09-18_{session}", session, filter_mode="causal")
        if (item["audit"]["sha256"] != f9["expected_session_sha256"][session]
                or len(item["trial"]) != f9["expected_formal_windows"][session]):
            raise ValueError(f"Song native source changed: {session}")
        data[session] = item
        print(f"F0 model loaded {session}: {len(item['trial'])} windows", flush=True)
    source_ids = set(np.concatenate([data[s]["trial"] for s in source_sessions]).tolist())
    if source_ids & set(np.concatenate([data[s]["trial"] for s in target_sessions]).tolist()):
        raise ValueError("source/target trial overlap")
    x = np.concatenate([data[s]["batch"].emg for s in source_sessions])
    y = np.concatenate([data[s]["hand"] for s in source_sessions])
    if set(y) != set(protocol["classes"]):
        raise ValueError("source class inventory changed")
    source_batch = FeatureBatch(x, parent["sample_rate_hz"])
    families = {
        "pooled_source_threshold": LocalDetailFamily().fit(source_batch),
        "rest_only_threshold": RestNoiseLocalDetailFamily(rest_label=parent["rest_label"]).fit(source_batch, y),
    }
    matched = json.loads((HERE / "F0_REST_NOISE_RESULTS.json").read_text(encoding="utf-8"))
    if (not np.array_equal(families["pooled_source_threshold"].thresholds_, matched["pooled_source_thresholds"])
            or not np.array_equal(families["rest_only_threshold"].thresholds_, matched["rest_only_thresholds"])):
        raise ValueError("source-fitted thresholds disagree with frozen feature comparison")
    rows = []
    fitted = {}
    for arm in protocol["arms"]:
        family = families[arm]
        matrix = family.transform(source_batch)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(matrix, y)
        frozen = pickle.dumps((family, model))
        fitted[arm] = hashlib.sha256(frozen).hexdigest()
        for session in target_sessions:
            item = data[session]
            probabilities = model.predict_proba(family.transform(item["batch"]))
            classes = list(model[-1].classes_)
            probabilities = probabilities[:, [classes.index(label) for label in protocol["classes"]]]
            for trial_id in dict.fromkeys(item["trial"]):
                indices = np.flatnonzero(item["trial"] == trial_id)
                if len(set(item["hand"][indices])) != 1:
                    raise ValueError("mixed native trial labels")
                p = probabilities[indices].mean(axis=0)
                rows.append({"session": session, "trial_id": trial_id, "arm": arm,
                             "hand": item["hand"][indices[0]], "windows": len(indices),
                             **{f"p_{label}": float(value) for label, value in zip(protocol["classes"], p)}})
            print(f"F0 model {arm} {session}: {len(set(item['trial']))} trials", flush=True)
        if pickle.dumps((family, model)) != frozen:
            raise ValueError("target inference mutated fitted source state")
    fields = ["session", "trial_id", "arm", "hand", "windows"] + [f"p_{label}" for label in protocol["classes"]]
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {session: {arm: scores([row for row in rows if row["session"] == session and row["arm"] == arm],
                                      protocol["classes"]) for arm in protocol["arms"]}
               for session in target_sessions}
    result = {"status": "retrospective_paired_local_only", "protocol_sha256": sha(PROTOCOL),
              "parent_f0_protocol_sha256": sha(PARENT),
              "source_hdf5_sha256": {s: f9["expected_session_sha256"][s] for s in source_sessions},
              "sklearn_version": sklearn.__version__, "source_state_sha256": fitted,
              "prediction_rows": len(rows), "prediction_csv_sha256": sha(CSV),
              "metrics": metrics, "boundary": protocol["boundary"]}
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    run(args.source)
