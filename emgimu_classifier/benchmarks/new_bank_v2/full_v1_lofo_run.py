"""Source-only full five-family bank and complete LOFO on frozen public axes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.grabmyo_crossday import run as grab
from benchmarks.new_bank_v2 import grab_day_v1_extension_run as day_parent
from benchmarks.new_bank_v2 import grab_v1_extension_run as user_parent
from benchmarks.new_bank_v2 import roam_v1_extension_run as roam_parent
from benchmarks.new_bank_v2 import force_v1_extension_run as force_parent
from benchmarks.new_bank_v2 import manus_v1_speed_run as speed_parent
from benchmarks.new_bank_v2 import wearing_v1_extension_run as wearing_parent
from benchmarks.new_bank_v2 import roam_v1_quality_run as quality_parent
from benchmarks.new_bank_v2.roam_quality_run import aggregate_families, fault, packed
from benchmarks.new_bank_v2.roam_posture_run import extract_archive, sha256
from benchmarks.new_bank_v2.manus_rest_transfer_run import external_rest
from benchmarks.new_bank_v2.manus_spatial_run import extract as extract_manus
from emgimu.datasets.libemg_force import FILE_RE, load_libemg_force_windows
from emgimu.datasets.semg_manus import load_semg_manus_windows
from emgimu.datasets.electrode_shift import PATH_RE
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v1 import NEW_BANK_V1
from emgimu.feature_bank.new_bank_v2 import RestNoiseDetailV2

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "FULL_V1_LOFO_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = PROTOCOL["arms"]


def verify_parent(axis: str) -> dict:
    name = PROTOCOL["axes"][axis]
    result_path = ROOT / f"{name}_RESULTS.json"
    prediction_path = ROOT / f"{name}_PREDICTIONS.csv"
    if (sha256(result_path) != PROTOCOL["parent_results_sha256"][name]
            or sha256(prediction_path) != PROTOCOL["parent_predictions_sha256"][name]):
        raise AssertionError(f"frozen full-bank parent changed: {axis}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def prepare_roam(parent: dict) -> dict:
    roam_parent.check_protocol()
    protocol = parent["protocol"]
    archive = Path(protocol["archive"])
    if sha256(archive).lower() != protocol["archive_sha256"].lower():
        raise AssertionError("ROAM native archive changed")
    batch, labels, identities, slices, _ = extract_archive(archive)
    subjects = np.asarray([row["subject"] for row in identities])
    postures = np.asarray([row["condition"] for row in identities])
    source = np.isin(subjects, protocol["source_subjects"]) & (postures == "resting")
    rest = np.concatenate([batch.emg[a:b] for i, (a, b) in enumerate(slices)
                           if source[i] and labels[i] == 0])
    rest_batch = FeatureBatch(rest, 200)
    if int(source.sum()) != 162 or rest_batch.windows != 1828:
        raise AssertionError("ROAM full-bank source changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                **{name: constructor().fit(rest_batch) for name, constructor in NEW_BANK_V1.items()}}
    vectors = {name: np.stack([grab.aggregate(values[a:b]) for a, b in slices])
               for name, family in families.items() for values in [family.transform(batch)]}
    masks = {phase: np.isin(subjects, protocol[f"{phase}_subjects"])
             for phase in ("validation", "final")}
    if (any(int(mask.sum()) != 180 for mask in masks.values())
            or [identities[i]["trial_id"] for i in np.flatnonzero(masks["validation"])]
            != parent["validation_trial_ids"]):
        raise AssertionError("ROAM full-bank target split changed")
    return {"vectors": vectors, "labels": labels, "subjects": subjects,
            "conditions": postures, "trial_ids": [row["trial_id"] for row in identities],
            "source": source, "targets": masks, "classes": np.asarray([0, 1, 2]),
            "class_weight": "balanced", "seed": 20260929,
            "source_rest_windows": rest_batch.windows}


def prepare_grab(axis: str, parent: dict) -> dict:
    if axis == "unseen_user":
        user_parent.check_protocol()
    else:
        day_parent.check_protocol()
    protocol = parent["protocol"]
    data_root = Path(protocol["data_root"])
    records = [row for row in grab.records()
               if axis == "cross_day" or row["session"] == protocol["day"]]
    if len(records) != (224 if axis == "unseen_user" else 672):
        raise AssertionError("GRAB native recording inventory changed")
    if axis == "unseen_user":
        user_parent.check_files(data_root, records)
    else:
        grab.check_all_files(data_root)
    rest_rows = [row for row in records if row["gesture"] == 17 and
                 (row["subject"] in protocol["source_subjects"] if axis == "unseen_user"
                  else row["session"] == 1)]
    rest_batch = FeatureBatch(np.concatenate([grab.read_record(data_root, row)
                                              for row in rest_rows]), grab.RATE)
    if rest_batch.windows != (560 if axis == "unseen_user" else 1120):
        raise AssertionError("GRAB source Rest windows changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(
                    rest_batch, np.full(rest_batch.windows, 17)),
                **{name: constructor().fit(rest_batch) for name, constructor in NEW_BANK_V1.items()}}
    vectors = {name: [] for name in families}
    for index, record in enumerate(records, 1):
        batch = FeatureBatch(grab.read_record(data_root, record), grab.RATE)
        for name, family in families.items():
            vectors[name].append(grab.aggregate(family.transform(batch)))
        if index % 100 == 0 or index == len(records):
            print(f"full bank {axis} extracted {index}/{len(records)} recordings", flush=True)
    vectors = {name: np.stack(values) for name, values in vectors.items()}
    subjects = np.asarray([row["subject"] for row in records])
    sessions = np.asarray([row["session"] for row in records])
    source = (np.isin(subjects, protocol["source_subjects"])
              if axis == "unseen_user" else sessions == 1)
    masks = ({phase: np.isin(subjects, protocol[f"{phase}_subjects"])
              for phase in ("validation", "final")}
             if axis == "unseen_user" else
             {phase: sessions == protocol["sessions"][phase]
              for phase in ("validation", "final")})
    expected = 56 if axis == "unseen_user" else 224
    if (int(source.sum()) != (112 if axis == "unseen_user" else 224)
            or any(int(mask.sum()) != expected for mask in masks.values())
            or [records[i]["stem"] for i in np.flatnonzero(masks["validation"])]
            != parent["validation_trial_ids"]):
        raise AssertionError("GRAB full-bank split changed")
    return {"vectors": vectors,
            "labels": np.asarray([row["gesture"] for row in records]),
            "subjects": subjects, "conditions": sessions.astype(str),
            "trial_ids": [row["stem"] for row in records], "source": source,
            "targets": masks, "classes": np.asarray(grab.GESTURES),
            "class_weight": None, "seed": 20260924,
            "source_rest_windows": rest_batch.windows}


def prepare_force(parent: dict) -> dict:
    force_parent.check_protocol()
    protocol = parent["protocol"]
    source = load_libemg_force_windows(force_parent.RAW,
                                       subjects=protocol["source_subjects"],
                                       conditions=protocol["source_conditions"])
    if (source.batch.channels != 8 or source.batch.sample_rate_hz != 1000
            or source.batch.emg.shape[1] != 200):
        raise AssertionError("force source signal contract changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(source.batch, source.labels),
                **{name: constructor().fit(source.batch, source.labels)
                   for name, constructor in NEW_BANK_V1.items()}}
    source_vectors, source_y, source_users, source_ids = force_parent.fit_vectors(source, families)
    if set(source_ids) != set(parent["split_trial_ids"]["validation"]["source_trials"]):
        raise AssertionError("force source native trials changed")
    targets = {}
    for phase in ("validation", "final"):
        data = load_libemg_force_windows(force_parent.RAW,
                                         subjects=protocol[f"{phase}_subjects"],
                                         conditions=protocol["target_conditions"])
        vectors, y, users, ids = force_parent.fit_vectors(data, families)
        if set(ids) != set(parent["split_trial_ids"][phase]["target_trials"]):
            raise AssertionError(f"force {phase} native trials changed")
        targets[phase] = vectors, y, users, ids
    vectors = {name: np.concatenate([source_vectors[name],
                                     targets["validation"][0][name],
                                     targets["final"][0][name]]) for name in families}
    labels = np.concatenate([source_y, targets["validation"][1], targets["final"][1]])
    users = np.concatenate([source_users, targets["validation"][2], targets["final"][2]])
    ids = np.concatenate([source_ids, targets["validation"][3], targets["final"][3]])
    n_source, n_validation = len(source_y), len(targets["validation"][1])
    source_mask = np.arange(len(labels)) < n_source
    masks = {"validation": (np.arange(len(labels)) >= n_source)
             & (np.arange(len(labels)) < n_source + n_validation),
             "final": np.arange(len(labels)) >= n_source + n_validation}
    conditions = []
    for trial in ids:
        match = FILE_RE.fullmatch(str(trial) + ".csv")
        if match is None:
            raise AssertionError("force native trial condition unavailable")
        conditions.append(match["condition"])
    return {"vectors": vectors, "labels": labels, "subjects": users,
            "conditions": np.asarray(conditions), "trial_ids": ids,
            "source": source_mask, "targets": masks,
            "classes": np.asarray(protocol["classes"]),
            "class_weight": "balanced", "seed": 20260924,
            "source_rest_windows": int(np.sum(source.labels == 0))}


def prepare_speed(parent: dict) -> dict:
    speed_parent.check_protocol()
    protocol = parent["protocol"]
    rest = external_rest()
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest, np.zeros(rest.windows, dtype=int)),
                **{name: constructor().fit(rest) for name, constructor in NEW_BANK_V1.items()}}
    source_protocol = json.loads((ROOT / "MANUS_REST_TRANSFER_PROTOCOL.json").read_text(encoding="utf-8"))
    archive = Path(source_protocol["manus_archive"])
    arguments = {"users": source_protocol["users"], "gestures": source_protocol["gestures"],
                 "speeds": source_protocol["conditions"],
                 "window_ms": source_protocol["window_ms"],
                 "maximum_windows_per_trial": source_protocol["maximum_windows_per_trial"]}
    data = [load_semg_manus_windows(archive, sessions=(session,), **arguments)
            for session in (1, 2, 3)]
    extracted = [extract_manus(item, families) for item in data]
    vectors = {name: np.concatenate([item[0][name] for item in extracted]) for name in families}
    meta = [item[1] for item in extracted]
    labels = np.concatenate([item[0] for item in meta])
    users = np.concatenate([item[1] for item in meta])
    speeds = np.concatenate([item[3] for item in meta])
    ids = np.concatenate([item[4] for item in meta])
    if (any(len(item[0]) != 108 for item in meta)
            or set(ids[108:216]) != set(parent["split_trial_ids"]["validation"]["target_trials"])
            or set(ids[216:]) != set(parent["split_trial_ids"]["final"]["target_trials"])):
        raise AssertionError("MANUS speed native split changed")
    indices = np.arange(len(labels))
    return {"vectors": vectors, "labels": labels, "subjects": users,
            "conditions": speeds, "trial_ids": ids,
            "source": indices < 108,
            "targets": {"validation": (indices >= 108) & (indices < 216),
                        "final": indices >= 216},
            "classes": np.asarray(protocol["classes"]),
            "class_weight": "balanced", "seed": 20260924,
            "source_rest_windows": rest.windows}


def replay_baseline(axis: str, rows: list[dict], classes: np.ndarray) -> float:
    name = PROTOCOL["axes"][axis]
    with (ROOT / f"{name}_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                    if r["arm"] == "F0v2"}
    baseline = [row for row in rows if row["arm"] == "F0v2"]
    if len(previous) != len(baseline):
        raise AssertionError("full-bank F0v2 replay coverage changed")
    maximum = 0.0
    for row in baseline:
        prior = previous[(row["phase"], row["trial_id"])]
        if int(prior.get("label", prior.get("gesture"))) != row["label"]:
            raise AssertionError("full-bank baseline class changed")
        p = np.asarray([row[f"p_{c}"] for c in classes])
        q = np.asarray([float(prior[f"p_{c}"]) for c in classes])
        maximum = max(maximum, float(np.max(np.abs(p - q))))
    if maximum > 1e-8:
        raise AssertionError(f"full-bank frozen F0v2 changed: {maximum}")
    return maximum


def write_result(axis: str, rows: list[dict], result: dict) -> dict:
    path = ROOT / f"FULL_V1_LOFO_{axis}_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result.update({"axis": axis, "protocol_sha256": sha256(PROTOCOL_PATH),
                   "parent_result_sha256": PROTOCOL["parent_results_sha256"][PROTOCOL["axes"][axis]],
                   "parent_prediction_sha256": PROTOCOL["parent_predictions_sha256"][PROTOCOL["axes"][axis]],
                   "prediction_sha256": sha256(path), "prediction_rows": len(rows),
                   "scope": PROTOCOL["boundary"]})
    (ROOT / f"FULL_V1_LOFO_{axis}_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_wearing() -> dict:
    axis = "wearing_shift"
    parent = verify_parent(axis)
    wearing_parent.check_protocol()
    protocol = parent["protocol"]
    classes = np.asarray(protocol["classes"])
    rows, splits, dimensions = [], {}, {}
    for phase in ("validation", "final"):
        for subject in protocol[f"{phase}_subjects"]:
            source = wearing_parent.load(wearing_parent.ARCHIVE, subject,
                                         (protocol["source_domain"],))
            target = wearing_parent.load(wearing_parent.ARCHIVE, subject,
                                         tuple(protocol["target_domains"]))
            if (source.batch.channels != 8 or source.batch.sample_rate_hz != 200
                    or source.batch.emg.shape[1] != 40 or set(source.labels) != set(classes)):
                raise AssertionError("wearing source signal contract changed")
            families = {"F0v2": RestNoiseDetailV2(rest_label=2).fit(source.batch, source.labels),
                        **{name: factory().fit(source.batch, source.labels)
                           for name, factory in NEW_BANK_V1.items()}}
            xs, ys, source_users, source_trials = wearing_parent.vectors(source, families)
            xt, yt, target_users, target_trials = wearing_parent.vectors(target, families)
            key = f"{phase}_{subject}"
            if (set(source_trials) & set(target_trials)
                    or set(source_trials) != set(parent["split_trial_ids"][key]["source"])
                    or set(target_trials) != set(parent["split_trial_ids"][key]["target"])
                    or set(source_users) != {subject} or set(target_users) != {subject}):
                raise AssertionError("wearing frozen native split changed")
            splits[key] = {"source": source_trials.tolist(), "target": target_trials.tolist()}
            for name, values in xs.items():
                if name in dimensions and dimensions[name] != values.shape[1]:
                    raise AssertionError("wearing feature dimension changed")
                dimensions[name] = int(values.shape[1])
            for arm, members in ARMS.items():
                x = np.concatenate([xs[name] for name in members], axis=1)
                x_target = np.concatenate([xt[name] for name in members], axis=1)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    random_state=20260924))
                model.fit(x, ys)
                np.testing.assert_array_equal(model[-1].classes_, classes)
                if np.max(model[-1].n_iter_) >= 2000:
                    raise RuntimeError(f"wearing full bank failed to converge: {key}/{arm}")
                for trial_id, label, p in zip(target_trials, yt, model.predict_proba(x_target)):
                    match = PATH_RE.fullmatch(str(trial_id))
                    if match is None or int(match["subject"]) != subject or int(match["label"]) != label:
                        raise AssertionError("wearing native trial metadata changed")
                    rows.append({"axis": axis, "phase": phase, "arm": arm,
                                 "trial_id": str(trial_id), "subject": subject,
                                 "condition": match["domain"], "label": int(label),
                                 **{f"p_{c}": float(value) for c, value in zip(classes, p)}})
            print(f"full bank wearing {key}: {len(target_trials)} target trials", flush=True)
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    with (ROOT / "WEARING_V1_EXTENSION_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = {(r["phase"], r["trial_id"]): r for r in csv.DictReader(stream)
                    if r["arm"] == "F0v2"}
    if len(baseline) != 240 or len(previous) != 240:
        raise AssertionError("wearing baseline coverage changed")
    maximum = 0.0
    for row in baseline:
        old = previous[(row["phase"], row["trial_id"])]
        if (row["subject"], row["condition"], row["label"]) != (
                int(old["subject"]), old["domain"], int(old["label"])):
            raise AssertionError("wearing baseline identity changed")
        maximum = max(maximum, max(abs(row[f"p_{c}"] - float(old[f"p_{c}"])) for c in classes))
    if maximum > 1e-8:
        raise AssertionError(f"wearing F0v2 replay changed: {maximum}")
    scores = {arm: {phase: wearing_parent._score(
        [{**r, "domain": r["condition"]} for r in rows], phase, arm)
                    for phase in ("validation", "final")} for arm in ARMS}
    return write_result(axis, rows, {"feature_dimensions": dimensions,
                                    "split_trial_ids": splits,
                                    "f0v2_parent_replay_max_abs_error": maximum,
                                    "scores": scores})


def evaluate_quality() -> dict:
    axis = "synthetic_quality"
    parent = verify_parent(axis)
    protocol = parent["protocol"]
    native_protocol = quality_parent.check_protocol()
    archive = Path(native_protocol["archive"])
    if sha256(archive).lower() != native_protocol["archive_sha256"].lower():
        raise AssertionError("ROAM quality archive changed")
    batch, labels, identities, slices, counts = extract_archive(archive)
    users = np.asarray([r["subject"] for r in identities])
    postures = np.asarray([r["condition"] for r in identities])
    source_indices = np.flatnonzero(np.isin(users, protocol["source_subjects"])
                                    & (postures == "resting"))
    source_windows, source_slices = packed(batch, slices, source_indices)
    rest = [source_windows[a:b] for j, (a, b) in enumerate(source_slices)
            if labels[source_indices[j]] == 0]
    rest_batch = FeatureBatch(np.concatenate(rest), 200)
    if len(source_indices) != 162 or rest_batch.windows != 1828:
        raise AssertionError("ROAM quality source changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(
                    rest_batch, np.zeros(rest_batch.windows, dtype=int)),
                **{name: factory().fit(rest_batch) for name, factory in NEW_BANK_V1.items()}}
    source_vectors = aggregate_families(source_windows, source_slices, families)
    q95 = np.quantile(np.abs(source_windows.astype(np.float64)), 0.95, axis=(0, 1)).astype(np.float32)
    rms = np.sqrt(np.mean(source_windows.astype(np.float64) ** 2, axis=(0, 1))).astype(np.float32)
    np.testing.assert_allclose(q95, parent["source_reference_q95"], rtol=0, atol=0)
    np.testing.assert_allclose(rms, parent["source_reference_rms"], rtol=0, atol=0)
    classes = np.asarray(protocol["classes"])
    models = {}
    for arm, members in ARMS.items():
        x = np.concatenate([source_vectors[name] for name in members], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=20260929))
        model.fit(x, labels[source_indices])
        np.testing.assert_array_equal(model[-1].classes_, classes)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"quality source model did not converge: {arm}")
        models[arm] = model
    rows = []
    scores = {arm: {phase: {} for phase in ("validation", "final")} for arm in ARMS}
    split_ids = {}
    for phase in ("validation", "final"):
        indices = np.flatnonzero(np.isin(users, protocol[f"{phase}_subjects"])
                                     & (postures == "resting"))
        trial_ids = [identities[int(i)]["trial_id"] for i in indices]
        if len(indices) != 45 or trial_ids != parent["split_trial_ids"][phase]:
            raise AssertionError("ROAM quality target changed")
        split_ids[phase] = trial_ids
        target_windows, target_slices = packed(batch, slices, indices)
        for condition in protocol["conditions"]:
            vectors = aggregate_families(fault(target_windows, condition, q95, rms),
                                         target_slices, families)
            for arm, members in ARMS.items():
                probability = models[arm].predict_proba(
                    np.concatenate([vectors[name] for name in members], axis=1))
                scores[arm][phase][condition] = grab.score(
                    labels[indices], probability, classes, users[indices])
                for i, p in zip(indices, probability):
                    identity = identities[int(i)]
                    rows.append({"axis": axis, "phase": phase, "arm": arm,
                                 "trial_id": identity["trial_id"],
                                 "subject": int(identity["subject"]),
                                 "condition": condition, "label": int(identity["label"]),
                                 **{f"p_{c}": float(value) for c, value in zip(classes, p)}})
            print(f"full bank quality {phase} {condition}", flush=True)
    with (ROOT / "ROAM_V1_QUALITY_PREDICTIONS.csv").open(newline="", encoding="utf-8") as stream:
        previous = {(r["phase"], r["condition"], r["trial_id"]): r
                    for r in csv.DictReader(stream) if r["arm"] == "F0v2"}
    baseline = [r for r in rows if r["arm"] == "F0v2"]
    if len(previous) != 1260 or len(baseline) != 1260:
        raise AssertionError("quality baseline coverage changed")
    maximum = 0.0
    for row in baseline:
        old = previous[(row["phase"], row["condition"], row["trial_id"])]
        if (row["subject"], row["label"]) != (int(old["subject"]), int(old["label"])):
            raise AssertionError("quality baseline identity changed")
        maximum = max(maximum, max(abs(row[f"p_{c}"] - float(old[f"p_{c}"])) for c in classes))
    if maximum > 1e-10:
        raise AssertionError(f"quality F0v2 replay changed: {maximum}")
    summary = {arm: {phase: quality_parent.summarize(scores[arm][phase])
                     for phase in ("validation", "final")} for arm in ARMS}
    return write_result(axis, rows, {"source_trial_ids": [identities[int(i)]["trial_id"]
                                                         for i in source_indices],
                                    "target_trial_ids": split_ids,
                                    "source_rest_windows": rest_batch.windows,
                                    "source_reference_q95": q95.tolist(),
                                    "source_reference_rms": rms.tolist(),
                                    "native_static_files_verified": len(counts),
                                    "feature_dimensions": {name: int(x.shape[1])
                                                           for name, x in source_vectors.items()},
                                    "f0v2_parent_replay_max_abs_error": maximum,
                                    "scores": scores, "summary": summary})


def evaluate(axis: str) -> dict:
    if axis == "wearing_shift":
        return evaluate_wearing()
    if axis == "synthetic_quality":
        return evaluate_quality()
    parent = verify_parent(axis)
    if axis == "posture":
        data = prepare_roam(parent)
    elif axis in ("unseen_user", "cross_day"):
        data = prepare_grab(axis, parent)
    elif axis == "force_intensity":
        data = prepare_force(parent)
    else:
        data = prepare_speed(parent)
    vectors = data["vectors"]
    labels = data["labels"]
    subjects = data["subjects"]
    classes = data["classes"]
    rows, scores = [], {}
    for arm, members in ARMS.items():
        x = np.concatenate([vectors[name] for name in members], axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight=data["class_weight"], max_iter=2000,
            random_state=data["seed"]))
        model.fit(x[data["source"]], labels[data["source"]])
        np.testing.assert_array_equal(model[-1].classes_, classes)
        if np.max(model[-1].n_iter_) >= 2000:
            raise RuntimeError(f"full-bank source model did not converge: {axis}/{arm}")
        scores[arm] = {}
        for phase in ("validation", "final"):
            mask = data["targets"][phase]
            probability = model.predict_proba(x[mask])
            scores[arm][phase] = grab.score(labels[mask], probability, classes,
                                            subjects[mask])
            for i, p in zip(np.flatnonzero(mask), probability):
                rows.append({"axis": axis, "phase": phase, "arm": arm,
                             "trial_id": data["trial_ids"][i],
                             "subject": int(subjects[i]),
                             "condition": str(data["conditions"][i]),
                             "label": int(labels[i]),
                             **{f"p_{c}": float(value) for c, value in zip(classes, p)}})
        print(f"full bank {axis} {arm}: validation F1={scores[arm]['validation']['macro_f1']:.4f}, "
              f"final F1={scores[arm]['final']['macro_f1']:.4f}", flush=True)
    replay_error = replay_baseline(axis, rows, classes)
    path = ROOT / f"FULL_V1_LOFO_{axis}_PREDICTIONS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"axis": axis, "protocol_sha256": sha256(PROTOCOL_PATH),
              "parent_result_sha256": PROTOCOL["parent_results_sha256"][PROTOCOL["axes"][axis]],
              "parent_prediction_sha256": PROTOCOL["parent_predictions_sha256"][PROTOCOL["axes"][axis]],
              "prediction_sha256": sha256(path), "prediction_rows": len(rows),
              "source_rest_windows": data["source_rest_windows"],
              "feature_dimensions": {name: int(value.shape[1]) for name, value in vectors.items()},
              "source_trial_ids": [data["trial_ids"][i] for i in np.flatnonzero(data["source"])],
              "target_trial_ids": {phase: [data["trial_ids"][i] for i in np.flatnonzero(mask)]
                                   for phase, mask in data["targets"].items()},
              "f0v2_parent_replay_max_abs_error": replay_error,
              "scores": scores, "scope": PROTOCOL["boundary"]}
    (ROOT / f"FULL_V1_LOFO_{axis}_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("axis", choices=list(PROTOCOL["axes"]))
    arguments = parser.parse_args()
    evaluate(arguments.axis)
