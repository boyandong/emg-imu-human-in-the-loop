"""Matched MANUS/GRAB native transfer of the document-exact uncentered F2a."""
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
from benchmarks.new_bank_v1.run import fit_predict
from benchmarks.new_bank_v2 import f2_manus_candidates_run as manus_parent
from benchmarks.new_bank_v2 import grab_user_run as grab_parent
from benchmarks.new_bank_v2.manus_spatial_run import extract, score as manus_score
from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.semg_manus import PATH_RE, load_semg_manus_windows
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.new_bank_v2 import DocumentTraceCovarianceV2, RestNoiseDetailV2


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "F2A_DOCUMENT_TRANSFER_PROTOCOL.json"
PREDICTIONS = ROOT / "F2A_DOCUMENT_TRANSFER_PREDICTIONS.csv"
RESULT = ROOT / "F2A_DOCUMENT_TRANSFER_RESULTS.json"


def evaluate(data_root: Path) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for name, key in (("F2_MANUS_CANDIDATES_RESULTS.json", "manus_parent_result_sha256"),
                      ("F2_MANUS_CANDIDATES_PREDICTIONS.csv", "manus_parent_prediction_sha256"),
                      ("F2_GRAB_CANDIDATES_RESULTS.json", "grab_parent_result_sha256"),
                      ("F2_GRAB_CANDIDATES_PREDICTIONS.csv", "grab_parent_prediction_sha256")):
        if sha256(ROOT / name) != protocol[key]:
            raise AssertionError(f"Frozen transfer parent changed: {name}")
    rows = []
    manus_parent.check_protocol()
    m = manus_parent.PARENT
    m_result = json.loads((ROOT / "F2_MANUS_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    archive = Path(m["manus_archive"])
    if sha256(archive).lower() != m["manus_archive_sha256"].lower():
        raise AssertionError("MANUS native archive changed")
    rest = manus_parent.external_rest()
    arguments = dict(users=m["users"], gestures=m["gestures"], speeds=m["conditions"],
                     window_ms=m["window_ms"], maximum_windows_per_trial=m["maximum_windows_per_trial"])
    source = load_semg_manus_windows(archive, sessions=(m["source_session"],), **arguments)
    if source.batch.channels != 8 or source.batch.sample_rate_hz != 200 or source.batch.emg.shape[1] != 40:
        raise AssertionError("MANUS source sensor/window contract changed")
    families = {"F0v2": RestNoiseDetailV2(rest_label=0).fit(rest, np.zeros(rest.windows, dtype=int)),
                "F2a_uncentered": DocumentTraceCovarianceV2(shrinkage=.05).fit(source.batch)}
    source_x, (source_y, _, _, _, source_trials) = extract(source, families)
    manus_scores = {}
    for phase in ("validation", "final"):
        session = m[f"{phase}_session"]
        target = load_semg_manus_windows(archive, sessions=(session,), **arguments)
        target_x, (target_y, target_users, target_sessions, target_speeds, target_trials) = extract(target, families)
        frozen = m_result["split_trial_ids"][phase]
        if (source_trials.tolist() != frozen["source_trials"] or
                target_trials.tolist() != frozen["target_trials"] or
                set(source_trials) & set(target_trials) or set(target_sessions) != {session}):
            raise AssertionError("MANUS matched native trial split changed")
        x = np.concatenate((source_x["F0v2"], source_x["F2a_uncentered"]), axis=1)
        xt = np.concatenate((target_x["F0v2"], target_x["F2a_uncentered"]), axis=1)
        classes, probabilities = fit_predict(x, source_y, xt)
        np.testing.assert_array_equal(classes, np.arange(6))
        for trial, user, speed, label, probability in zip(
                target_trials, target_users, target_speeds, target_y, probabilities):
            native = PATH_RE.fullmatch(str(trial))
            if (native is None or int(native["user"]) != user or
                    int(native["session"]) != session or native["speed"] != speed or
                    native["gesture"] != m["gestures"][label]):
                raise AssertionError("MANUS native target metadata changed")
            rows.append({"axis": "manus_session", "phase": phase, "subject": int(user),
                         "condition": str(speed), "trial_id": str(trial), "label": int(label),
                         "class_order": "0|1|2|3|4|5",
                         **{f"p_{c}": float(p) for c, p in zip(classes, probability)}})
        manus_scores[phase] = manus_score(
            [{**row, "arm": protocol["candidate"]} for row in rows if row["axis"] == "manus_session"],
            phase, protocol["candidate"])
        print(f"F2a document MANUS {phase}: {len(target_trials)} held-out trials", flush=True)

    grab_parent.check_protocol()
    g = grab_parent.PROTOCOL
    g_result = json.loads((ROOT / "F2_GRAB_CANDIDATES_RESULTS.json").read_text(encoding="utf-8"))
    records = [row for row in grab.records() if row["session"] == g["day"]]
    if len(records) != 224 or grab_parent.check_files(data_root, records) != g_result["official_sha256_manifest_sha256"]:
        raise AssertionError("GRAB official native file inventory changed")
    windows = [grab.read_record(data_root, row) for row in records]
    subjects = np.asarray([row["subject"] for row in records])
    labels = np.asarray([row["gesture"] for row in records])
    source_mask = np.isin(subjects, g["source_subjects"])
    if [row["stem"] for row, selected in zip(records, source_mask) if selected] != g_result["source_trial_ids"]:
        raise AssertionError("GRAB source native trial IDs changed")
    source_windows = np.concatenate([window for window, selected in zip(windows, source_mask) if selected])
    rest_windows = np.concatenate([window for window, selected, label in zip(windows, source_mask, labels)
                                   if selected and label == g["rest_code"]])
    rest_batch = FeatureBatch(rest_windows, grab.RATE)
    source_batch = FeatureBatch(source_windows, grab.RATE)
    g_families = {"F0v2": RestNoiseDetailV2(rest_label=17).fit(rest_batch,
                                                                 np.full(rest_batch.windows,17)),
                  "F2a_uncentered": DocumentTraceCovarianceV2(shrinkage=.05).fit(source_batch)}
    vectors = {name: np.stack([grab.aggregate(family.transform(FeatureBatch(window,grab.RATE)))
                               for window in windows]) for name, family in g_families.items()}
    features = np.concatenate((vectors["F0v2"], vectors["F2a_uncentered"]), axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., max_iter=2000, random_state=20260924))
    model.fit(features[source_mask],labels[source_mask])
    classes = np.asarray(grab.GESTURES)
    np.testing.assert_array_equal(model[-1].classes_, classes)
    if np.max(model[-1].n_iter_) >= 2000:
        raise RuntimeError("GRAB document F2a source classifier did not converge")
    grab_scores = {}
    for phase in ("validation", "final"):
        target = np.isin(subjects, g[f"{phase}_subjects"])
        ids = [row["stem"] for row, selected in zip(records, target) if selected]
        if ids != g_result[f"{phase}_trial_ids"] or np.any(target & source_mask):
            raise AssertionError("GRAB held-out native trial IDs changed")
        p = model.predict_proba(features[target])
        grab_scores[phase] = {"trials": int(target.sum()),
                              **grab.score(labels[target],p,classes,subjects[target])}
        for row, probability in zip(np.asarray(records,dtype=object)[target],p):
            rows.append({"axis": "grab_unseen_user", "phase": phase,
                         "subject": int(row["subject"]), "condition": "Day1",
                         "trial_id": row["stem"], "label": int(row["gesture"]),
                         "class_order": "4|15|16|17",
                         **{f"p_{i}": float(value) for i,value in enumerate(probability)},
                         "p_4": "", "p_5": ""})
        print(f"F2a document GRAB {phase}: {target.sum()} held-out trials", flush=True)
    with PREDICTIONS.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    result={"status":"document_uncentered_f2a_matched_transfer",
            "protocol_sha256":sha256(PROTOCOL_PATH),
            "manus_parent_result_sha256":protocol["manus_parent_result_sha256"],
            "grab_parent_result_sha256":protocol["grab_parent_result_sha256"],
            "prediction_sha256":sha256(PREDICTIONS),"prediction_rows":len(rows),
            "feature_dimensions":{"F0v2":48,"F2a_uncentered":36},
            "scores":{"manus_session":manus_scores,"grab_unseen_user":grab_scores},
            "scope":protocol["boundary"]}
    RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(rows),
                      "manus_validation_f1":manus_scores["validation"]["pooled"]["macro_f1"],
                      "grab_validation_f1":grab_scores["validation"]["macro_f1"]}))
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--grab-data-root",type=Path,
                        default=Path("D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1"))
    evaluate(parser.parse_args().grab_data_root)
