"""Join publisher-added DS2 v9 force labels to exact v8 raw-trial windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


TRIALS = 2863
WINDOWS_PER_TRIAL = 116
FORCE_NAMES = {0: "low", 1: "average", 2: "high"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(v9_dir: Path, discovery: Path) -> dict:
    release_path = discovery / "DS2_V9_RELEASE_METADATA.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release["pinned_version"] != 9 or
            "Forces are listed from 0 to 2 (Low, Average, High)" not in
            release["force_label_description"]):
        raise ValueError("publisher's v9 force-code semantics changed")
    previous = json.loads((discovery / "DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json").read_text(encoding="utf-8"))
    join_path = discovery / "DS2_MAT_TRIAL_WINDOW_JOIN.csv"
    if sha256(join_path) != previous["trial_join_csv_sha256"]:
        raise ValueError("v8 raw-to-window join changed")
    identical_files = {}
    for filename, prior_key in (("Feature_MAV_all.mat", "mav_mat_sha256"),
                                ("Window_label_gestures.mat", "gesture_label_mat_sha256")):
        path = v9_dir / filename
        digest = sha256(path)
        if digest != previous["source_sha256"][prior_key]:
            raise ValueError(f"v9 {filename} differs from the exact v8 raw-window join")
        identical_files[filename] = digest
    force_path = v9_dir / "LabelForces_All.mat"
    force_digest = sha256(force_path)
    force = loadmat(force_path, variable_names=["y"])["y"]
    if force.shape != (TRIALS * WINDOWS_PER_TRIAL, 1) or not np.issubdtype(force.dtype, np.integer):
        raise ValueError("publisher force label array shape/type changed")
    force = force.reshape(TRIALS, WINDOWS_PER_TRIAL)
    if not set(np.unique(force)).issubset(FORCE_NAMES):
        raise ValueError("publisher force codes changed")
    if not np.all(force == force[:, :1]):
        raise ValueError("at least one raw trial has mixed force labels")

    rows = list(csv.DictReader(join_path.open(encoding="utf-8", newline="")))
    subject_path = discovery / "DS2_TDMS_RAW_EXACT_JOIN.csv"
    subjects = {int(row["raw_trial_index_zero_based"]): int(row["subject_folder"])
                for row in csv.DictReader(subject_path.open(encoding="utf-8", newline=""))}
    if len(rows) != TRIALS or len(subjects) != 2833:
        raise ValueError("raw-trial or subject join count changed")
    output_rows = []
    subject_gesture_force = Counter()
    for index, row in enumerate(rows):
        if (int(row["raw_trial_index_zero_based"]) != index or
                int(row["first_window_index_zero_based"]) != index * WINDOWS_PER_TRIAL or
                int(row["last_window_index_zero_based"]) != (index + 1) * WINDOWS_PER_TRIAL - 1):
            raise ValueError("noncontiguous raw-to-window join")
        code = int(force[index, 0])
        gesture = row["gesture_label_if_uniform"]
        subject = subjects.get(index)
        if subject is not None and gesture != "N/A":
            subject_gesture_force[(subject, int(gesture), code)] += 1
        output_rows.append({
            "raw_trial_index_zero_based": index,
            "subject_folder": subject if subject is not None else "N/A",
            "gesture_code_if_uniform": gesture,
            "force_code": code,
            "force_name_publisher": FORCE_NAMES[code],
            "first_window_index_zero_based": index * WINDOWS_PER_TRIAL,
            "last_window_index_zero_based": (index + 1) * WINDOWS_PER_TRIAL - 1,
        })
    output = discovery / "DS2_V9_FORCE_TRIAL_JOIN.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    subject_counts = {str(subject): {str(gesture): {str(code): subject_gesture_force[
        (subject, gesture, code)] for code in FORCE_NAMES} for gesture in range(5)}
        for subject in range(1, 21)}
    audit = {
        "status": "publisher_v9_force_codes_exactly_joined_to_public_raw_trials",
        "publisher_dataset": "https://www.kaggle.com/datasets/cinthyazuniga/ds2-emg-signals-three-force-type/versions/9",
        "publisher_version": 9,
        "publisher_version_date_utc": "2026-09-25T02:38:56.137Z",
        "force_code_names": {str(k): v for k, v in FORCE_NAMES.items()},
        "source_sha256": {
            "publisher_release_metadata": sha256(release_path),
            "v9_force_mat": force_digest,
            "v9_identical_to_v8": identical_files,
            "v8_raw_mat": previous["source_sha256"]["raw_mat_sha256"],
            "v8_raw_to_window_join_csv": sha256(join_path),
            "v8_tdms_subject_join_csv": sha256(subject_path),
        },
        "window_contract": previous["window_contract"],
        "raw_trials": TRIALS,
        "force_window_labels": int(force.size),
        "uniform_force_trials": TRIALS,
        "force_trial_counts_all": {str(code): int(np.sum(force[:, 0] == code)) for code in FORCE_NAMES},
        "unmatched_subject_trials": [i for i in range(TRIALS) if i not in subjects],
        "mixed_gesture_trials": [int(row["raw_trial_index_zero_based"]) for row in rows
                                  if row["gesture_label_if_uniform"] == "N/A"],
        "eligible_subject_gesture_force_trials": sum(subject_gesture_force.values()),
        "subject_gesture_force_counts": subject_counts,
        "trial_join_csv": output.name,
        "trial_join_csv_sha256": sha256(output),
        "boundary": "The v9 publisher file supplies three force codes per published feature window. Its MAV and gesture MAT files are byte-identical to v8, whose 116-window blocks were exactly reconstructed from public raw trials. This establishes force codes for all 2863 public raw-MAT trials. Only 2832 trials have both a verified TDMS subject folder and one uniform gesture code. Force is an instructed subjective category, not measured mechanical force; historical B0/X1-H/X2 input identity remains unproven.",
    }
    (discovery / "DS2_V9_FORCE_LABEL_AUDIT.json").write_bytes(
        (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"force_labeled_trials": TRIALS,
                      "eligible_subject_gesture_force_trials": sum(subject_gesture_force.values())}))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-dir", required=True, type=Path)
    parser.add_argument("--discovery", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.v9_dir, args.discovery)


if __name__ == "__main__":
    main()
