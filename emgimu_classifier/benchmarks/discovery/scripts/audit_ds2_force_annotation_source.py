"""Audit whether public DS2 v8 actually supplies per-trial force annotations."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import h5py
from scipy.io import whosmat


API = "https://www.kaggle.com/api/v1/datasets/view/cinthyazuniga/ds2-emg-signals-three-force-type"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mat_inventory(member: bytes) -> list[dict]:
    try:
        return [{"variable": name, "shape": list(shape), "type": kind}
                for name, shape, kind in whosmat(io.BytesIO(member))]
    except NotImplementedError:
        with h5py.File(io.BytesIO(member), "r") as file:
            return [{"variable": name, "shape_on_disk": list(file[name].shape),
                     "type": str(file[name].dtype), "format": "MATLAB v7.3 HDF5"}
                    for name in file.keys()]


def audit(archive_path: Path, archive_audit_path: Path,
          tdms_metadata_path: Path, tdms_first_path: Path,
          mat_join_path: Path, tdms_join_path: Path,
          output_path: Path) -> dict:
    request = urllib.request.Request(API, headers={"User-Agent": "Codex-DS2-provenance-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    description = metadata["description"]
    if metadata["currentVersionNumber"] != 8 or not isinstance(description, str):
        raise ValueError("Publisher-linked Kaggle metadata version/description changed")
    archive_audit = json.loads(archive_audit_path.read_text(encoding="utf-8"))
    if sha256(archive_path) != archive_audit["archive_sha256"]:
        raise ValueError("Public DS2 archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        mat_names = sorted(name for name in names if name.lower().endswith(".mat"))
        tdms_names = [name for name in names if name.lower().endswith(".tdms")]
        if len(names) != 102 or len(mat_names) != 5 or len(tdms_names) != 97:
            raise ValueError("Public DS2 member inventory changed")
        variables = {name: mat_inventory(archive.read(name)) for name in mat_names}
    expected = {
        "Data_all_Raw.mat": {"data_final_all"},
        "EMG_WS_all.mat": {"final_matrix"},
        "Feature_AAV_all.mat": {"matrix_all_AAV"},
        "Feature_MAV_all.mat": {"matrix_all_MAV"},
        "Window_label_gestures.mat": {"y"},
    }
    if {name: {item["variable"] for item in contents}
        for name, contents in variables.items()} != expected:
        raise ValueError("MAT variables no longer match five signal/gesture containers")
    tdms = json.loads(tdms_metadata_path.read_text(encoding="utf-8"))
    first = json.loads(tdms_first_path.read_text(encoding="utf-8"))
    mat_join = json.loads(mat_join_path.read_text(encoding="utf-8"))
    tdms_join = json.loads(tdms_join_path.read_text(encoding="utf-8"))
    if (tdms["groups"] != 3210 or tdms["group_property_count_distribution"] != {"0": 3210}
            or tdms["file_property_keys"] != {"name": 97}
            or mat_join["uniform_gesture_label_trials"] != 2862
            or tdms_join["uniquely_matched_raw_trials"] != 2833):
        raise ValueError("Published signal and label provenance changed")
    file_names = list(first["first_segment_name_values"])
    force_clues = [name for name in file_names if re.search(
        r"force|fuerza|\b(low|medium|normal|high|baja|media|alta)\b",
        name, re.IGNORECASE)]
    if force_clues:
        raise ValueError("New force clue exists in TDMS file-level names; review needed")
    required_description_fragments = (
        "three force levels (low, normal, and high)",
        "10 repetitions per gesture at each force level",
        "120 samples per subject",
        "Each subject has approximately 150 samples",
        "Each window has a 375 ms length",
    )
    if any(fragment not in description for fragment in required_description_fragments):
        raise ValueError("Kaggle description changed; review force annotation audit")
    result = {
        "status": "force_protocol_declared_but_per_trial_annotation_unverified",
        "publisher_linked_api": API,
        "kaggle_version": metadata["currentVersionNumber"],
        "kaggle_description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "source_archive_sha256": archive_audit["archive_sha256"],
        "source_evidence_sha256": {
            "archive_audit": sha256(archive_audit_path),
            "tdms_group_audit": sha256(tdms_metadata_path),
            "tdms_first_metadata_audit": sha256(tdms_first_path),
            "raw_mav_join_audit": sha256(mat_join_path),
            "tdms_raw_join_audit": sha256(tdms_join_path),
        },
        "archive_files": len(names),
        "mat_files": len(mat_names),
        "tdms_files": len(tdms_names),
        "mat_variables": variables,
        "tdms_groups": tdms["groups"],
        "tdms_groups_without_group_properties": tdms["group_property_count_distribution"]["0"],
        "tdms_file_level_property_keys": tdms["file_property_keys"],
        "tdms_file_level_names_with_force_clue": force_clues,
        "raw_trials_with_verified_subject_folder": tdms_join["uniquely_matched_raw_trials"],
        "raw_trials_with_uniform_gesture_code": mat_join["uniform_gesture_label_trials"],
        "publisher_description_discrepancies": [
            "The same dataset description says 120 samples per subject and approximately 150 samples per subject; 5 gestures x 30 repetitions is 150 before exclusions.",
            "The description calls a 375-sample window at 1500 Hz 375 ms; arithmetic gives 250 ms.",
            "The description names low/normal/high while the publication names low/medium/high; these textual names do not create per-trial labels.",
        ],
        "per_trial_force_label_source": None,
        "force_order_within_each_gesture": None,
        "boundary": "The publisher describes three subjective force conditions and 10 attempts per gesture per condition, but does not give a trial-to-condition key. All five MAT variables are signal, feature or five-code gesture arrays; inspected TDMS groups have no properties and file-level names have no force clue. Subject and gesture joins do not establish low/medium/high trial labels. No force task or historical B0/X1-H/X2 reproduction may be claimed without an authoritative annotation/order mapping and old-input identity.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "mat_files": len(mat_names),
                      "tdms_groups_without_properties": result["tdms_groups_without_group_properties"],
                      "per_trial_force_label_source": result["per_trial_force_label_source"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-audit", required=True, type=Path)
    parser.add_argument("--tdms-groups", required=True, type=Path)
    parser.add_argument("--tdms-first", required=True, type=Path)
    parser.add_argument("--mat-join", required=True, type=Path)
    parser.add_argument("--tdms-join", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.archive, args.archive_audit, args.tdms_groups, args.tdms_first,
          args.mat_join, args.tdms_join, args.output)
