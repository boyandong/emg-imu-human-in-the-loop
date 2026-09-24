"""Bound the DS2 30-trial gap without assigning an unverified subject identity.

The candidate long TDMS group is chosen from adjacent MAT/TDMS join order only.
Correlation is a diagnostic, not an identity criterion or force annotation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve


MISSING = list(range(389, 419))
LONG_MEMBER = "SEMG-04/SEMG-04_Mv1.tdms"
CONTROL_TRIAL = 388
SAMPLES = 15000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_pearson(template: np.ndarray, signal: np.ndarray) -> tuple[int, float]:
    """Signed strongest normalized Pearson correlation over full-length offsets."""
    x = np.asarray(template, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    width = len(x)
    if width > len(y) or np.std(x) == 0:
        raise ValueError("invalid correlation input")
    x = x - x.mean()
    cumulative = np.concatenate(([0.0], np.cumsum(y)))
    cumulative2 = np.concatenate(([0.0], np.cumsum(y * y)))
    sums = cumulative[width:] - cumulative[:-width]
    sums2 = cumulative2[width:] - cumulative2[:-width]
    energy = np.maximum(sums2 - sums * sums / width, 0)
    numerator = fftconvolve(y, x[::-1], mode="valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(numerator, np.sqrt(energy) * np.linalg.norm(x),
                         out=np.zeros_like(numerator), where=energy > 1e-20)
    index = int(np.argmax(np.abs(corr)))
    return index, float(corr[index])


def run(archive_path: Path, raw_path: Path, join_csv: Path, join_audit: Path,
        mat_join_csv: Path, output: Path) -> dict:
    from nptdms import TdmsFile

    prior = json.loads(join_audit.read_text(encoding="utf-8"))
    if (prior["unmatched_raw_trial_indices_zero_based"] != MISSING
            or prior["nonzero_offset_exact_subsequence_matches"]
            or sha256(archive_path) != prior["source_archive_sha256"]
            or sha256(raw_path) != prior["source_raw_mat_sha256"]
            or sha256(join_csv) != prior["join_csv_sha256"]
            or sha256(mat_join_csv) != prior["source_trial_window_join_sha256"]):
        raise ValueError("exact-join inputs or source bytes changed")
    rows = {int(row["raw_trial_index_zero_based"]): row for row in
            csv.DictReader(join_csv.open(newline="", encoding="utf-8"))}
    label_rows = {int(row["raw_trial_index_zero_based"]): row for row in
                  csv.DictReader(mat_join_csv.open(newline="", encoding="utf-8"))}
    if (rows[CONTROL_TRIAL]["subject_folder"] != "3"
            or any(index in rows or label_rows[index]["gesture_label_if_uniform"] != "0"
                   for index in MISSING)
            or rows[MISSING[-1] + 1]["subject_folder"] != "4"):
        raise ValueError("the positional context of the unmatched block changed")
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    if raw.shape != (2863, 3, SAMPLES):
        raise ValueError("raw MAT shape changed")
    with zipfile.ZipFile(archive_path) as archive:
        control = TdmsFile.read(io.BytesIO(archive.read(rows[CONTROL_TRIAL]["tdms_member"])))
        control_group = control.groups()[int(rows[CONTROL_TRIAL]["tdms_group_ordinal_zero_based"])]
        control_signal = control_group.channels()[0][:]
        if not np.array_equal(raw[CONTROL_TRIAL, 0], control_signal[:SAMPLES]):
            raise ValueError("positive control is no longer an exact waveform join")
        control_offset, control_corr = peak_pearson(raw[CONTROL_TRIAL, 0], control_signal)
        if control_offset != 0 or not np.isclose(control_corr, 1.0, atol=1e-10):
            raise ValueError("correlation positive control failed")
        candidate = TdmsFile.read(io.BytesIO(archive.read(LONG_MEMBER)))
        if len(candidate.groups()) != 1:
            raise ValueError("candidate TDMS group structure changed")
        channels = candidate.groups()[0].channels()
        if len(channels) != 3 or any(len(channel) != 980100 for channel in channels):
            raise ValueError("candidate long group shape changed")
        main_signal = channels[0][:]
        best_first_channel = []
        for trial in MISSING:
            offset, corr = peak_pearson(raw[trial, 0], main_signal)
            best_first_channel.append({"raw_trial_index_zero_based": trial,
                                       "best_offset": offset, "signed_pearson": corr})
        cross_channel_checks = {}
        for trial in (389, 406, 418):
            matrix = [[peak_pearson(raw[trial, raw_channel], channels[tdms_channel][:])[1]
                       for tdms_channel in range(3)] for raw_channel in range(3)]
            cross_channel_checks[str(trial)] = matrix
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "candidate_long_group_not_verified_as_source",
        "source_archive_sha256": prior["source_archive_sha256"],
        "source_raw_mat_sha256": prior["source_raw_mat_sha256"],
        "exact_join_csv_sha256": prior["join_csv_sha256"],
        "mat_trial_window_join_csv_sha256": prior["source_trial_window_join_sha256"],
        "candidate_member": LONG_MEMBER,
        "candidate_group_samples_per_channel": len(main_signal),
        "basis_for_candidate_only": "30 consecutive gesture-code-0 MAT trials occur between exact subject-folder-3 and subject-folder-4 joins; this is not an identity key",
        "positive_control": {"raw_trial_index_zero_based": CONTROL_TRIAL,
                             "best_offset": control_offset, "signed_pearson": control_corr},
        "unmatched_first_channel": best_first_channel,
        "max_absolute_first_channel_pearson": max(abs(row["signed_pearson"])
                                                  for row in best_first_channel),
        "cross_channel_probe_trials": cross_channel_checks,
        "max_absolute_cross_channel_probe_pearson": max(abs(value)
            for matrix in cross_channel_checks.values() for row in matrix for value in row),
        "boundary": "The 30 MAT trials have no exact TDMS waveform join. Their weak strongest linear correlations with this positional candidate, even after arbitrary time shift and selected channel swaps, do not establish a source, subject, or force level. Nonlinear preprocessing or another source is not excluded. Leave subject unknown.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "max_abs_first_channel_r": result["max_absolute_first_channel_pearson"],
                      "control_r": control_corr}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("archive", "raw_mat", "join_csv", "join_audit", "mat_join_csv", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    run(args.archive, args.raw_mat, args.join_csv, args.join_audit, args.mat_join_csv, args.output)
