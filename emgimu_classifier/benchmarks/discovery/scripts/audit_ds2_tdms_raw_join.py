"""Match public DS2 raw-MAT trials to TDMS groups by exact three-channel signal.

Only the first 15,000 samples of each TDMS group are candidates. A 64-sample
fingerprint narrows the search; all 45,000 floating-point values must match
before a subject folder is assigned. Force and historical-run identities are
never inferred from group position.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import loadmat


TRIALS = 2863
CHANNELS = 3
SAMPLES = 15000
PREFIX = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(signal: np.ndarray) -> bytes:
    if signal.shape != (CHANNELS, PREFIX):
        raise ValueError("Expected a three-channel 64-sample prefix")
    return hashlib.blake2b(signal.tobytes(), digest_size=16).digest()


def audit(archive_path: Path, raw_path: Path, trial_join_csv: Path,
          native_audit_path: Path, archive_audit_path: Path, output: Path) -> dict:
    from nptdms import TdmsFile

    native = json.loads(native_audit_path.read_text(encoding="utf-8"))
    archive_audit = json.loads(archive_audit_path.read_text(encoding="utf-8"))
    if sha256(raw_path) != native["raw_mat_sha256"]:
        raise ValueError("Raw MAT source hash changed")
    archive_sha256 = sha256(archive_path)
    if archive_sha256 != archive_audit["archive_sha256"]:
        raise ValueError("Public archive hash changed")
    raw = loadmat(raw_path, variable_names=["data_final_all"])["data_final_all"]
    if raw.shape != (TRIALS, CHANNELS, SAMPLES) or not np.isfinite(raw).all():
        raise ValueError("Unexpected raw MAT signal")
    label_rows = list(csv.DictReader(trial_join_csv.open(encoding="utf-8", newline="")))
    if len(label_rows) != TRIALS or any(
            int(row["raw_trial_index_zero_based"]) != i for i, row in enumerate(label_rows)):
        raise ValueError("Trial-window join index changed")

    fingerprints: dict[bytes, list[int]] = defaultdict(list)
    for i in range(TRIALS):
        fingerprints[fingerprint(raw[i, :, :PREFIX].copy())].append(i)
    prefix_collisions = sum(len(v) > 1 for v in fingerprints.values())

    rows = []
    eligible_groups = 0
    unmatched_eligible_groups = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(info.filename for info in archive.infolist()
                       if info.filename.lower().endswith(".tdms"))
        if len(names) != 97:
            raise ValueError("Expected 97 TDMS members")
        for file_number, member in enumerate(names, 1):
            match = re.fullmatch(r"SEMG-(\d{2})/.*\.tdms", member)
            if not match:
                raise ValueError(f"Unknown TDMS member path: {member}")
            subject = int(match.group(1))
            tdms = TdmsFile.read(io.BytesIO(archive.read(member)))
            member_matches = 0
            for ordinal, group in enumerate(tdms.groups()):
                channels = group.channels()
                if len(channels) != CHANNELS or any(len(c) < SAMPLES for c in channels):
                    continue
                eligible_groups += 1
                if [c.name for c in channels] != ["Sensor 1", "Sensor 2", "Sensor 3"]:
                    raise ValueError(f"Unexpected TDMS channel order in {member} group {ordinal}")
                key = fingerprint(np.stack([c[:PREFIX] for c in channels]))
                exact = [i for i in fingerprints.get(key, []) if all(
                    np.array_equal(channels[c][:SAMPLES], raw[i, c])
                    for c in range(CHANNELS))]
                if not exact:
                    unmatched_eligible_groups += 1
                    continue
                for i in exact:
                    member_matches += 1
                    rows.append({"raw_trial_index_zero_based": i,
                                 "subject_folder": subject,
                                 "tdms_member": member,
                                 "tdms_group_ordinal_zero_based": ordinal,
                                 "tdms_group_name": group.name,
                                 "tdms_samples_per_channel": len(channels[0]),
                                 "matched_samples_per_channel": SAMPLES,
                                 "gesture_code_if_uniform": label_rows[i]["gesture_label_if_uniform"]})
            print(f"TDMS {file_number}/{len(names)} {member}: {member_matches} exact matches", flush=True)

    rows.sort(key=lambda row: (row["raw_trial_index_zero_based"], row["tdms_member"],
                               row["tdms_group_ordinal_zero_based"]))
    counts = Counter(row["raw_trial_index_zero_based"] for row in rows)
    unmatched = [i for i in range(TRIALS) if counts[i] == 0]
    duplicated = [i for i in range(TRIALS) if counts[i] > 1]
    # A first-sample-only search would miss a repetition embedded in a long
    # TDMS group. Search every possible exact starting position for remaining
    # MAT trials, then compare the entire three-channel waveform again.
    subsequence_matches = []
    if unmatched:
        prefixes: dict[tuple[float, ...], list[int]] = defaultdict(list)
        for i in unmatched:
            prefixes[tuple(raw[i, 0, :4])].append(i)
        first_values = np.array([raw[i, 0, 0] for i in unmatched])
        with zipfile.ZipFile(archive_path) as archive:
            for member in names:
                tdms = TdmsFile.read(io.BytesIO(archive.read(member)))
                for ordinal, group in enumerate(tdms.groups()):
                    channels = group.channels()
                    if len(channels) != CHANNELS or len(channels[0]) <= SAMPLES:
                        continue
                    signal = channels[0][:]
                    starts = np.flatnonzero(np.isin(signal[:-3], first_values))
                    for start in starts:
                        if start + SAMPLES > len(signal):
                            continue
                        for i in prefixes.get(tuple(signal[start:start + 4]), []):
                            if all(np.array_equal(channels[c][start:start + SAMPLES], raw[i, c])
                                   for c in range(CHANNELS)):
                                subsequence_matches.append({"raw_trial_index_zero_based": i,
                                                            "tdms_member": member,
                                                            "tdms_group_ordinal_zero_based": ordinal,
                                                            "start_sample_zero_based": int(start)})
        print(f"Nonzero-offset search: {len(subsequence_matches)} exact matches", flush=True)
        if subsequence_matches:
            raise ValueError("Nonzero-offset exact matches require explicit join review")
    subject_counts = Counter(row["subject_folder"] for row in rows
                             if counts[row["raw_trial_index_zero_based"]] == 1)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "DS2_TDMS_RAW_EXACT_JOIN.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["raw_trial_index_zero_based", "subject_folder",
                                 "tdms_member", "tdms_group_ordinal_zero_based",
                                 "tdms_group_name", "tdms_samples_per_channel",
                                 "matched_samples_per_channel", "gesture_code_if_uniform"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "full_exact_join" if not unmatched and not duplicated else "partial_exact_join",
        "source_archive_sha256": archive_sha256,
        "source_raw_mat_sha256": native["raw_mat_sha256"],
        "source_trial_window_join_sha256": sha256(trial_join_csv),
        "tdms_members": len(names),
        "eligible_groups_at_least_15000_samples": eligible_groups,
        "eligible_groups_without_exact_mat_match": unmatched_eligible_groups,
        "raw_trials": TRIALS,
        "exact_signal_matches": len(rows),
        "uniquely_matched_raw_trials": sum(count == 1 for count in counts.values()),
        "unmatched_raw_trial_indices_zero_based": unmatched,
        "multiply_matched_raw_trial_indices_zero_based": duplicated,
        "prefix_fingerprint_collision_keys": prefix_collisions,
        "unmatched_trials_searched_at_nonzero_tdms_group_offsets": len(unmatched),
        "nonzero_offset_exact_subsequence_matches": subsequence_matches,
        "unique_subject_trial_counts": {str(i): subject_counts[i] for i in range(1, 21)},
        "join_csv": csv_path.name,
        "join_csv_sha256": sha256(csv_path),
        "boundary": "Each assigned match compares all 45000 three-channel raw samples exactly, beginning at TDMS group sample zero. Unmatched MAT trials are additionally sought at every nonzero starting sample of sufficiently long TDMS groups; none is assigned without an exact match. Subject folder identity is supported only for uniquely matched trials. The TDMS group name and file movement suffix do not establish per-trial low/medium/high force. Public-v8 identity with historical B0/X1-H/X2 input remains unproven.",
    }
    (output / "DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "exact_signal_matches", "uniquely_matched_raw_trials",
        "eligible_groups_without_exact_mat_match", "prefix_fingerprint_collision_keys")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--raw-mat", required=True, type=Path)
    parser.add_argument("--trial-join-csv", required=True, type=Path)
    parser.add_argument("--native-audit", required=True, type=Path)
    parser.add_argument("--archive-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.archive, args.raw_mat, args.trial_join_csv, args.native_audit,
          args.archive_audit, args.output)
