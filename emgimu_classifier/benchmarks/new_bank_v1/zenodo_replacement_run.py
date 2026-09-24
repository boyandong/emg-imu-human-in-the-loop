"""File-level, strictly QC-filtered electrode-position confirmation.

Whole-file movement intent is not a time-local gesture label. This runner does
not manufacture repetition boundaries or classify selected windows as trials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.electrode_replacement import (
    load_electrode_replacement_recording, parse_recording_name,
)
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.new_bank_v1 import CorrelationSpectrumV1, RingLagV1


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "ZENODO_REPLACEMENT_PROTOCOL.json").read_text(encoding="utf-8"))
RAW_ROOT = Path(__file__).resolve().parents[4] / "work/secondary_raw/electrode_replacement"
ARCHIVE = RAW_ROOT / "EMG dataset.7z"
EXTRACTED = RAW_ROOT / "full_native_v1"
FACTORIES = {"F0": LocalDetailFamily, "ring_lag": RingLagV1,
             "correlation_spectrum": CorrelationSpectrumV1}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inventory():
    import py7zr

    if digest(ARCHIVE) != PROTOCOL["archive_sha256"]:
        raise ValueError("verified official electrode-replacement archive changed")
    with py7zr.SevenZipFile(ARCHIVE) as archive:
        names = archive.getnames()
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or ":" in name or "\\" in name:
            raise ValueError("unsafe 7z member")
    recordings = sorted(name for name in names if name.endswith(".txt"))
    identities = [parse_recording_name(name) for name in recordings]
    if len(recordings) != 267 or len(set(identities)) != 267:
        raise ValueError("unexpected native recording inventory")
    return recordings


def acquire():
    import py7zr

    names = inventory()
    missing = [name for name in names if not (EXTRACTED / name).exists()]
    if missing:
        EXTRACTED.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(ARCHIVE) as archive:
            archive.extract(path=EXTRACTED, targets=missing, recursive=False)
    absent = [name for name in names if not (EXTRACTED / name).is_file()]
    if absent:
        raise ValueError(f"extracted recording missing: {absent[0]}")
    print(f"verified archive; {len(names)} recordings available; extracted {len(missing)} now", flush=True)


def windows(signal):
    width = int(PROTOCOL["window_samples"])
    count = min(int(PROTOCOL["selected_windows_per_recording"]), len(signal) // width)
    if count <= 0:
        raise ValueError("recording shorter than one window")
    starts = np.linspace(0, len(signal) - width, count).round().astype(int)
    if len(starts) > 1 and np.any(np.diff(starts) < width):
        raise ValueError("window selection would overlap")
    return FeatureBatch(np.stack([signal[start:start + width] for start in starts]),
                        float(PROTOCOL["sample_rate_hz"]))


def descriptors(batch_by_code, source_codes):
    source = FeatureBatch(np.concatenate([batch_by_code[(code, "P1")].emg for code in source_codes]),
                          float(PROTOCOL["sample_rate_hz"]))
    families = {name: factory().fit(source) for name, factory in FACTORIES.items()}
    result = {}
    for key, batch in batch_by_code.items():
        result[key] = {name: family.transform(batch).mean(axis=0)
                       for name, family in families.items()}
    return result


def evaluate():
    names = inventory()
    missing = [name for name in names if not (EXTRACTED / name).is_file()]
    if missing:
        raise ValueError("run acquire first")
    lookup = {parse_recording_name(name): name for name in names}
    rows, rejections, by_subject, exclusions = [], [], {}, []
    for subject in PROTOCOL["subjects"]:
        batches = {}
        for code in PROTOCOL["movement_codes"]:
            for position in ("P1", "P2", "P3"):
                key = (subject, code, position)
                if key not in lookup:
                    rejections.append({"subject": subject, "movement": code, "position": position,
                                       "reason": "recording absent from official archive"})
                    continue
                try:
                    record = load_electrode_replacement_recording(EXTRACTED / lookup[key])
                    batches[(code, position)] = windows(record.emg)
                except ValueError as error:
                    rejections.append({"subject": subject, "movement": code, "position": position,
                                       "reason": str(error)})
        source_codes = [code for code in PROTOCOL["movement_codes"] if (code, "P1") in batches]
        if len(source_codes) < 2:
            exclusions.append({"subject": subject, "phase": "both", "reason":
                               "fewer than two complete P1 gallery movement recordings",
                               "valid_gallery_classes": len(source_codes)})
            print(f"subject {subject}: excluded by source-only recording QC", flush=True)
            continue
        vectors = descriptors(batches, source_codes)
        by_subject[str(subject)] = {}
        for phase, position in (("validation", "P2"), ("final", "P3")):
            matched = [code for code in source_codes if (code, position) in batches]
            if len(matched) < 2:
                exclusions.append({"subject": subject, "phase": phase, "reason":
                                   "fewer than two matched complete gallery/target movement recordings",
                                   "matched_classes": len(matched)})
                by_subject[str(subject)][phase] = {"matched_movement_codes": matched,
                    "gallery_position": "P1", "target_position": position, "excluded": True}
                continue
            by_subject[str(subject)][phase] = {"matched_movement_codes": matched,
                                               "gallery_position": "P1", "target_position": position}
            for arm in PROTOCOL["arms"]:
                members = arm.split("+")
                gallery = np.stack([np.concatenate([vectors[(code, "P1")][name] for name in members])
                                    for code in matched])
                query = np.stack([np.concatenate([vectors[(code, position)][name] for name in members])
                                  for code in matched])
                scaler = StandardScaler().fit(gallery)
                distance = np.linalg.norm(scaler.transform(query)[:, None, :] -
                                          scaler.transform(gallery)[None, :, :], axis=2)
                ranks = np.argsort(distance, axis=1)
                for index, code in enumerate(matched):
                    predicted = matched[int(ranks[index, 0])]
                    rank = int(np.flatnonzero(ranks[index] == index)[0]) + 1
                    rows.append({"subject": subject, "phase": phase, "target_position": position,
                                 "arm": arm, "movement": code, "predicted": predicted,
                                 "correct": int(predicted == code), "true_match_rank": rank,
                                 "gallery_classes": len(matched)})
        print(f"subject {subject}: {len(source_codes)} gallery; "
              f"P2={len(by_subject[str(subject)]['validation']['matched_movement_codes'])}, "
              f"P3={len(by_subject[str(subject)]['final']['matched_movement_codes'])}", flush=True)
    results = {"protocol": PROTOCOL, "archive_sha256": digest(ARCHIVE),
               "native_recordings": len(names), "rejected_recordings": rejections,
               "excluded_subject_positions": exclusions,
               "matched_sets": by_subject, "results": {}}
    for phase in ("validation", "final"):
        results["results"][phase] = {}
        for arm in PROTOCOL["arms"]:
            selected = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            individual = {str(subject): float(np.mean([row["correct"] for row in selected
                                                       if row["subject"] == subject]))
                          for subject in PROTOCOL["subjects"]
                          if any(row["subject"] == subject for row in selected)}
            by_movement = {code: float(np.mean([row["correct"] for row in selected
                                                if row["movement"] == code]))
                           for code in PROTOCOL["movement_codes"]
                           if any(row["movement"] == code for row in selected)}
            results["results"][phase][arm] = {
                "matched_recordings": len(selected),
                "included_subjects": len(individual),
                "mean_subject_top1_accuracy": float(np.mean(list(individual.values()))),
                "minimum_subject_top1_accuracy": min(individual.values()),
                "pooled_top1_accuracy": float(np.mean([row["correct"] for row in selected])),
                "mean_true_match_rank": float(np.mean([row["true_match_rank"] for row in selected])),
                "per_subject_top1_accuracy": individual,
                "per_movement_recall": by_movement}
            print(f"{phase} {arm}: file-level top1="
                  f"{results['results'][phase][arm]['pooled_top1_accuracy']:.4f}", flush=True)
    results["validation_selected_arm"] = max(PROTOCOL["arms"], key=lambda arm:
        results["results"]["validation"][arm]["mean_subject_top1_accuracy"])
    (ROOT / "ZENODO_REPLACEMENT_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "ZENODO_REPLACEMENT_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("acquire", "evaluate"))
    args = parser.parse_args()
    if args.phase == "acquire":
        acquire()
    else:
        evaluate()


if __name__ == "__main__":
    main()
