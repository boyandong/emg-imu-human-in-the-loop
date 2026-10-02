"""Native-input sensitivity of exact versus frozen personal Q95 denominators."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.datasets.epn612 import load_epn612_windows
from emgimu.feature_bank.calibration import (
    EPS, DocumentPersonalNormalizerV2, PersonalNormalizer,
)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "DOCUMENT_NORMALIZER_EPN_PROTOCOL.json"


def comparison(batch, labels, *, phase: str, user: int | str, shots: int | str, trials) -> dict:
    legacy = PersonalNormalizer(rest_label=0).fit(batch, labels)
    exact = DocumentPersonalNormalizerV2(rest_label=0).fit(batch, labels)
    np.testing.assert_array_equal(legacy.center_, exact.center_)
    expected_legacy = np.maximum(exact.scale_, EPS)
    np.testing.assert_allclose(legacy.scale_, expected_legacy, atol=0, rtol=0)
    legacy_denominator = legacy.scale_
    exact_denominator = exact.scale_ + EPS
    relative_difference = np.abs(exact_denominator - legacy_denominator) / legacy_denominator
    transformed_difference = np.max(np.abs(
        legacy.transform(batch).emg - exact.transform(batch).emg))
    identifiers = sorted(set(map(str, trials)))
    return {
        "phase": phase, "user": str(user), "shots_per_class": str(shots),
        "calibration_trials_count": len(identifiers),
        "calibration_trials_sha256": hashlib.sha256(
            json.dumps(identifiers, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "calibration_windows": batch.windows,
        "minimum_q95": float(np.min(exact.scale_)),
        "channels_q95_at_or_below_epsilon": int(np.sum(exact.scale_ <= EPS)),
        "maximum_relative_denominator_difference": float(np.max(relative_difference)),
        "maximum_calibration_signal_difference": float(transformed_difference),
    }


def audit() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    archive = Path(protocol["archive"])
    if archive.stat().st_size != protocol["archive_size_bytes"]:
        raise AssertionError("EPN archive size differs from verified manifest")
    source = load_epn612_windows(archive, users=range(1, 16))
    rows = []
    rows.append(comparison(source.batch, source.labels, phase="source_population", user="ALL",
                           shots="population", trials=source.trials))
    for user in range(1, 16):
        index = np.flatnonzero(source.users == user)
        data = source.take(index)
        rows.append(comparison(data.batch, data.labels, phase="source_user", user=user,
                               shots="all_source", trials=data.trials))
        print(f"Normalizer source {user}/15", flush=True)
    for phase, users in (("validation", protocol["target_validation_users"]),
                         ("descriptive_final", protocol["target_descriptive_final_users"])):
        target = load_epn612_windows(archive, users=users)
        for user in users:
            data = target.take(np.flatnonzero(target.users == user))
            for shots in protocol["target_shots_per_class"]:
                if shots == 0:
                    continue  # The same source-population state is already recorded above.
                rng = np.random.default_rng(protocol["selection_seed"] + user)
                chosen = []
                for label in range(6):
                    trials = np.unique(data.trials[data.labels == label])
                    chosen.extend(rng.permutation(trials)[:shots])
                cal = np.flatnonzero(np.isin(data.trials, chosen))
                rows.append(comparison(data.batch.take(cal), data.labels[cal],
                                       phase=phase, user=user, shots=shots,
                                       trials=data.trials[cal]))
            print(f"Normalizer {phase} user {user}", flush=True)
    path = ROOT / "DOCUMENT_NORMALIZER_EPN_CELLS.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "archive_sha256_from_verified_manifest": protocol["archive_sha256_from_verified_manifest"],
        "cells_sha256": sha256(path), "cells": len(rows),
        "source_cells": sum(row["phase"].startswith("source") for row in rows),
        "target_cells": sum(row["phase"] in ("validation", "descriptive_final") for row in rows),
        "minimum_q95": min(row["minimum_q95"] for row in rows),
        "channels_q95_at_or_below_epsilon": sum(row["channels_q95_at_or_below_epsilon"] for row in rows),
        "maximum_relative_denominator_difference": max(
            row["maximum_relative_denominator_difference"] for row in rows),
        "maximum_calibration_signal_difference": max(
            row["maximum_calibration_signal_difference"] for row in rows),
        "scope": protocol["boundary"],
    }
    (ROOT / "DOCUMENT_NORMALIZER_EPN_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "cells", "minimum_q95", "channels_q95_at_or_below_epsilon",
        "maximum_relative_denominator_difference")}), flush=True)
    return result


if __name__ == "__main__":
    audit()
