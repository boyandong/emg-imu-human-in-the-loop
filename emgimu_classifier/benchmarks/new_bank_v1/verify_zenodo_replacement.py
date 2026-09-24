"""Read-back verification and descriptive paired uncertainty for the frozen run."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "ZENODO_REPLACEMENT_PREDICTIONS.csv"
RESULTS = ROOT / "ZENODO_REPLACEMENT_RESULTS.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    with CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    arms = result["protocol"]["arms"]
    expected = sum(len(phase["matched_movement_codes"])
                   for subject in result["matched_sets"].values()
                   for phase in subject.values() if not phase.get("excluded")) * len(arms)
    assert len(rows) == expected
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        key = (row["subject"], row["phase"], row["arm"], row["movement"])
        assert key not in seen
        seen.add(key)
        assert row["arm"] in arms
        assert row["phase"] in ("validation", "final")
        assert row["target_position"] == ("P2" if row["phase"] == "validation" else "P3")
        assert row["movement"] in result["matched_sets"][row["subject"]][row["phase"]]["matched_movement_codes"]
        assert int(row["correct"]) == int(row["movement"] == row["predicted"])
        assert 1 <= int(row["true_match_rank"]) <= int(row["gallery_classes"])
        groups[row["phase"], row["arm"]].append(row)
    assert len(groups) == 2 * len(arms)
    for (phase, arm), part in groups.items():
        recorded = result["results"][phase][arm]
        assert len(part) == recorded["matched_recordings"]
        assert np.isclose(np.mean([int(row["correct"]) for row in part]),
                          recorded["pooled_top1_accuracy"], atol=1e-12)
        per_subject = {subject: np.mean([int(row["correct"]) for row in part
                                         if row["subject"] == subject])
                       for subject in {row["subject"] for row in part}}
        assert set(per_subject) == set(recorded["per_subject_top1_accuracy"])
        for subject, score in per_subject.items():
            assert np.isclose(score, recorded["per_subject_top1_accuracy"][subject], atol=1e-12)
        assert np.isclose(np.mean(list(per_subject.values())),
                          recorded["mean_subject_top1_accuracy"], atol=1e-12)
        assert np.isclose(min(per_subject.values()),
                          recorded["minimum_subject_top1_accuracy"], atol=1e-12)
        assert np.isclose(np.mean([int(row["true_match_rank"]) for row in part]),
                          recorded["mean_true_match_rank"], atol=1e-12)
    selected = result["validation_selected_arm"]
    assert selected == max(arms, key=lambda arm:
                           result["results"]["validation"][arm]["mean_subject_top1_accuracy"])
    base = {(row["subject"], row["movement"]): int(row["correct"])
            for row in groups["final", "F0"]}
    chosen = {(row["subject"], row["movement"]): int(row["correct"])
              for row in groups["final", selected]}
    assert base.keys() == chosen.keys()
    improved = sum(chosen[key] == 1 and base[key] == 0 for key in base)
    degraded = sum(chosen[key] == 0 and base[key] == 1 for key in base)
    subjects = sorted({subject for subject, _ in base}, key=int)
    differences = np.array([
        np.mean([chosen[key] - base[key] for key in base if key[0] == subject])
        for subject in subjects
    ])
    rng = np.random.default_rng(20260924)
    replicates = differences[rng.integers(0, len(subjects), size=(100_000, len(subjects)))].mean(axis=1)
    summary = {
        "prediction_csv_sha256": sha256(CSV),
        "result_json_sha256": sha256(RESULTS),
        "rows_verified": len(rows),
        "metric_groups_verified": len(groups),
        "selected_arm": selected,
        "final_paired_improved_recordings": improved,
        "final_paired_degraded_recordings": degraded,
        "final_paired_exact_two_sided_p": float(binomtest(min(improved, degraded),
                                                          improved + degraded, .5).pvalue),
        "final_mean_subject_delta": float(differences.mean()),
        "final_subject_bootstrap_percentile_95": [float(value) for value in
                                                  np.quantile(replicates, [.025, .975])],
        "bootstrap_subjects": len(subjects),
        "bootstrap_replicates": len(replicates),
        "uncertainty_scope": "Descriptive post-result paired analysis; subjects resampled, no model selection or live-accuracy inference",
    }
    (ROOT / "ZENODO_REPLACEMENT_VERIFICATION.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    verify()
