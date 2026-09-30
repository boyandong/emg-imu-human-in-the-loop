"""Clause-level acceptance of the versioned replacement, with explicit gaps."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "feature_bank"
DOCUMENTS = {
    "docx_pre.txt": "9b70fb41ebbb4f8183e19533facf80df81b1d7ce73bd3a976a2714347d3f11d8",
    "docx_goal.txt": "4da8b372c8f07936c1156935114849f85c0d83957c1ecacc6a0b6462bdf3b1f0",
}
# Status is a reviewed scientific judgement. Presence and hashes only bind its evidence.
CLAUSES = (
    ("PRE-01", "docx_pre.txt", 16, 99, "Repository and historical source review", "superseded_history", "feature_bank/results/historical_source_recovery_audit.json", "The unavailable historical algorithm is explicitly superseded by the requested versioned replacement; original ordering cannot be reconstructed."),
    ("PRE-02", "docx_pre.txt", 100, 167, "Candidate scoring before training", "superseded_history", "benchmarks/discovery/SCORE_REVIEW.md", "Candidate scores are transparent but retrospective, so original pretraining chronology is not proven."),
    ("PRE-03", "docx_pre.txt", 168, 649, "Mandatory dataset research", "verified_scoped", "benchmarks/discovery/DATASET_CANDIDATES.csv", "Mandatory native capabilities and confounds are researched; historical DS2 experiment identity is outside the replacement scope."),
    ("PRE-04", "docx_pre.txt", 650, 777, "Failure-to-benchmark mapping", "verified_scoped", "benchmarks/discovery/FAILURE_BENCHMARK_MATRIX.md", "Unsupported factors stay N/A."),
    ("PRE-05", "docx_pre.txt", 778, 904, "Tiered acquisition and download order", "superseded_history", "benchmarks/discovery/DATASET_MANIFEST.json", "Current archives and tiers are recorded; transient original acquisition order and safeguards cannot be proved retrospectively."),
    ("PRE-06", "docx_pre.txt", 905, 1049, "Paths, native sanity, labels and topology", "verified_scoped", "benchmarks/discovery/SANITY_AUDIT.json|benchmarks/discovery/GESTURE_ONTOLOGY.md|benchmarks/discovery/SENSOR_LAYOUTS.md", "Native labels/topology are preserved; unknown electrode geometry is not fabricated."),
    ("PRE-07", "docx_pre.txt", 1050, 1125, "Benchmark selection and discovery delivery", "verified_scoped", "benchmarks/discovery/BENCHMARK_SELECTION_REPORT.md|benchmarks/discovery/DISCOVERY_DELIVERY_AUDIT.json", "Selection is evidence-bounded; original historical timing is not claimed."),
    ("GOAL-01", "docx_goal.txt", 90, 132, "Dataset capability and N/A audit", "verified_scoped", "feature_bank/EXPERIMENT_CAPABILITIES.md", "Own-device multi-user/day and real hardware quality remain N/A."),
    ("GOAL-02", "docx_goal.txt", 133, 175, "Versioned baseline, split and state freeze", "verified_scoped", "benchmarks/new_bank_v2/V1_REPRODUCIBILITY_AUDIT.json", "Seven new-v1 packages bind protocols, 19,060 predictions and disjoint native trial identities; full numerical-environment identity is not established."),
    ("GOAL-03", "docx_goal.txt", 176, 446, "F0–F9 family implementation and native eligibility", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.csv|benchmarks/new_bank_v2/F2_WEARING_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F4D_MANUS_SESSION_REPORT.md|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_REPORT.md|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_REPORT.md|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json", "F0–F9 have a ten-family public-evidence scope matrix. Independent F0v2/four additions have seven-axis screens; five F2 candidates, including document-consistent uncentered F2a/F2c have matched wearing, MANUS session and GRAB unseen-user increments. All F2 additions hurt GRAB unseen-user F1/log loss, so no universal winner. F4d has a six-user predictive session correction control. F5c has a frozen full-bout UniBo increment: validation F1 gains do not survive final pooled F1. F8 Rest-noise uses isolated Song calibration blocks; F9 source-frozen Song and GRAB gates reject many correct frozen F0 trials. Calibrated body frame, own-device session and measured hardware quality remain unproved or unavailable."),
    ("GOAL-04", "docx_goal.txt", 447, 533, "0/1/2/5-shot personal calibration", "verified_scoped", "benchmarks/new_bank_v2/V1_PERSONAL_SCORE_CAL_REPORT.md|benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_REPORT.md", "Fixed score-space and feature-space methods are evaluated, including negative outcomes; this does not validate every possible personalization method."),
    ("GOAL-05", "docx_goal.txt", 534, 581, "Calibrated provider fusion", "verified_scoped", "feature_bank/results/calibration_burden_audit.json", "Selected source OOF-calibrated providers and reliability fusion have native controls; universal improvement is not claimed."),
    ("GOAL-06", "docx_goal.txt", 582, 632, "Stage 1 family and seven-axis screening", "verified_scoped", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_CELLS.csv|benchmarks/new_bank_v2/V1_CROSS_AXIS_SUBJECT_CELLS.csv", "Seven available axes use distinct public tasks; GRAB and ROAM subsets are correlated rather than independent replications."),
    ("GOAL-07", "docx_goal.txt", 633, 657, "Stage 2 conditional held-out increments", "verified_scoped", "feature_bank/delivery/conditional_incremental.csv|benchmarks/new_bank_v2/V1_CANONICAL_SCHEMA_AUDIT.json", "Matched F0v2-plus-family increments are delivered with no missing required new-v1 fields; not direct mutual information."),
    ("GOAL-08", "docx_goal.txt", 658, 684, "Stage 3 paired error complementarity", "verified_scoped", "feature_bank/delivery/error_complementarity.csv|feature_bank/STAGE3_4_CLAUSE_AUDIT.json", "Complete new-v1 pairwise error cells and bounded named-reference comparisons exist; no historical-family identity is asserted."),
    ("GOAL-09", "docx_goal.txt", 685, 714, "Stage 4 finite interaction tests", "verified_scoped", "benchmarks/new_bank_v2/RING_FREQ_INTERACTION_REPORT.md|feature_bank/STAGE3_4_CLAUSE_AUDIT.json", "A finite theory-motivated new-v1 interaction and older reference pairs were tested; final-only positives do not override validation."),
    ("GOAL-10", "docx_goal.txt", 715, 740, "Stage 5 full-bank shortlist and LOFO", "verified_scoped", "benchmarks/new_bank_v2/FULL_V1_LOFO_REPORT.md|benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|benchmarks/new_bank_v2/FULL_V1_LOFO_CELLS.csv", "The complete finite five-family new-v1 candidate and every leave-one-family-out arm were tested on seven public axes with 26,684 saved predictions and frozen F0v2 replay. Six of seven validation coordinates decline, so this candidate is not promoted; historical or own-device equivalence is not claimed."),
    ("GOAL-11", "docx_goal.txt", 741, 766, "Final family roles", "verified_scoped", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_REPORT.md", "Backbone, research specialists and screened-out generic additions are explicit; no family is falsely called a proven calibration amplifier."),
    ("GOAL-12", "docx_goal.txt", 767, 799, "Classification, calibration and subject metrics", "verified_scoped", "feature_bank/delivery/feature_family_results.csv|benchmarks/new_bank_v2/V1_CROSS_AXIS_SUBJECT_SUMMARY.csv", "Native held-out F1, loss, Brier, ECE, class and subject metrics are available where supported."),
    ("GOAL-13", "docx_goal.txt", 800, 836, "Mean and minimum seven-axis robustness vector", "verified_scoped", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json", "Unweighted descriptive mean and minimum are shown with all seven coordinates; no universal classifier accuracy is inferred."),
    ("GOAL-14", "docx_goal.txt", 837, 863, "Calibration burden and product time", "verified_scoped", "feature_bank/results/calibration_burden.csv|benchmarks/new_bank_v2/V1_PERSONAL_SCORE_CAL_REPORT.md", "Public 0/1/2/5-shot signal time is quantified; physical setup and live device wall time are deferred."),
    ("GOAL-15", "docx_goal.txt", 864, 893, "Trial split and source-only fit invariants", "verified_scoped", "benchmarks/new_bank_v2/V1_REPRODUCIBILITY_AUDIT.json|feature_bank/results/integrity_audit.json", "New-version saved trial disjointness, protocol hashes and selected leakage guards pass; no claim of exhaustive older-run audit."),
    ("GOAL-16", "docx_goal.txt", 894, 932, "Versioned staged execution", "verified_scoped", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_PROTOCOL.json|benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json", "Replacement experiments and decision rule are versioned; nonexistent historical execution chronology was superseded."),
    ("GOAL-17", "docx_goal.txt", 933, 1012, "Required scientific output files", "verified_scoped", "feature_bank/REPORT.md|feature_bank/delivery/PROVENANCE_AUDIT.json|benchmarks/new_bank_v2/V1_CANONICAL_SCHEMA_AUDIT.json", "All named files exist; 5,312 new-v1 family/conditional/error rows have complete required values while older records retain explicit N/A."),
    ("GOAL-18", "docx_goal.txt", 1013, 1051, "Questions A–H", "verified_scoped", "feature_bank/REPORT.md", "Answers include negative findings, subject effects, calibration signal time and evidence boundaries; device-level claims remain N/A."),
    ("GOAL-19", "docx_goal.txt", 1052, 1109, "Prior separation, tests and Git delivery", "verified_scoped", "feature_bank/REPORT.md|feature_bank/REQUIREMENT_AUDIT.json", "Historical priors are distinguished from new evidence; current tests and authorized branch delivery are separately checked at release time."),
    ("GOAL-20", "docx_goal.txt", 1110, 2300, "Detailed F0–F9 formulas and dimensions", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_SUBSECTION_COVERAGE_AUDIT.json|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.json|benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|tests/test_complete_sequence_contract.py|tests/test_f5c_complete_signature.py|tests/test_f4_spectral_known_signal.py|tests/test_body_frame.py|tests/test_f7_anchor_known_geometry.py|tests/test_f8_known_geometry.py|tests/test_session_shift_summary.py|tests/test_f9_known_signal.py|tests/test_f9_grab_gate_readback.py|tests/test_quality_mask_v1.py", "Fifty-six reviewed formula/source rows map all 32 formula-bearing appendix sections to exact source symbols, with three contextual headings separate. Exact F0/F2, F3/F4 known signals, F4b/F4c direct-Fourier summaries, DTW path and explicit contiguous cued-bout assembly, F5c complete-sequence signature plus native full-bout increment, F6 fifteen-output body-frame, F7 two-dimensional distance/similarity/margin, F8 geometry plus calibration-only Rest-noise, and F9 direct-Fourier/robust-reference plus Song and unseen-user GRAB negative gate checks exist. Source mapping and selected oracles do not prove every equation, native eligibility or safe live routing."),
    ("GOAL-21", "docx_goal.txt", 2301, 2475, "Personal-calibration formulas", "verified_scoped", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_REPORT.md", "Native fixed-rule evidence includes failure; some high-dimensional Mahalanobis options are unsupported by 1/2/5-shot counts."),
    ("GOAL-22", "docx_goal.txt", 2476, 2513, "Session calibration and immutable profile", "deferred_data_device", "feature_bank/results/family_specific_session_shift_audit.json|benchmarks/new_bank_v2/SESSION_UNLABELED_RESULTS.json|benchmarks/new_bank_v2/SESSION_UNLABELED_PREDICTIONS.csv|tests/test_session_unlabeled_delivery.py", "Public wearing/session controls and label-free native prediction parity exist, with disjoint evaluation trials and immutable source/session state. Own-device re-donning and multi-day validation need unavailable recordings or hardware."),
    ("GOAL-23", "docx_goal.txt", 2514, 2562, "Late fusion and Unknown decision", "verified_scoped", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/results/quality_unknown_replay.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json", "Fusion equations and explicit Unknown path are implemented. A public unseen-user F9 gate rejects many correct trials; a separate severe structural rule rejects no natural Song, GRAB or public re-wearing trials, catches synthetic constant channels, and catches no natural classification errors. Effective live hardware-quality gating is not established or promoted."),
    ("GOAL-24", "docx_goal.txt", 2563, 2605, "Family diagnostics", "verified_scoped", "feature_bank/delivery/conditional_incremental.csv|feature_bank/delivery/error_complementarity.csv", "Predictive incremental and paired-error diagnostics are available; not every physiological distance metric is identifiable in every dataset."),
    ("GOAL-25", "docx_goal.txt", 2606, 2640, "Channel/topology/fitting boundaries", "verified_scoped", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/EXPERIMENT_CAPABILITIES.md", "Three-, four- and eight-channel datasets remain separate; absent IMU/topology information is not fabricated."),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(pre: Path, goal: Path) -> dict:
    sources = {"docx_pre.txt": pre, "docx_goal.txt": goal}
    lines = {}
    for name, path in sources.items():
        if sha(path) != DOCUMENTS[name]:
            raise AssertionError(f"specification text changed: {name}")
        lines[name] = path.read_text(encoding="utf-8-sig").splitlines()
    rows = []
    for ident, doc, start, end, title, status, evidence, boundary in CLAUSES:
        if not (0 < start <= end <= len(lines[doc])):
            raise AssertionError(f"invalid specification locator: {ident}")
        paths = [ROOT / item for item in evidence.split("|")]
        if any(not path.is_file() for path in paths):
            raise AssertionError(f"missing acceptance evidence: {ident}")
        rows.append({"requirement_id": ident, "document": doc,
                     "document_sha256": DOCUMENTS[doc], "line_start": start,
                     "line_end": end, "requirement": title, "status": status,
                     "evidence": evidence,
                     "evidence_sha256_json": json.dumps({path.relative_to(ROOT).as_posix(): sha(path)
                                                         for path in paths}, sort_keys=True),
                     "boundary": boundary})
    if len(rows) != 32 or len({row["requirement_id"] for row in rows}) != len(rows):
        raise AssertionError("new-version acceptance clause inventory changed")
    csv_path = OUTPUT / "NEW_VERSION_ACCEPTANCE_AUDIT.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = dict(Counter(row["status"] for row in rows))
    open_items = [row["requirement_id"] for row in rows
                  if row["status"] in ("partial_active", "deferred_data_device")]
    audit = {"completion_proven": False, "clauses": len(rows),
             "status_counts": counts, "open_items": open_items,
             "csv_sha256": sha(csv_path), "documents": DOCUMENTS,
             "boundary": "A reviewed major-clause audit of the user-authorized versioned replacement. It does not override the lossless 2,586-line source queue or the 78-section audit; precise formula and native-device gaps remain explicitly open. Historical-code absence and original run chronology are superseded, not retroactively proved."}
    (OUTPUT / "NEW_VERSION_ACCEPTANCE_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clauses": len(rows), "status_counts": counts,
                      "open_items": open_items}), flush=True)
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pre", type=Path)
    parser.add_argument("goal", type=Path)
    arguments = parser.parse_args()
    build(arguments.pre, arguments.goal)
