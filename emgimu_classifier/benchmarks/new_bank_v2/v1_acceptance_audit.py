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
    ("GOAL-02", "docx_goal.txt", 133, 175, "Versioned baseline, split and state freeze", "verified_scoped", "benchmarks/new_bank_v2/V1_REPRODUCIBILITY_AUDIT.json|benchmarks/new_bank_v3/SPEC_F2C_GRAB_RESULTS.json|benchmarks/new_bank_v3/SPEC_F2C_WEARING_RESULTS.json|benchmarks/new_bank_v3/SPEC_F2C_TRANSFER_RESULTS.json", "Seven new-v1 packages bind protocols, 19,060 predictions and disjoint native trial identities. Latest V3 F2c runs additionally record numerical-library and thread-backend fingerprints; the older seven-package run-time identity is not established."),
    ("GOAL-03", "docx_goal.txt", 176, 446, "F0–F9 family implementation and native eligibility", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.csv|benchmarks/song_real8/F0_REST_NOISE_RESULTS.json|tests/test_f0_rest_noise_song_delivery.py|benchmarks/song_real8/F0_REST_MODEL_RESULTS.json|benchmarks/song_real8/F0_REST_MODEL_TRIAL_PREDICTIONS.csv|tests/test_f0_rest_model_song_delivery.py|benchmarks/new_bank_v2/F2_WEARING_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/PUBLIC_DEFAULT_EXTENSION_AUDIT.json|benchmarks/new_bank_v2/F4D_MANUS_SESSION_REPORT.md|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_REPORT.md|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_REPORT.md|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/new_bank_v2/F8_GRAB_DAY_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json", "F0–F9 have a ten-family public-evidence scope matrix. A source-frozen 250 Hz Song F0 replay confirms that Rest-only noise thresholds change native counts and a matched source-trained classifier comparison loses S03 F1/loss/Brier and S04 F1, so no default is promoted. Independent F0v2/four additions have seven-axis screens; historical V2 uncentered F2a/F2c alternatives have matched wearing, MANUS session and GRAB unseen-user increments. Centered V3 F2a/F2c/F3c follow the actual goal equations. Formula-correct F2c has separate frozen GRAB cross-day and wearing/MANUS/unseen-user matched-trial replays; the three-axis default guard fails on MANUS loss and GRAB F1/loss, while its cross-day Day2 gain reverses on descriptive Day3. All F2 additions hurt GRAB unseen-user F1/log loss, so no universal winner. A hash-bound V3 public default extension guard rejects all seven newer F2a/F2c/F3c/F5c/F7 local checks, retaining F0v2 as the public benchmark default. F4d has a six-user predictive session correction control. F5c has a frozen full-bout UniBo increment: validation F1 gains do not survive final pooled F1. F8 Rest-noise uses Song calibration blocks; a fixed GRAB cross-day routing replay changes no gesture decisions; F9 source-frozen Song and GRAB gates reject many correct frozen F0 trials; opt-in V3 observations also have a source-frozen 847-window Song native readout without gate promotion. Calibrated body frame, own-device session and measured hardware quality remain unproved or unavailable."),
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
    ("GOAL-20", "docx_goal.txt", 1110, 2300, "Detailed F0–F9 formulas and dimensions", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_SUBSECTION_COVERAGE_AUDIT.json|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.json|benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/PUBLIC_DEFAULT_EXTENSION_AUDIT.json|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|tests/test_document_personal_anchor_v2.py|tests/test_f7_document_anchor_native.py|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/new_bank_v2/F8_GRAB_DAY_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json|tests/test_complete_sequence_contract.py|tests/test_f5c_complete_signature.py|tests/test_f4_spectral_known_signal.py|tests/test_body_frame.py|tests/test_f7_anchor_known_geometry.py|tests/test_f8_known_geometry.py|tests/test_session_shift_summary.py|tests/test_f9_known_signal.py|tests/test_f9_grab_gate_readback.py|tests/test_quality_mask_v1.py|tests/test_document_quality_v3.py|benchmarks/song_real8/F0_REST_NOISE_RESULTS.json|tests/test_f0_rest_noise_song_delivery.py|benchmarks/song_real8/F0_REST_MODEL_RESULTS.json|benchmarks/song_real8/F0_REST_MODEL_TRIAL_PREDICTIONS.csv|tests/test_f0_rest_model_song_delivery.py", "Sixty-six reviewed formula/source rows map all 32 formula-bearing appendix sections to exact source symbols, with three contextual headings separate. Exact F0, including a source-frozen Rest-only 250 Hz eight-channel threshold feature replay plus a matched negative Song classifier increment, and centered V3 F2a/F2c/F3c analytical oracles, plus a checksum-frozen three-axis F2c matched-trial guard, F3/F4 known signals, F4b/F4c direct-Fourier summaries, DTW path and explicit contiguous cued-bout assembly, F5c complete-sequence signature plus native full-bout increment, F6 fifteen-output body-frame, F7 document-exact additive-epsilon distance/similarity/margin plus a frozen negative GRAB native replay, F8 geometry, calibration-only Rest-noise and a negative public cross-day routing control, F9 strict-threshold/additive-epsilon primitive oracles plus Song and unseen-user GRAB negative gate checks exist. Source mapping and selected oracles do not prove every equation, native eligibility or safe live routing."),
    ("GOAL-21", "docx_goal.txt", 2301, 2475, "Personal-calibration formulas", "verified_scoped", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_REPORT.md|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|benchmarks/new_bank_v3/DOCUMENT_RELIABILITY_V2_REPORT.md|benchmarks/new_bank_v3/DOCUMENT_RELIABILITY_GRAB_RESULTS.json|benchmarks/new_bank_v3/DOCUMENT_RELIABILITY_GRAB_PREDICTIONS.csv|tests/test_document_reliability_v2.py|tests/test_document_reliability_grab_delivery.py|tests/test_activation_profile.py|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json", "Native fixed-rule evidence includes failure; opt-in D/E exact formula has trial-balanced oracles and a Day2 source-CV-selected, Day3 descriptive native replay. Its 1/2/5-shot log loss and Brier worsen against the source population prior, so no default change is made. A known-waveform oracle checks the personal activation envelope and within-gesture spread. Some high-dimensional Mahalanobis options are unsupported by 1/2/5-shot counts."),
    ("GOAL-22", "docx_goal.txt", 2476, 2513, "Session calibration and immutable profile", "deferred_data_device", "feature_bank/results/family_specific_session_shift_audit.json|benchmarks/new_bank_v2/SESSION_UNLABELED_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json|benchmarks/new_bank_v2/SESSION_UNLABELED_PREDICTIONS.csv|tests/test_session_unlabeled_delivery.py", "Public wearing/session controls, label-free native prediction parity and a separate document-exact normalization route exist, with disjoint evaluation trials and immutable source/session state. Own-device re-donning and multi-day validation need unavailable recordings or hardware."),
    ("GOAL-23", "docx_goal.txt", 2514, 2562, "Late fusion and Unknown decision", "verified_scoped", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/results/quality_unknown_replay.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json", "Fusion equations and explicit Unknown path are implemented. A public unseen-user F9 gate rejects many correct trials; a separate severe structural rule rejects no natural Song, GRAB or public re-wearing trials, catches synthetic constant channels, and catches no natural classification errors. Effective live hardware-quality gating is not established or promoted."),
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
        if ident in ("GOAL-03", "GOAL-20"):
            evidence += "|tests/test_f1_scale_pattern_oracle.py"
            boundary += (" The versioned F1 RMS/global-RMS formula has an "
                         "independent eight-channel numeric oracle, with channel "
                         "and sample-rate fit contracts; historical X1-H identity "
                         "is not asserted.")
            evidence += "|tests/test_f3a_rlcs_direct_oracle.py"
            boundary += (" A direct Pearson F3a circular-lag oracle and a fitted "
                         "ring sample-rate guard verify the new-version geometry; "
                         "older serialized fits retain legacy compatibility.")
            evidence += "|tests/test_f4a_frequency_coord_oracle.py"
            boundary += (" A direct-complex-DFT 250 Hz F4a oracle and a fitted "
                         "sample-rate mismatch guard verify sub-Nyquist band coordinates; "
                         "older serialized fits predate the guard.")
            evidence += "|benchmarks/new_bank_v3/F5B_CUED_REPLAY_RESULTS.json|tests/test_cued_replay.py"
            boundary += (" A fixed-cue irregular-chunk replay exactly reconstructs all 3,391 "
                         "retained UniBo Day7/8 bouts and their envelope paths; it does not "
                         "detect biological onset or measure streaming accuracy.")
            evidence += ("|benchmarks/new_bank_v3/F3B_CES_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F3B_CES_GRAB_REPORT.md"
                         "|tests/test_document_ces_v3.py")
            boundary += (" Independent V3 F3b CES has a known-spectrum oracle and "
                         "frozen GRAB unseen-user matched-trial replay; both validation "
                         "and descriptive final F1/log loss worsen, so it remains opt-in.")
            evidence += ("|benchmarks/new_bank_v3/F5_TEMPORAL_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F5_TEMPORAL_GRAB_REPORT.md"
                         "|tests/test_document_temporal_v3.py")
            boundary += (" A separate goal-exact V3 F5 window feature passes a "
                         "known-waveform oracle and frozen GRAB unseen-user replay; "
                         "F1/loss worsen on validation/final, so it remains opt-in.")
            evidence += ("|benchmarks/new_bank_v3/F7_AFFINE_EPN/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_EPN_REPORT.md"
                         "|benchmarks/new_bank_v3/F7_AFFINE_CORE_MATCHED/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_CORE_MATCHED_REPORT.md"
                         "|tests/test_affine_spd_anchor.py"
                         "|tests/test_f7_affine_epn_delivery.py"
                         "|tests/test_f7_affine_core_matched_delivery.py"
                         "|feature_bank/F6_PUBLIC_ELIGIBILITY.md")
            boundary += (" The exact affine-invariant F7 SPD distance has independent "
                         "geometry/trial-mass oracles, an EPN612 1/2/5-shot "
                         "standalone screen and positive exact-trial fixed-Core "
                         "increments beyond uniform softening; inspected subjects "
                         "prevent prospective default promotion. "
                         "Strict public F6 calibration eligibility remains unproved.")
            evidence += ("|benchmarks/new_bank_v3/EPN107_F6_ELIGIBILITY.json"
                         "|benchmarks/new_bank_v3/epn107_f6_eligibility_probe.py"
                         "|tests/test_epn107_f6_eligibility.py")
            boundary += (" A complete 107-MAT EPN107 archive-schema census finds "
                         "raw accel/gyro absent for 69 gForce users and no explicit "
                         "anatomical forward-axis field for any device; this source "
                         "cannot support strict F6 recognition claims.")
            evidence += ("|benchmarks/new_bank_v3/F7_AFFINE_FRESH_PROTOCOL.md"
                         "|benchmarks/new_bank_v3/F7_AFFINE_FRESH/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_FRESH_REPORT.md"
                         "|tests/test_f7_affine_fresh_protocol.py"
                         "|tests/test_f7_affine_fresh_delivery.py")
            boundary += (" A precommitted, first-read EPN users22-31 test passes all five "
                         "five-shot primary guards for fixed Core+F7 on 1,200 disjoint trials; "
                         "7/10 users gain log loss. This is public-cohort evidence only, "
                         "and does not validate the user's montage or live deployment.")
        if ident in ("GOAL-03", "GOAL-20"):
            evidence += ("|src/emgimu/feature_bank/document_session_v3.py"
                         "|tests/test_document_session_v3.py"
                         "|tests/test_f8_document_manus_delivery.py"
                         "|benchmarks/new_bank_v3/F8_DOCUMENT_MANUS_RESULTS.json")
            boundary += (" Document F8 V3 now concatenates the four required "
                         "descriptor blocks, enforces same-user/new-session identity, "
                         "uses centered SPD geometry and separates channel-quality "
                         "scores from variance. A frozen-split MANUS replay verifies "
                         "24 six-class 180-dimensional descriptors; it is not a "
                         "predictive routing gain.")
            boundary=boundary.replace('Sixty-six reviewed formula/source rows',
                'Seventy-six reviewed formula/source rows')
            evidence += ("|src/emgimu/feature_bank/document_scale_v3.py"
                         "|tests/test_document_scale_v3.py"
                         "|tests/test_f1_scale_grab_delivery.py"
                         "|benchmarks/new_bank_v3/F1_SCALE_GRAB_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F1_SCALE_GRAB_RESULTS.json")
            boundary += (" Separate F1 V3 follows the additive-epsilon "
                         "RMS/global-RMS formula, with ordinary/zero/near-zero "
                         "independent numeric oracles. Log scale stays in a "
                         "separate context API and Document F8 uses these exact "
                         "coordinates. A frozen GRAB unseen-user matched "
                         "increment loses validation/final F1/loss; no default change.")
            evidence += ("|src/emgimu/feature_bank/document_path_v3.py"
                         "|tests/test_document_path_v3.py"
                         "|tests/test_f5_path_unibo_delivery.py"
                         "|benchmarks/new_bank_v3/F5_PATH_UNIBO_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F5_PATH_UNIBO_RESULTS.json")
            boundary += (" Document F5b/c V3 follows additive-epsilon full-envelope "
                         "normalization with independent exhaustive-DTW and near-zero "
                         "polygon oracles, frozen warp policy and explicit calibration "
                         "trial exclusion. Frozen UniBo candidates and complete held-out "
                         "Days6-8 coordinates are replayed; no live or calibrated "
                         "fusion benefit is claimed.")
            evidence += ("|src/emgimu/feature_bank/document_spectral_v3.py"
                         "|tests/test_document_spectral_v3.py"
                         "|tests/test_f4_spectral_grab_delivery.py"
                         "|benchmarks/new_bank_v3/F4_SPECTRAL_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F4_SPECTRAL_GRAB_PROTOCOL.json")
            boundary += (" Separate F4 V3 follows additive-epsilon frequency "
                         "orientation/centroid, raw entropy and log(power+epsilon) "
                         "orthonormal DCT-II with source-frozen B/K. Independent "
                         "normal/zero/near-zero Fourier/DCT oracles pass. A "
                         "fixed GRAB unseen-user matched increment gains validation "
                         "F1/loss but loses descriptive final F1/loss, so remains opt-in.")
            evidence += "|tests/test_session_shift_summary.py|src/emgimu/feature_bank/session_shift_summary.py"
            evidence = "|".join(dict.fromkeys(evidence.split("|")))
            boundary += (" The F8 family-summary API retains immutable source "
                         "trial identities and rejects calibration overlap, "
                         "invalid trial contracts and missing provenance. "
                         "Independent leakage and unequal-window trial-mass "
                         "tests preserve the source state.")
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
