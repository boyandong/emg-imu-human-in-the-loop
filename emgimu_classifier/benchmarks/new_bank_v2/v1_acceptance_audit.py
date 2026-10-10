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
    ("GOAL-03", "docx_goal.txt", 176, 446, "F0閳ユ強9 family implementation and native eligibility", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.csv|benchmarks/song_real8/F0_REST_NOISE_RESULTS.json|tests/test_f0_rest_noise_song_delivery.py|benchmarks/song_real8/F0_REST_MODEL_RESULTS.json|benchmarks/song_real8/F0_REST_MODEL_TRIAL_PREDICTIONS.csv|tests/test_f0_rest_model_song_delivery.py|benchmarks/new_bank_v2/F2_WEARING_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_REPORT.md|benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/PUBLIC_DEFAULT_EXTENSION_AUDIT.json|benchmarks/new_bank_v2/F4D_MANUS_SESSION_REPORT.md|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_REPORT.md|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_REPORT.md|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/new_bank_v2/F8_GRAB_DAY_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json", "F0閳ユ強9 have a ten-family public-evidence scope matrix. A source-frozen 250 Hz Song F0 replay confirms that Rest-only noise thresholds change native counts and a matched source-trained classifier comparison loses S03 F1/loss/Brier and S04 F1, so no default is promoted. Independent F0v2/four additions have seven-axis screens; historical V2 uncentered F2a/F2c alternatives have matched wearing, MANUS session and GRAB unseen-user increments. Centered V3 F2a/F2c/F3c follow the actual goal equations. Formula-correct F2c has separate frozen GRAB cross-day and wearing/MANUS/unseen-user matched-trial replays; the three-axis default guard fails on MANUS loss and GRAB F1/loss, while its cross-day Day2 gain reverses on descriptive Day3. All F2 additions hurt GRAB unseen-user F1/log loss, so no universal winner. A hash-bound V3 public default extension guard rejects all seven newer F2a/F2c/F3c/F5c/F7 local checks, retaining F0v2 as the public benchmark default. F4d has a six-user predictive session correction control. F5c has a frozen full-bout UniBo increment: validation F1 gains do not survive final pooled F1. F8 Rest-noise uses Song calibration blocks; a fixed GRAB cross-day routing replay changes no gesture decisions; F9 source-frozen Song and GRAB gates reject many correct frozen F0 trials; opt-in V3 observations also have a source-frozen 847-window Song native readout without gate promotion. Calibrated body frame, own-device session and measured hardware quality remain unproved or unavailable."),
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
    ("GOAL-18", "docx_goal.txt", 1013, 1051, "Questions A閳ユ弻", "verified_scoped", "feature_bank/REPORT.md", "Answers include negative findings, subject effects, calibration signal time and evidence boundaries; device-level claims remain N/A."),
    ("GOAL-19", "docx_goal.txt", 1052, 1109, "Prior separation, tests and Git delivery", "verified_scoped", "feature_bank/REPORT.md|feature_bank/REQUIREMENT_AUDIT.json", "Historical priors are distinguished from new evidence; current tests and authorized branch delivery are separately checked at release time."),
    ("GOAL-20", "docx_goal.txt", 1110, 2300, "Detailed F0閳ユ強9 formulas and dimensions", "partial_active", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.csv|feature_bank/FORMULA_SUBSECTION_COVERAGE_AUDIT.json|feature_bank/FORMULA_FAMILY_SCOPE_AUDIT.json|benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/PUBLIC_DEFAULT_EXTENSION_AUDIT.json|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|tests/test_document_personal_anchor_v2.py|tests/test_f7_document_anchor_native.py|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/new_bank_v2/F8_GRAB_DAY_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json|tests/test_complete_sequence_contract.py|tests/test_f5c_complete_signature.py|tests/test_f4_spectral_known_signal.py|tests/test_body_frame.py|tests/test_f7_anchor_known_geometry.py|tests/test_f8_known_geometry.py|tests/test_session_shift_summary.py|tests/test_f9_known_signal.py|tests/test_f9_grab_gate_readback.py|tests/test_quality_mask_v1.py|tests/test_document_quality_v3.py|benchmarks/song_real8/F0_REST_NOISE_RESULTS.json|tests/test_f0_rest_noise_song_delivery.py|benchmarks/song_real8/F0_REST_MODEL_RESULTS.json|benchmarks/song_real8/F0_REST_MODEL_TRIAL_PREDICTIONS.csv|tests/test_f0_rest_model_song_delivery.py", "Sixty-six reviewed formula/source rows map all 32 formula-bearing appendix sections to exact source symbols, with three contextual headings separate. Exact F0, including a source-frozen Rest-only 250 Hz eight-channel threshold feature replay plus a matched negative Song classifier increment, and centered V3 F2a/F2c/F3c analytical oracles, plus a checksum-frozen three-axis F2c matched-trial guard, F3/F4 known signals, F4b/F4c direct-Fourier summaries, DTW path and explicit contiguous cued-bout assembly, F5c complete-sequence signature plus native full-bout increment, F6 fifteen-output body-frame, F7 document-exact additive-epsilon distance/similarity/margin plus a frozen negative GRAB native replay, F8 geometry, calibration-only Rest-noise and a negative public cross-day routing control, F9 strict-threshold/additive-epsilon primitive oracles plus Song and unseen-user GRAB negative gate checks exist. Source mapping and selected oracles do not prove every equation, native eligibility or safe live routing."),
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
        if ident in ('PRE-03', 'PRE-05', 'PRE-07'):
            evidence += '|benchmarks/discovery/CURRENT_DISCOVERY_STATE.json|tests/test_current_discovery_state.py'
            boundary += (' Current discovery entry binds the publisher-v9 subjective force join '
                         'and independent DS2 study; the earlier no-force-label conclusion is '
                         'superseded. Legacy experiment recovery is not a prerequisite. '
                         'Recorded licensing and secondary provenance limits remain explicit.')
        if ident in ("GOAL-03", "GOAL-18", "GOAL-20"):
            evidence += ("|benchmarks/new_bank_v3/ROAM_CAUSAL_WINDOW_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/ROAM_CAUSAL_WINDOW_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv"
                         "|src/emgimu/feature_bank/causal_window_recognition_v1.py"
                         "|tests/test_causal_window_recognition_v1.py"
                         "|tests/test_roam_causal_window_v1_delivery.py")
            boundary += (" A precommitted new source-only ROAM48-coordinate window control "
                         "replays40 full recordings/393776 nominal samples with no future "
                         "context or target-label segmentation. First39 samples per file "
                         "remain unknown. Equal-recording F1 .7868/.8105 coexists with "
                         "23/160 and43/160 successful transition-hold events and1795 hold "
                         "switches; strong average classification does not imply stable "
                         "continuous output. Previously inspected cohorts are descriptive, "
                         "and this versioned tolerance metric is not ReactEMG reproduction "
                         "or hardware/own-device validation.")
        if ident in ("GOAL-03", "GOAL-18", "GOAL-20"):
            evidence += ("|benchmarks/new_bank_v3/ROAM_DEBOUNCE_CONTROL_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/ROAM_DEBOUNCE_CONTROL_V1_PAIRED.csv"
                         "|src/emgimu/feature_bank/causal_label_debounce_v1.py"
                         "|tests/test_causal_label_debounce_v1.py"
                         "|tests/test_roam_debounce_control_v1_delivery.py")
            boundary += (" A precommitted fixed two-emission confirmation replay reduces "
                         "hold switches1007 to561 and788 to436, but paired F1 falls "
                         ".7867 to.7840 and.8105 to.8061. Hold success26/160 and57/160 "
                         "remains limited. Extra unknown startup samples are retained, "
                         "full-record unknown-as-wrong accuracy is reported, and label-only "
                         "probability metrics stay N/A. No default promotion or device proof.")
        if ident in ("GOAL-12", "GOAL-14"):
            evidence += ("|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.csv"
                         "|benchmarks/new_bank_v3/epn_holdout_user_robustness_v2.py"
                         "|tests/test_epn_holdout_user_robustness_v2.py")
            boundary += (" Frozen EPN holdout equal-user means, sample SD and worst-user "
                         "outcomes are now distinct from pooled F1. Mahalanobis improves "
                         "worst-user F1 versus Euclidean at both budgets; increasing its "
                         "budget10 to20 harms3/10 users and lowers its worst-user F1 "
                         "from .5315 to .5026 despite mean gain. These two personally "
                         "calibrated methods do not establish improvement over an "
                         "uncalibrated baseline or seven-axis R_min.")
        if ident == "PRE-06":
            evidence += ("|benchmarks/discovery/GRAB_PRIMARY_TOPOLOGY_V1.json"
                         "|benchmarks/discovery/scripts/grab_primary_topology_v1.py"
                         "|tests/test_grab_primary_topology_v1.py")
            boundary += (" Official GRAB JATS acquisition text now establishes28 stored "
                         "monopolar electrodes in two8-electrode forearm and two6-electrode "
                         "wrist rings with2cm axial separation,2048Hz and header-based mV "
                         "scaling. Bipolar pairs are separately constructed; our selected "
                         "eight single-ring columns do not authenticate an eight-bipolar "
                         "device match. Direction/sign and figure geometry remain unverified.")
        if ident == "PRE-06":
            evidence += ("|benchmarks/discovery/GRAB_CHANNEL_CENSUS_V1.json"
                         "|benchmarks/discovery/scripts/grab_channel_census_v1.py"
                         "|tests/test_grab_channel_census_v1.py")
            boundary += (" All672 existing GRAB subset headers and signal checksums now "
                         "bind F1-F8 selected columns, 2048Hz and 10240samples. Independent "
                         "signed-byte/gain/baseline decoding checks21504 values against the "
                         "actual benchmark loader. Direction, our wiring and six-axis IMU "
                         "remain unverified; no differential-pair conversion inferred.")
        if ident == "GOAL-14":
            evidence += ("|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json"
                         "|benchmarks/new_bank_v3/epn_holdout_burden_v2.py"
                         "|feature_bank/delivery/new_bank_v3/calibration_burden.csv"
                         "|tests/test_mahalanobis_epn_holdout_v2.py"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_COST_CURVE.json"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_COST_CURVE.svg"
                         "|benchmarks/new_bank_v3/epn_holdout_cost_curve_v2.py"
                         "|tests/test_epn_holdout_cost_curve_v2.py")
            boundary += (" New holdout 10/20-shot-per-class budgets use 60/120 trials "
                         "and 48/96 seconds extracted signal per user; full stored recording "
                         "durations are separate. Physical setup/session times stay N/A, "
                         "so few-second calibration is not proven. A frozen performance-versus-exposure "
                         "curve reports pooled and individual-user outcomes at supported 10/20 budgets. "
                         "All six labelled gestures are required; repeated-session, force and posture "
                         "transfer requirements are untested rather than asserted optional.")
        if ident == "GOAL-18":
            evidence += ("|feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json"
                         "|benchmarks/new_bank_v3/current_scientific_conclusions_v1.py"
                         "|tests/test_current_scientific_conclusions_v1.py"
                         "|benchmarks/new_bank_v3/render_current_scientific_report_v1.py"
                         "|tests/test_current_scientific_report_v1.py")
            boundary += (" An independently versioned current A-H synthesis now binds "
                         "the newer F7 confirmation, EPN user variation/calibration costs, "
                         "MANUS routing and full-record ROAM stability evidence. Numerical "
                         "deltas, sample versus trial denominators and original guard-sign "
                         "conventions are explicit. Improved calibrated-method comparisons "
                         "do not prove improvement versus no anchor or only-after-calibration "
                         "benefit, and single-axis minima are not full seven-axis R_min.")
        if ident in ("GOAL-03", "GOAL-12", "GOAL-18", "GOAL-20"):
            evidence += ("|benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_PREDICTIONS.csv"
                         "|tests/test_emg_f0_f7_bank_v1.py"
                         "|tests/test_emg_f0_f7_bank_v1_delivery.py"
                         "|benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json"
                         "|benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.csv"
                         "|tests/test_emg_bank_class_diagnostics_v1.py"
                         "|benchmarks/new_bank_v3/EMG_WINDOW_BANK_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/EMG_WINDOW_BANK_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/EMG_WINDOW_BANK_V1_PREDICTIONS.csv"
                         "|tests/test_emg_window_bank_v1_delivery.py")
            boundary += (" A precommitted EMG-only F0/F7 two-provider bank tests1200 "
                         "fixed held-out trials of EPN42-51 with nested0/1/2/5 budgets "
                         "and both provider removals. Five-shot F1 .4633 to.4836 and "
                         "loss improves by.06383 (.09246 beyond uniform), but only6/10 "
                         "users win loss, failing the7/10 primary guard. No IMU features "
                         "or default promotion. A separate precommitted EPN52-61 "
                         "six-declared-group265-coordinate EMG window bank evaluates "
                         "1500 trials and13 source-fitted compositions with all group "
                         "removals. All primary guards fail, with0/10 user loss wins; "
                         "F1 .4361 to.3902. Not a replacement for document-wide "
                         "full-bank/LOFO, strict F6 or actual device efficacy.")
        if ident in ("GOAL-03", "GOAL-05", "GOAL-20", "GOAL-23"):
            evidence += ("|feature_bank/AVAILABLE_BANK_FUSION_ACCEPTANCE_V1.json"
                         "|src/emgimu/feature_bank/available_bank_fusion_v1.py"
                         "|tests/test_available_bank_fusion_v1.py"
                         "|tests/test_available_bank_fusion_v1_delivery.py")
            boundary += (" An opt-in arbitrary-provider interface now skips missing "
                         "calibration branches, renormalizes source population before "
                         "document-exact reliability shrinkage and renormalizes frozen "
                         "weights if a provider is absent at inference. Explicit per-provider "
                         "trial/class axes and disjoint source/calibration/evaluation IDs "
                         "are checked.24 imported native MANUS fusion blocks,96 provider "
                         "omissions and96 rejected invalid native calls are replayed. "
                         "This is software behavior, not sensor-fault efficacy or "
                         "document-wide native integration; no default promotion.")
        if ident in ("GOAL-03", "GOAL-05", "GOAL-10", "GOAL-12", "GOAL-18", "GOAL-20"):
            evidence += ("|benchmarks/new_bank_v3/EMG_CALIBRATED_FUSION_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/EMG_CALIBRATED_FUSION_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/EMG_CALIBRATED_FUSION_V1_PREDICTIONS.csv"
                         "|tests/test_emg_calibrated_fusion_v1_delivery.py")
            boundary += (" A precommitted EPN62-71 six-provider EMG experiment "
                         "refits representations/scalers/classifiers in3 source-user OOF "
                         "folds, fits source-only probability temperatures and computes "
                         "document-exact personal reliability from nested0/1/2/5-shot "
                         "calibration, on1200 identical held-out trials. All6 frozen-weight "
                         "provider removals are retained. Five-shot F1 .4969 to.5178 "
                         "but loss worsens by.11041 versus F0 and.00370 versus uniform; "
                         "0/10 user loss wins. Only weights adapt; no personal prototype "
                         "or normalization adaptation, complete F0-F9 or hardware claim.")
        if ident in ("GOAL-12", "GOAL-14", "GOAL-18"):
            evidence += ("|benchmarks/new_bank_v3/EMG_CALIBRATION_BURDEN_V1.json"
                         "|benchmarks/new_bank_v3/EMG_CALIBRATION_BURDEN_V1.csv"
                         "|benchmarks/new_bank_v3/EMG_CALIBRATION_COST_CURVE_V1.json"
                         "|benchmarks/new_bank_v3/EMG_CALIBRATION_COST_CURVE_V1.png"
                         "|tests/test_emg_calibration_burden_v1.py"
                         "|tests/test_emg_calibration_cost_curve_v1.py")
            boundary += (" Method-specific native cost accounting now binds690 "
                         "rows and600 reserved trial durations for EPN42-51 F0/F7 "
                         "and62-71 calibrated fusion.0/1/2/5-shot costs distinguish "
                         "used trials/windows from reserved and complete stored EMG "
                         "durations. Source-only controls use0 target calibration trials. "
                         "Five-shot uses30 trials and24 seconds extracted signal, "
                         "but147.9-149.72 seconds complete stored recordings. "
                         "Product setup/rest/wall time and per-session/controlled "
                         "force/posture requirements remain unmeasured or N/A. "
                         "Separate-cohort exported plots do not imply monotone benefit "
                         "or few-second physical onboarding.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-05", "GOAL-19", "GOAL-20", "GOAL-23"):
            evidence += ("|feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json"
                         "|feature_bank/models/epn_emg_calibrated_bank_v1.pkl"
                         "|src/emgimu/feature_bank/frozen_emg_provider_bank_v1.py"
                         "|tests/test_frozen_emg_provider_bank_v1.py"
                         "|src/emgimu/feature_bank/frozen_emg_bank_cli_v1.py"
                         "|tests/test_frozen_emg_bank_cli_v1.py")
            boundary += (" A source-fitted portable six-provider checkpoint now "
                         "includes complete transforms/scalers/models/temperatures. "
                         "Pure EMG8-channel200Hz40-sample cued-trial inference needs "
                         "native trial IDs and chronological window offsets but no "
                         "evaluation labels or source archive.10 provider,40 fusion "
                         "and240 omission cases reproduce frozen native probabilities "
                         "below1e-12; user calibration leaves source state immutable. "
                         "This does not validate autonomous250Hz device inference, "
                         "the full document bank or a default promotion. An offline "
                         "CLI checks checkpoint SHA-256, accepts explicit NPZ windows, "
                         "rejects prediction labels, preserves existing output files "
                         "and separately saves/loads same-user calibration profiles; "
                         "a separate-process check needs no source archive.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-20", "GOAL-23"):
            evidence += ("|feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json"
                         "|feature_bank/models/song_f0_250hz_v1.pkl"
                         "|src/emgimu/feature_bank/song_f0_runtime_v1.py"
                         "|tests/test_song_f0_runtime_v1.py")
            boundary += (" Both existing Song eight-channel250Hz F0 source models "
                         "are independently packaged. Original source state fingerprints "
                         "matched before packaging; all568 paired native trial predictions "
                         "replay below1e-12. Stateful three-stage causal filtering agrees "
                         "with independent one-pass and arbitrary-chunk references. "
                         "Explicit cued-trial windows and probability means are distinct "
                         "from EPN feature means; no evaluation labels or IMU are needed. "
                         "Single-user/day readiness and prior-inspection limits persist; "
                         "no autonomous live or default accuracy claim.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-20", "GOAL-23"):
            evidence += ("|feature_bank/SONG_GUI_V2_ACCEPTANCE.json"
                         "|tests/test_song_gui_v2_delivery.py")
            boundary += (" Both fixed Song250Hz models are integrated into the "
                         "collection page as checksum-verified JSON bundles. "
                         "All126800 native stream emissions and confirmation labels "
                         "match the frozen reference; offscreen Qt verifies model "
                         "loading, pause, packet loss, index-gap resets and disconnect. "
                         "The fixed200ms/40ms/two-confirmation policy cannot be changed "
                         "by the confidence control. Software integration does not "
                         "prove current-device accuracy or physical timing.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-08", "GOAL-09", "GOAL-20", "GOAL-22", "GOAL-23"):
            evidence += ("|feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json"
                         "|benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_PROTOCOL.json"
                         "|benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_RESULTS.json"
                         "|tests/test_personal_session_workflow_v1.py"
                         "|tests/test_personal_session_cli_v1.py"
                         "|tests/test_song_personal_session_v1_delivery.py")
            boundary += (" New Song offline personal/session lifecycle binds source, "
                         "user, recording and preprocessing identities and verifies "
                         "persistent profiles. Six source-temperature-calibrated window "
                         "providers, same124 S04 trials,44 cells and all24 provider "
                         "removals are retained. Long-term20 trials plus current0/4/8/20 "
                         "are separate budgets. Current calibration changes reliability "
                         "weights; normalization, F7 anchors, F8 and F9 remain separate "
                         "context outputs. Session effects are mixed, and more shots "
                         "can harm loss. The offline result alone does not prove complete document "
                         "F0-F9 integration, native anatomical F6, quality rejection "
                         "or cross-day/re-donning/device efficacy.")
            evidence += ("|feature_bank/SONG_PERSONAL_GUI_V1_ACCEPTANCE.json"
                         "|benchmarks/song_real8/SONG_PERSONAL_GUI_V1_PROTOCOL.json"
                         "|tests/test_personal_session_stream_v1.py"
                         "|tests/test_personal_session_gui_v1_delivery.py")
            boundary += (" A separate opt-in collection-page lifecycle now supports guided "
                         "personal enrollment and new-session calibration, persistent profiles "
                         "with separate calibration-window/label/capture companions, source-population "
                         "or calibrated live inference, unlabeled-window replay and explicit identity "
                         "checks. Qt plus an actual classifier subprocess verifies the flow; both "
                         "population and five-shot session streams reproduce all29,790 S04 windows "
                         "within1e-12 without source/profile mutation. This is numerical equivalence "
                         "on previously inspected data including calibration intervals, not a new "
                         "efficacy experiment. Device disconnection or discontinuity cancels pending "
                         "calibration and clears displayed recognition. Full F0-F9 fusion, quality "
                         "rejection and hardware efficacy remain unproved.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-08", "GOAL-09", "GOAL-20", "GOAL-21", "GOAL-22", "GOAL-23"):
            evidence += ("|feature_bank/SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json"
                         "|benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_PROTOCOL.json"
                         "|benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_RESULTS.json"
                         "|tests/test_matched_normalized_bank_v1.py"
                         "|tests/test_matched_normalization_cli_v1.py"
                         "|tests/test_matched_normalization_delivery_v1.py")
            boundary += (" A separate matched source-trained normalization candidate now applies "
                         "Rest median and active Q95+epsilon before feature extraction in both "
                         "training and inference. Both raw/normalized branches fit the same245 "
                         "source trials, reserving40 normalization trials. Same124 S04 evaluation "
                         "trials,80 cells/9920 predictions, all48 provider removals and calibration "
                         "budgets are retained. The fixed five-shot normalized candidate worsens "
                         "loss/Brier/F1 and fist recall; all four primary guards fail. Separate "
                         "normalized source/profile identities, label-free CLI, source-only "
                         "temperatures/policy and independent no-fit normalization oracles verify "
                         "execution, not a benefit. Population/F0 normalization still consumes "
                         "long-term20 plus current0/4/8/20 trials. No GUI/default promotion, "
                         "raw-ADC quality, full F0-F9 or hardware efficacy is proved.")
        if ident in ("GOAL-02", "GOAL-03", "GOAL-18", "GOAL-20", "GOAL-23"):
            evidence += ("|benchmarks/song_real8/SONG_F0_STREAM_V1_PROTOCOL.json"
                         "|benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json"
                         "|benchmarks/song_real8/SONG_F0_STREAM_V1_EMISSIONS.npz"
                         "|src/emgimu/feature_bank/song_f0_stream_v1.py"
                         "|tests/test_song_f0_stream_v1.py"
                         "|tests/test_song_f0_stream_v1_delivery.py"
                         "|tests/test_song_f0_stream_native_annotations_v1.py")
            boundary += (" The separately frozen full-recording Song250Hz stream "
                         "carries causal filter/window/confirmation memory and uses "
                         "no cues or labels at inference. All63400 emission positions "
                         "per arm match independent one-pass probabilities below1e-12. "
                         "Only4521 fully contained formal-stable windows across284 "
                         "trials are scored with equal trial mass; unknown intervals "
                         "remain unscored. Two-confirmation reduces within-stable "
                         "switches and increases whole-stable holds but lowers F1 in "
                         "all4 cells. Native HDF5 exclusions are independently checked. "
                         "No physiological onset, physical latency or default efficacy "
                         "claim follows from this retrospective one-user/day replay.")
        if ident == "GOAL-02":
            evidence += ("|feature_bank/OFFICIAL_UNIBO_ADAPTER_ACCEPTANCE_V1.json"
                         "|feature_bank/CLASSIFIER_REGRESSION_20261008.json"
                         "|benchmarks/official_unibo_adapter_acceptance_v1.py")
            boundary += (" Full-suite code/test fingerprints accompany517 passed,1 skipped "
                         "and21 subtests at a named revision. The optional original-MAT "
                         "conversion check then passes separately on an archive-bound "
                         "unchanged source sample; neither result is a full scientific "
                         "completion or actual eight-channel sensor-compatibility claim.")
        if ident == "GOAL-17":
            evidence += ("|feature_bank/delivery/INDEX.json"
                         "|feature_bank/delivery/new_bank_v3/MANIFEST.json"
                         "|benchmarks/new_bank_v3/export_current_delivery.py"
                         "|tests/test_current_v3_delivery.py")
            boundary += (" Current V3 adds 8,784 hash-bound table rows with an additive "
                         "version index; paired alternatives are explicitly distinguished "
                         "from added-feature increments, DTW probability metrics are N/A, "
                         "and 130 insufficient independent-trial budgets remain ineligible. "
                         "Every paired comparison carries its calibration budget; loss "
                         "and Brier deltas use base minus alternative, while F1 uses "
                         "alternative minus base, so positive consistently means improvement. "
                         "Paired-error probabilities now include their evaluation-trial "
                         "denominators; per-class precision/recall/F1 carry explicit support "
                         "and undefined/absent-ground-truth values rather than fake zeros.")
        if ident in ("GOAL-03", "GOAL-20"):
            evidence += ("|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json"
                         "|tests/test_mahalanobis_epn_holdout_v2.py")
            boundary += (" A precommitted EPN users32-41 confirmation fixes eight-coordinate "
                         "10/20-trial calibration before target loading. Macro-F1 improves "
                         "from .5129/.5115 to .6329/.6836 and loss improves for all ten users "
                         "at both budgets. This does not prove all historical access events, "
                         "few-second calibration burden or own-device efficacy.")
            evidence += ("|feature_bank/HARDWARE_PREPARATION_ACCEPTANCE.json"
                         "|src/emgimu/feature_bank/body_frame_v2.py"
                         "|src/emgimu/feature_bank/electrode_layout_v1.py"
                         "|src/emgimu/feature_bank/fault_gate_evaluation_v1.py"
                         "|tests/test_hardware_preparation_v1.py"
                         "|tests/test_hardware_preparation_delivery.py")
            boundary += (" User items5/6/7 software preparation now verifies explicit "
                         "SI conversion, trial provenance, eight-channel physical-layout "
                         "fingerprints and trial-balanced caller-labelled fault metrics. "
                         "Physical metadata, fault annotations and device efficacy remain unproved.")
            evidence += ("|benchmarks/new_bank_v3/f8_calibrated_manus_v2.py"
                         "|benchmarks/new_bank_v3/F8_CALIBRATED_MANUS_V2_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F8_CALIBRATED_MANUS_V2_RESULTS.json"
                         "|tests/test_f8_calibrated_manus_v2_delivery.py")
            boundary += (" A separate F8 V2 source-user OOF temperature-calibrated "
                         "fusion preserves target split identities; validation loss "
                         "improves but descriptive-final loss worsens versus raw V1, "
                         "so this is no stable default improvement.")
            evidence += ("|src/emgimu/feature_bank/autonomous_g5_stream_v1.py"
                         "|tests/test_autonomous_g5_stream_v1.py"
                         "|benchmarks/new_bank_v3/autonomous_g5_stream_replay.py"
                         "|benchmarks/new_bank_v3/AUTONOMOUS_G5_STREAM_V1_RESULTS.json")
            boundary += (" A chunk-to-release-confirmed G5 API reproduces all 1,257 "
                         "saved detection boundaries/predictions across 28 continuous native "
                         "recordings with unchanged source predictor state; this is not "
                         "hardware throughput or onset-time recognition evidence.")
            evidence += ("|src/emgimu/feature_bank/session_fusion_v2.py"
                         "|tests/test_session_fusion_v2.py"
                         "|benchmarks/new_bank_v3/f8_checked_fusion_replay.py"
                         "|benchmarks/new_bank_v3/F8_CHECKED_FUSION_V2_RESULTS.json")
            boundary += (" The opt-in V2 probability fusion API rejects source/calibration/"
                         "evaluation overlap and provider/class-axis mismatch; 24 frozen "
                         "native blocks preserve their probabilities and reject 96 invalid calls.")
            evidence += ("|benchmarks/new_bank_v3/DETECTED_G5_OUTCOME_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/detected_g5_outcome_audit.py"
                         "|tests/test_detected_g5_outcome_audit.py")
            boundary += (" Full reference accounting separates 516 correct, 234 wrong "
                         "and 100 missed supported actions, retaining 561 unsupported "
                         "references and 21 unmatched detections without inventing neutral truth.")
            evidence += ("|src/emgimu/feature_bank/detected_g5_reader_v1.py"
                         "|tests/test_detected_g5_reader_v1.py"
                         "|tests/test_detected_g5_unibo_v1_delivery.py"
                         "|benchmarks/new_bank_v3/DETECTED_G5_UNIBO_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/DETECTED_G5_UNIBO_V1_RESULTS.json")
            boundary += (" Frozen existing G5 models now consume every complete "
                         "200ms window of the same estimated native intervals, "
                         "with explicit unrepresented tails and source-Day5 "
                         "temperature. No training or default change occurs. "
                         "516/750 supported detections are correctly classified "
                         "(.688), versus .744 on the same oracle intervals; "
                         "end-to-end reference success is 516/850. Native "
                         "four-channel public transfer remains distinct from "
                         "the user's eight-channel live system.")
            evidence += ("|benchmarks/new_bank_v3/DETECTED_DTW_UNIBO_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/DETECTED_DTW_UNIBO_V1_RESULTS.json"
                         "|tests/test_detected_dtw_unibo_v1_delivery.py")
            boundary += (" A complete continuous detected-bout-to-DTW chain "
                         "now replays all 1257 intervals, without changing source "
                         "medoids. Supported matched accuracy .4533 versus "
                         "same-reference oracle accuracy .4413 exposes weak "
                         "gesture separation; 340/850 end-to-end references "
                         "succeed. Unsupported gestures and unmatched detections "
                         "remain explicit, with no default promotion.")
            evidence += ("|feature_bank/FORMULA_NUMERICAL_ACCEPTANCE.json"
                         "|benchmarks/formula_numerical_acceptance.py"
                         "|benchmarks/new_bank_v3/AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json"
                         "|tests/test_autonomous_continuous_unibo_v1_delivery.py"
                         "|src/emgimu/feature_bank/detected_template_reader_v1.py"
                         "|tests/test_detected_template_reader_v1.py"
                         "|benchmarks/new_bank_v3/F8_ROUTER_MANUS_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/F8_ROUTER_MANUS_V1_PROTOCOL.json"
                         "|tests/test_f8_router_manus_v1_delivery.py"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_BUDGET_V1_PROTOCOL.json"
                         "|tests/test_mahalanobis_epn_budget_v1_delivery.py")
            boundary += (" A 32-section explicit arithmetic fixture register "
                         "now records tested assertions separately from efficacy. "
                         "A frozen source-Rest continuous UniBo onset/release "
                         "screen obtains .9833 precision/.8760 recall; estimated "
                         "full-path DTW readout never certifies oracle coverage. "
                         "New four-provider MANUS F8 routing has mixed F1/loss "
                         "changes, so remains opt-in. Independent-trial EPN "
                         "Mahalanobis 8-coordinate 10/20-shot cases improve "
                         "descriptive F1/loss; insufficient 24/36-coordinate "
                         "budgets are explicitly rejected. Paused public body-frame, "
                         "physical electrode order and physical-fault evidence "
                         "remain separate, as do own-device claims.")
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
        if ident in ('GOAL-03','GOAL-20','GOAL-22','GOAL-23'):
            evidence += ('|benchmarks/song_real8/SONG_RAW_QUALITY_V1_PROTOCOL.json'
                         '|benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'
                         '|feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json'
                         '|src/emgimu/feature_bank/source_quality_gate_v1.py'
                         '|src/emgimu/feature_bank/personal_session_stream_v2.py'
                         '|tests/test_source_quality_gate_v1.py'
                         '|tests/test_personal_session_stream_v2.py'
                         '|tests/test_song_raw_quality_v1_delivery.py')
            boundary += (' Separate versioned raw pre-software-highpass F9 integration now uses explicit signed24 transport bounds and source-only quantiles. '
                         'All24 same124-trial cells/2976 predictions are independently verified; structural mode rejects all four injected severe faults and no unmodified trials. '
                         'Unknown hardware truth prevents a real false-positive rate. Soft routing hurts unmodified accuracy and can accept single-channel severe faults, so remains off by default. '
                         'Four full S04 population/session by off/structural streams each reproduce29790 emissions and confirmation labels without source/profile mutation. '
                         'Qt exercises separate raw calibration/replay, retry of invalid calibration trials and explicit Unknown. Numerical/synthetic integration is not physical efficacy or full F0-F9 completion.')
        if ident in ('GOAL-03','GOAL-20','GOAL-21','GOAL-22','GOAL-23'):
            evidence += ('|benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_PROTOCOL.json'
                         '|benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_RESULTS.json'
                         '|feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json'
                         '|src/emgimu/feature_bank/personal_session_decision_v1.py'
                         '|src/emgimu/feature_bank/personal_session_decision_cli_v1.py'
                         '|tests/test_personal_session_decision_v1.py'
                         '|tests/test_integrated_decision_cli_v1.py'
                         '|tests/test_integrated_decision_delivery_v1.py')
            boundary += (' Operational F7/F8/raw-F9 now surrounds six frozen providers: calibration-only ordinary/affine-SPD prototypes contribute probabilities, '
                         'and same-user generic/family drift adjusts weights. Nested0/1/2/5 current budgets retain56 cells/6944 same124-trial predictions, six provider removals and F7/F8/F9 removals. '
                         'Zero current still uses20 long-term trials. Independent geometry/routing/probability and versioned profile/CLI checks pass. '
                         'Five-shot F1/recall improve but loss/Brier worsen, failing the joint guard. No default promotion or full-document/subfamily/device proof.')
            evidence += ('|feature_bank/SONG_DECISION_GUI_V1_ACCEPTANCE.json'
                         '|benchmarks/song_real8/SONG_DECISION_GUI_V1_PROTOCOL.json'
                         '|src/emgimu/feature_bank/personal_session_stream_v3.py'
                         '|tests/test_personal_session_stream_v3.py'
                         '|tests/test_decision_gui_v1_delivery.py')
            boundary += (' A separately frozen operational GUI/stream V3 now provides explicit baseline/F8/F7+F8 choices, persistent new-schema profiles, guided calibration and actual mixed-provider raw F9. '
                         'Ten full-S04 streams/297900 emissions match independent filter/Euclidean/generalized-SPD/weight/gate arithmetic and chronological confirmations under irregular chunks. '
                         'Qt exercises the actual default application entry, personal/new-session lifecycle and packet-loss clearing. '
                         'No stream labels, fitting or source/profile mutation occur. Calibration intervals remain included, so this is execution parity, not new efficacy, hardware throughput or a reversal of negative offline guards.')
        if ident in ('GOAL-03','GOAL-20','GOAL-22'):
            evidence += ('|benchmarks/song_real8/SONG_EXTENDED_WINDOW_V1_PROTOCOL.json'
                         '|benchmarks/song_real8/SONG_EXTENDED_WINDOW_V1_RESULTS.json'
                         '|feature_bank/SONG_EXTENDED_WINDOW_ACCEPTANCE_V1.json'
                         '|benchmarks/song_real8/SONG_EXTENDED_GUI_V1_PROTOCOL.json'
                         '|feature_bank/SONG_EXTENDED_GUI_V1_ACCEPTANCE.json'
                         '|src/emgimu/feature_bank/extended_window_decision_v1.py'
                         '|src/emgimu/feature_bank/extended_window_cli_v1.py'
                         '|src/emgimu/feature_bank/personal_session_stream_v4.py'
                         '|tests/test_extended_window_delivery_v1.py')
            boundary += (' A separately versioned source-only CSP extension retains all six source models and adds16 coordinates, with76 cells/9424 predictions on the same124 trials. '
                         'Five-shot new-seven reliability improves F1 .9154 to.9422 and loss .3303 to.2675, passing all four predeclared old-six comparison guards. '
                         'F4d now emits trial-balanced window/trial/session-minus-long context; it is not a classifier or fatigue measurement. '
                         'Independent native eigen/variance/DFT/prototype/routing checks and ten new-seven full-stream arms/297900 emissions pass. '
                         'The explicit versioned desktop entry offers six/seven selection, separate persistent profiles and spectral context; actual Qt/subprocess lifecycle tests pass. '
                         'New-seven full F7/F8 still has worse loss/Brier than new-seven reliability. No default promotion, all-subfamily/full-bout composition, physical ring/F6, independent user/day or device proof follows.')
        if ident in ('GOAL-03','GOAL-20','GOAL-22'):
            evidence += ('|benchmarks/new_bank_v3/PERSONAL_TEMPORAL_UNIBO_V1_PROTOCOL.json'
                         '|benchmarks/new_bank_v3/PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json'
                         '|feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json'
                         '|src/emgimu/feature_bank/personal_temporal_bouts_v1.py'
                         '|src/emgimu/feature_bank/complete_bout_window_adapter_v1.py'
                         '|src/emgimu/feature_bank/personal_temporal_cli_v1.py'
                         '|tests/test_personal_temporal_unibo_v1_delivery.py')
            boundary += (' Independent full-bout personal/session DTW and path fusion now delivers84 blocks/924 arm cells/187044 retained probabilities, '
                         'with complete-versus-estimated contracts, persistent profiles and eight-channel adapter/CLI checks. '
                         'Recording-level excluded calibration, independent DP/medoid/signature/probability and all weighted metric/cost checks pass. '
                         'Day6 five-shot primary guard fails: F1 rises slightly but log loss/Brier worsen, only1/7 user loss wins. '
                         'Native scope is four-channel200Hz oracle boundaries; physical eight-channel efficacy remains unproven.')
            evidence += ('|feature_bank/TEMPORAL_LIVE_GUI_V1_ACCEPTANCE.json'
                         '|benchmarks/song_real8/TEMPORAL_LIVE_GUI_V1_PROTOCOL.json'
                         '|src/emgimu/feature_bank/temporal_bout_live_v1.py'
                         '|src/emgimu/feature_bank/personal_session_stream_v5.py'
                         '|tests/test_temporal_live_gui_v1_delivery.py')
            boundary += (' Independent V5 full-action desktop/stream composition now has cued calibration, raw/filtered companions, persistent personal/session bundles and neutral-only fixed detector fitting. '
                         'Manual and estimated-auto decisions remain separate from window output. '
                         '197 software checks include real Qt/subprocess acquisition lifecycle, frozen window parity, chunk/direct-interval parity, profile identity, quality Unknown and gap/overflow/censoring. '
                         'Desktop shortcut is installed to main_decision_v3.py; initial six-provider model and off temporal mode are preserved. '
                         'This software scope does not prove native automatic-fusion gain, physical throughput, biological boundaries or independent multi-day eight-channel efficacy.')
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
