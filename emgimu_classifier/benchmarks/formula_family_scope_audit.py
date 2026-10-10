"""Reviewed F0–F9 native-evidence boundaries, separate from formula equivalence."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "feature_bank"

# This is a human-reviewed scope table. Evidence presence/hash does not turn a
# scientific judgement into proof of complete specification acceptance.
REVIEWS = (
    ("F0", "public_screened", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json|benchmarks/song_real8/F0_REST_NOISE_RESULTS.json|benchmarks/song_real8/F0_REST_NOISE_REPORT.md|benchmarks/song_real8/F0_REST_MODEL_RESULTS.json|benchmarks/song_real8/F0_REST_MODEL_REPORT.md",
     "Rest-fitted eight-channel F0v2 has seven public-axis native screens and an exact six-block known-waveform test. A source-frozen Song 250 Hz replay proves that Rest-only threshold selection changes native F0 counts. Matched source-trained classifiers test the corresponding predictive change on the same native trials.",
     "Rest-only thresholds worsen S03 trial F1/loss/Brier and S04 F1 against pooled-source thresholds; no default promotion. The single user, same-day sessions, failed readiness and prior S04 inspection limit generalization. Historical R0 extras are user-superseded; own-device independent cohort remains unavailable."),
    ("F1", "public_screened", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json",
     "Independent scale-pattern formula is tested across seven public axes and in the complete-bank LOFO.",
     "This is not the missing historical X1-H source or a universally winning addition."),
    ("F2", "public_axis_limited", "benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2B_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_RESULTS.json|benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_RESULTS.json|benchmarks/new_bank_v2/F2_CROSS_AXIS_REVIEW_AUDIT.json|benchmarks/new_bank_v2/F2A_DOCUMENT_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v2/DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_RESULTS.json|benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_GRAB_RESULTS.json|benchmarks/new_bank_v3/SPEC_F2C_GRAB_REPORT.md|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_REPORT.md",
     "Goal F2a is centered; F2b alone specifies uncentered XX transpose. V3 centered F2a/F2c have independent oracles, including the corrected zero-variance F2c tangent origin, checksum-frozen GRAB cross-day increments, and a three-axis centered F2c matched-trial replay. Earlier V2 uncentered F2a/F2c wearing, MANUS and GRAB results are preserved as alternative-feature evidence.",
     "V3 F2a worsens GRAB Day2/Day3 F1/loss. V3 F2c gains wearing validation F1/loss, but MANUS validation loss and GRAB unseen-user F1/loss worsen; the public universal-default guard fails. Its GRAB cross-day validation gain also reverses on descriptive final. Inspected cohorts are not prospective independent tests."),
    ("F3", "public_screened", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|feature_bank/results/wearing_ring_session_interaction.json|benchmarks/new_bank_v2/DOCUMENT_SPATIAL_CROSS_AXIS_AUDIT.json|benchmarks/new_bank_v2/F3C_CHANNEL_ORDER_RESULTS.json|benchmarks/new_bank_v2/F3C_DOCUMENT_GRAB_RESULTS.json|benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_RESULTS.json|benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_REPORT.md",
     "Independent ring lag and correlation spectrum have seven-axis screens. V3 F3c now derives from goal-centered F2a with direct-matrix and rotation oracles and a frozen 2048 Hz GRAB three-day ring1 screen; older V2 uncentered wearing/GRAB results retain alternative-feature identity.",
     "The Myo CSV ring order remains unattested. The official GRAB diagram shows visible 8-1-2 adjacency, but the hidden-side sequence is inferred. V3 F3c worsens GRAB Day2/Day3 F1 and log loss versus F0; no universal or own-device default is supported. Historical RLCS/CES equivalence is superseded."),
    ("F4", "public_context_limited", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|benchmarks/song_real8/F4D_SESSION_SHIFT_AUDIT.json|benchmarks/new_bank_v2/F4D_MANUS_SESSION_RESULTS.json|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_RESULTS.json|benchmarks/new_bank_v2/F4D_MANUS_SAME_SPEED_RESULTS.json",
     "Frequency direction has seven-axis public screening and a four-tone known-signal check; personal spectral shift has Song and six-user MANUS repeated-session diagnostics, frozen predictive control and matched-speed leave-one-trial-out control.",
     "MANUS F4d same-speed session-centering helps pooled validation F1/loss but final minimum-user F1 declines. The offline reference excludes the held gesture, so this does not prove a deployable session correction or measured fatigue effect."),
    ("F5", "public_bout_screened", "feature_bank/results/manifests/feature_bank_unibo_sequence_temporal_final_20260928__replay_audit.json|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json",
     "Validated G5, complete-bout DTW and optional order-two F5c have frozen native UniBo evidence; F5c rejects short windows and has a known-path oracle.",
     "G5+F5c improves Day6 weighted macro F1 but worsens log loss; Days7-8 pooled F1 slips. Complete-bout oracle boundaries do not provide live onset/bout segmentation or eight-channel device transfer."),
    ("F6", "public_context_limited", "feature_bank/results/manifests/feature_bank_manus_full_fusion_final_20260915_v2__replay_audit.json|feature_bank/EXPERIMENT_CAPABILITIES.md",
     "Real public IMU and oracle posture contexts have bounded native results; the calibrated body-frame API has rotation, trial-exclusion and independent 15-output analytical checks.",
     "No native calibrated forward-axis/neutral-trial metadata support the proposed gravity-relative body-frame evaluation."),
    ("F7", "public_calibration_screened", "benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_VERIFICATION.json|feature_bank/results/epn_spd_anchor_trial_study.json|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json|benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_REPORT.md",
     "Public 0/1/2/5-shot score- and feature-space anchors plus frozen-source SPD tangent prototypes have native trial-aware evidence. The opt-in document-exact additive-epsilon anchor has near-zero-scale independent oracles and a frozen 1/2/5-shot GRABMyo native replay.",
     "The document-exact fixed mixture worsens Day2 log loss at every budget; its probability readout is exploratory, not a default. Some high-dimensional classwise Mahalanobis options cannot be estimated from these shot counts."),
    ("F8", "public_session_limited", "feature_bank/results/family_specific_session_shift_audit.json|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json|benchmarks/new_bank_v2/F8_GRAB_DAY_RESULTS.json|benchmarks/new_bank_v2/DOCUMENT_SESSION_RESULTS.json",
     "Long-versus-current summaries use source/calibration trials; MANUS and wearing controls, Song Rest-noise shifts, and a fixed two-provider GRAB cross-day predictive replay are available. A separate document-exact session normalization route retains 1,440 paired native provider predictions.",
     "The GRAB Day2/Day3 F8 rule changes no gesture decisions and improves log loss by only 0.000219/0.000525. It is retrospective on inspected public subjects, not a deployment gain or a Song Rest-noise routing result; own-device session behavior is unavailable."),
    ("F9", "public_gate_negative", "benchmarks/new_bank_v2/ROAM_V1_QUALITY_ANALYSIS_AUDIT.json|feature_bank/results/quality_unknown_replay.json|benchmarks/song_real8/QUALITY_MASK_V1.json|benchmarks/song_real8/F9_STRUCTURAL_SONG_RESULTS.json|benchmarks/new_bank_v2/F9_GRAB_GATE_RESULTS.json|benchmarks/new_bank_v2/F9_STRUCTURAL_GRAB_RESULTS.json|benchmarks/new_bank_v2/F9_WEARING_STRUCTURAL_RESULTS.json|benchmarks/new_bank_v3/F9_DOCUMENT_OBSERVATIONS_REPORT.md|benchmarks/song_real8/F9_DOCUMENT_V3_RESULTS.json|benchmarks/song_real8/F9_DOCUMENT_V3_REPORT.md|tests/test_document_quality_v3.py|tests/test_f9_document_song_delivery.py",
     "Availability-aware quality observations have direct-Fourier line/low-band, robust amplitude, neighbor-correlation and trace-covariance oracles. Opt-in V3 primitives additionally check strict threshold boundaries and additive-epsilon line/MAD denominators, without changing legacy gating; 849 source and 847 held-out Song raw windows confirm native shape, finite outputs and unavailable metadata. Frozen Song and GRAB F0 controls show the soft gate's false rejections; the separate severe structural rule rejects no natural Song, GRAB or electrode-re-wearing trials and detects copied constant channels.",
     "The illustrative 0.5 gate rejects mostly correct F0 trials on Song and GRAB; the severe structural rule catches synthetic failures but no natural F0 errors. Physical-fault ground truth, independent own-device cohort and safe live gating remain unresolved."),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    formula = OUT / "FORMULA_IMPLEMENTATION_AUDIT.csv"
    with formula.open(newline="", encoding="utf-8") as stream:
        reviewed = list(csv.DictReader(stream))
    if len(reviewed) < 39 or len({r["item_id"] for r in reviewed}) != len(reviewed):
        raise AssertionError("source formula review inventory is incomplete")
    rows = []
    for family, status, evidence, supported, unresolved in REVIEWS:
        if family == "F5":
            evidence += ("|src/emgimu/feature_bank/detected_g5_reader_v1.py"
                         "|tests/test_detected_g5_reader_v1.py"
                         "|tests/test_detected_g5_unibo_v1_delivery.py"
                         "|benchmarks/new_bank_v3/DETECTED_G5_UNIBO_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/DETECTED_G5_UNIBO_V1_RESULTS.json")
            supported += (" Existing frozen source-Days1-5 G5 classifiers and "
                          "source-Day5 temperatures are replayed on the identical "
                          "1257 detected intervals without training. All complete "
                          "200ms windows are averaged, with tails reported. "
                          "Supported matched G5 accuracy is .688 versus DTW .4533; "
                          "same-reference G5 oracle accuracy is .744, and G5 "
                          "end-to-end success is 516/850. Per-user/class and paired "
                          "error counts remain explicit.")
            unresolved += (" Classification and segmentation errors remain; "
                           "one of seven users loses against DTW. This stronger "
                           "public four-muscle pipeline does not establish "
                           "performance for an eight-electrode device or justify "
                           "a change to the collection application's default.")
            evidence += ("|benchmarks/new_bank_v3/DETECTED_DTW_UNIBO_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/DETECTED_DTW_UNIBO_V1_RESULTS.json"
                         "|tests/test_detected_dtw_unibo_v1_delivery.py")
            supported += (" The estimated-boundary reader is now exercised in "
                          "a native continuous recording chain against the same "
                          "source twenty-candidate DTW medoids. Of 850 supported "
                          "active references, 750 match detections; 340 are "
                          "classified correctly. On these same 750 references, "
                          "oracle-boundary nearest-template accuracy is .4413 "
                          "versus .4533 with estimated boundaries.")
            unresolved += (" End-to-end supported-reference success is .40; "
                           "486 matched unsupported gestures and 21 unmatched "
                           "detections are separately retained. The standalone "
                           "DTW gesture decision remains weak even with oracle "
                           "boundaries, rather than establishing a live model.")
            evidence += ("|src/emgimu/feature_bank/autonomous_bouts_v1.py"
                         "|src/emgimu/feature_bank/detected_template_reader_v1.py"
                         "|tests/test_autonomous_bouts_v1.py"
                         "|tests/test_detected_template_reader_v1.py"
                         "|tests/test_autonomous_continuous_unibo_v1_delivery.py"
                         "|benchmarks/new_bank_v3/AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json")
            supported += (" Source-Day1-Rest-fitted causal onset/release detection "
                          "is replayed on all 28 uninterrupted Day6 recordings: "
                          "1236/1411 reference intervals matched, 1257 detections, "
                          "precision .9833 and recall .8760. A separate DTW reader "
                          "uses the entire detected native RMS path and explicitly "
                          "marks estimated rather than certified boundaries.")
            unresolved += (" Protocol intervals do not measure physiological "
                           "onset; missed bouts and .23/.35-second onset/offset "
                           "errors remain. This is retrospective boundary evidence, "
                           "not a native classification gain or device test.")
            evidence += ("|src/emgimu/feature_bank/document_path_v3.py"
                         "|tests/test_document_path_v3.py"
                         "|tests/test_f5_path_unibo_delivery.py"
                         "|benchmarks/new_bank_v3/F5_PATH_UNIBO_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F5_PATH_UNIBO_RESULTS.json")
            supported += (" Separate document F5b/c V3 uses additive-epsilon "
                          "complete-envelope normalization. Source-only medoids "
                          "retain unique trial identities; query overlap and "
                          "sensor/path-contract changes reject. Independent exhaustive "
                          "DTW and near-zero polygon oracles pass. Frozen candidate "
                          "IDs and all Day6/7/8 native complete bouts are replayed.")
            unresolved += (" This coordinate/nearest-template replay does not "
                           "establish automatic boundaries, calibrated fusion gains "
                           "or live-device transfer.")
        if family == "F8":
            evidence += "|tests/test_session_shift_summary.py|src/emgimu/feature_bank/session_shift_summary.py"
            supported += (" The family-summary API retains immutable long-term trial "
                          "identities and rejects any calibration/source overlap or "
                          "missing provenance. Independent tests verify equal trial "
                          "mass with unequal window counts and unchanged source state.")
            evidence += ("|src/emgimu/feature_bank/document_session_v3.py"
                         "|tests/test_document_session_v3.py"
                         "|tests/test_f8_document_manus_delivery.py"
                         "|benchmarks/new_bank_v3/F8_DOCUMENT_MANUS_RESULTS.json")
            supported += (" The opt-in document V3 assembles all four F8 blocks "
                          "with explicit user/session separation, document-centered SPD "
                          "geometry and channel-quality differences. Frozen MANUS "
                          "calibration splits yield 24 finite 180-dimensional descriptors, "
                          "with no evaluation windows or classifier fitting.")
        if family == "F1":
            evidence += ("|src/emgimu/feature_bank/document_scale_v3.py"
                         "|tests/test_document_scale_v3.py"
                         "|tests/test_f1_scale_grab_delivery.py"
                         "|benchmarks/new_bank_v3/F1_SCALE_GRAB_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F1_SCALE_GRAB_RESULTS.json")
            supported += (" Separate document F1 V3 follows the additive-epsilon "
                          "RMS/global-RMS formula with ordinary/zero/near-zero "
                          "numeric oracles and source-contract checks. Global "
                          "log scale is available only through a separate context "
                          "API; the matched GRAB increment adds eight H features.")
            unresolved += (" The exact F1 V3 addition loses validation and "
                           "descriptive-final F1/log loss; remains opt-in. "
                           "Additive epsilon is not exact scale invariance near zero.")
            evidence += "|tests/test_f1_scale_pattern_oracle.py"
            supported += (" An independent eight-channel RMS/global-RMS oracle "
                          "confirms exact pattern values, scale invariance and "
                          "exclusion of the log global scale from H. New fits "
                          "reject changed channel count or sampling rate.")
            unresolved += (" Historical X1-H equivalence was explicitly "
                           "superseded by the user's versioned replacement.")
        if family == "F4":
            evidence += ("|src/emgimu/feature_bank/document_spectral_v3.py"
                         "|tests/test_document_spectral_v3.py"
                         "|tests/test_f4_spectral_grab_delivery.py"
                         "|benchmarks/new_bank_v3/F4_SPECTRAL_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F4_SPECTRAL_GRAB_PROTOCOL.json")
            supported += (" Separate document F4 V3 uses additive epsilon, raw "
                          "spectral entropy and explicit orthonormal DCT-II. "
                          "Independent normal/zero/near-zero Fourier and DCT "
                          "oracles and source-grid contracts pass. A fixed GRAB "
                          "unseen-user 112/56/56-trial paired increment is saved.")
            unresolved += (" F4 V3 gains validation macro F1/loss but loses "
                           "descriptive final F1/loss; no default promotion.")
            evidence += "|tests/test_f4a_frequency_coord_oracle.py"
            supported += (" Independent direct-complex-DFT 250 Hz checks verify "
                          "sub-Nyquist bandwise channel orientation, and newly "
                          "fitted spectral states reject a changed sampling rate.")
            unresolved += (" Persisted older spectral fits predate the new "
                           "sample-rate field and retain legacy compatibility.")
        if family == "F3":
            evidence += "|tests/test_f3a_rlcs_direct_oracle.py"
            supported += (" A direct Pearson known-envelope F3a test confirms "
                          "circular-lag means/standard deviations and new fits "
                          "reject a different sampling rate.")
            unresolved += (" Persisted older ring fits predate the rate field; "
                           "historical RLCS identity remains user-superseded.")
            evidence += ("|benchmarks/new_bank_v3/F3B_CES_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F3B_CES_GRAB_REPORT.md"
                         "|tests/test_document_ces_v3.py")
            supported += (" A separate V3 F3b CES passes a known 3+1 spectrum and "
                          "permutation oracle, with a frozen 112/56/56-trial GRAB "
                          "unseen-user matched F0v2 increment.")
            unresolved += (" The isolated V3 CES addition worsens validation and "
                           "descriptive final F1 and log loss; no default promotion.")
        if family == "F5":
            evidence += ("|benchmarks/new_bank_v3/F5_TEMPORAL_GRAB_RESULTS.json"
                         "|benchmarks/new_bank_v3/F5_TEMPORAL_GRAB_REPORT.md"
                         "|tests/test_document_temporal_v3.py")
            supported += (" A separate V3 short-window family follows the goal's raw "
                          "ratio, slope, peak and entropy equations with a known-waveform "
                          "oracle and a frozen GRAB unseen-user paired increment.")
            unresolved += (" The V3 F5 addition worsens GRAB validation/final F1 "
                           "and log loss and is not a full-bout or live detector.")
        if family == "F7":
            evidence += ("|src/emgimu/feature_bank/trial_mahalanobis_v1.py"
                         "|tests/test_trial_mahalanobis_v1.py"
                         "|tests/test_mahalanobis_epn_budget_v1_delivery.py"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_BUDGET_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json")
            supported += (" Independent-trial Mahalanobis budgets are screened "
                          "on previously inspected EPN users22-31 at 1/2/5/10/20 "
                          "shots, with fixed evaluation trials across budgets. "
                          "8-coordinate 10/20-shot models improve descriptive "
                          "F1/loss over same-budget Euclidean anchors; windows "
                          "cannot inflate the dimension+2 per-class trial guard.")
            unresolved += (" 24/36-coordinate budgets remain ineligible rather "
                           "than silently reduced or estimated from dependent "
                           "windows. Larger native trial budgets and live accuracy "
                           "are not established by this study.")
            evidence += ("|benchmarks/new_bank_v3/F7_AFFINE_EPN/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_EPN_REPORT.md"
                         "|benchmarks/new_bank_v3/F7_AFFINE_CORE_MATCHED/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_CORE_MATCHED_REPORT.md"
                         "|tests/test_affine_spd_anchor.py"
                         "|tests/test_f7_affine_epn_delivery.py"
                         "|tests/test_f7_affine_core_matched_delivery.py")
            supported += (" A separate exact affine-invariant matrix-log F7 candidate "
                          "has nonorthogonal geometry/trial-mass oracles, native "
                          "six-user 1/2/5-shot trial screens and a positive matched "
                          "fixed-Core increment beyond a uniform-softening control.")
            unresolved += (" The users were previously inspected; this is not "
                           "prospective validation, a learned concatenated Core, "
                           "or grounds for own-device default promotion.")
            evidence += ("|benchmarks/new_bank_v3/F7_AFFINE_FRESH_PROTOCOL.md"
                         "|benchmarks/new_bank_v3/F7_AFFINE_FRESH/results.json"
                         "|benchmarks/new_bank_v3/F7_AFFINE_FRESH_REPORT.md"
                         "|tests/test_f7_affine_fresh_protocol.py"
                         "|tests/test_f7_affine_fresh_delivery.py")
            supported += (" A separately precommitted first-read users22-31 cohort "
                          "passes all five fixed five-shot guards on 1,200 held-out "
                          "native trials, with 7/10 user log-loss gains.")
            unresolved += (" The older inspected-cohort limitation does not apply "
                           "to the fresh users22-31 readout, but public gesture transfer "
                           "still does not establish own-device/live reliability.")
        if family == "F8":
            evidence += ("|src/emgimu/feature_bank/session_router_v1.py"
                         "|tests/test_session_router_v1.py"
                         "|tests/test_f8_router_manus_v1_delivery.py"
                         "|benchmarks/new_bank_v3/F8_ROUTER_MANUS_V1_PROTOCOL.json"
                         "|benchmarks/new_bank_v3/F8_ROUTER_MANUS_V1_RESULTS.json")
            supported += (" New F8 V3 drift/source-class-geometry routing is "
                          "tested on 24 frozen MANUS one/two-shot blocks using "
                          "identical TD24/pattern/SPD/log-band providers. Native "
                          "weights, probabilities and paired metrics are retained; "
                          "three of four pooled log-loss comparisons improve.")
            unresolved += (" F1 changes are mixed and one loss comparison "
                           "worsens. No stable F8 advantage or default promotion "
                           "is established; TD24 omits unavailable Rest-noise "
                           "threshold features and is not document six-block F0.")
        if family == "F6":
            evidence += ("|feature_bank/F6_PUBLIC_ELIGIBILITY.md"
                         "|benchmarks/new_bank_v3/EPN107_F6_ELIGIBILITY.json"
                         "|benchmarks/new_bank_v3/epn107_f6_eligibility_probe.py"
                         "|tests/test_epn107_f6_eligibility.py")
            supported += (" A four-candidate public-source eligibility review "
                          "checks raw IMU and anatomical-axis provenance before F6 use. "
                          "All 107 published EPN107 MAT schemas were then checked: "
                          "38 Myo members have raw six-axis IMU, 69 gForce members "
                          "have empty accel/gyro arrays in every subset.")
            unresolved += (" The full EPN107 archive has no explicit "
                           "device-frame anatomical forward-axis or IMU-unit fields; "
                           "binary sync labels and quaternions do not replace them. "
                           "Strict F6 native evaluation is ineligible here.")
        if family == 'F9':
            evidence += ('|benchmarks/song_real8/SONG_RAW_QUALITY_V1_PROTOCOL.json'
                         '|benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'
                         '|feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json'
                         '|tests/test_source_quality_gate_v1.py'
                         '|tests/test_personal_session_stream_v2.py'
                         '|tests/test_song_raw_quality_v1_delivery.py')
            supported += (' A versioned GUI raw-input gate uses explicit signed24 transport extrema and source-only thresholds. '
                          'Independent raw-mask and fusion oracles verify24 cells/2976 predictions. Structural mode rejects '
                          'all124 trials in each of four artificial fault scenarios and no unmodified trials. '
                          'Four whole-recording streams verify119160 emissions and chronological confirmations; invalid guided trials are not counted.')
            unresolved += (' Unmodified hardware-fault truth is unknown, preventing a physical false-positive rate. '
                           'Soft routing changes3 correct unmodified trials to wrong, none to correct, and can accept severe single-channel faults. '
                           'Quality remains off by default; numerical/synthetic checks do not validate physical faults or full-bank efficacy.')
        if family in ('F7','F8','F9'):
            evidence += ('|benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_RESULTS.json'
                         '|feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json'
                         '|src/emgimu/feature_bank/personal_session_decision_v1.py'
                         '|tests/test_personal_session_decision_v1.py'
                         '|tests/test_integrated_decision_cli_v1.py'
                         '|tests/test_integrated_decision_delivery_v1.py')
            supported += (' A separate operational six-provider decision layer now fuses ordinary/affine-SPD F7 probabilities and applies calibration-only F8 drift/geometry weights, with optional raw F9. '
                          'Same124 native evaluation trials yield56 cells/6944 predictions and all six provider removals. Independent prototypes, generalized SPD eigenvalues, routing and persistence are verified; a label-free CLI supports separate-process lifecycle.')
            evidence += ('|feature_bank/SONG_DECISION_GUI_V1_ACCEPTANCE.json'
                         '|benchmarks/song_real8/SONG_DECISION_GUI_V1_PROTOCOL.json'
                         '|tests/test_decision_gui_v1_delivery.py'
                         '|tests/test_personal_session_stream_v3.py')
            supported += (' The versioned application now has explicit operational F7/F8 controls with persistent decision profiles and raw F9. '
                          'Ten full-S04 streams/297900 emissions match independent filtering/geometry/fusion/gating and chronological confirmation. '
                          'Actual Qt/subprocess tests exercise guided personal/new-session registration and packet-loss clearing. Source/profile state stays immutable.')
            unresolved += (' Five-shot full F1/pinch recall improve, but loss/Brier worsen; joint guard fails. No default promotion. '
                           'Continuous numerical integration includes calibration intervals, does not score accuracy and does not cover every document subfamily or device efficacy.')
        if family in ('F2','F4','F7','F8','F9'):
            evidence += ('|benchmarks/song_real8/SONG_EXTENDED_WINDOW_V1_RESULTS.json'
                         '|feature_bank/SONG_EXTENDED_WINDOW_ACCEPTANCE_V1.json'
                         '|feature_bank/SONG_EXTENDED_GUI_V1_ACCEPTANCE.json'
                         '|src/emgimu/feature_bank/extended_window_decision_v1.py'
                         '|src/emgimu/feature_bank/extended_window_cli_v1.py'
                         '|src/emgimu/feature_bank/personal_session_stream_v4.py'
                         '|tests/test_extended_window_delivery_v1.py')
            supported += (' A separate source-only CSP/16-coordinate extension retains six original models byte-identical. '
                          'Seven declared window groups/281coordinates preserve76 cells/9424 predictions and all seven frozen-weight removals. '
                          'Five-shot reliability F1 .9154 to.9422 and loss .3303 to.2675 pass all four declared old-six comparison guards. '
                          'Independent CSP eigen/variance, direct DFT context and trial-balanced prototype/routing checks pass. '
                          'V4 GUI/subprocess supports explicit six/seven selection, extended profiles and context-only F4d; ten full-S04 streams verify297900 emissions without source/profile mutation.')
            unresolved += (' This positive CSP increment is retrospective on one user/day, not a reversal of historical public-cohort negatives. '
                           'New-seven F7/F8 still worsens probability loss/Brier versus new-seven reliability despite higher F1; no default promotion. '
                           'F4d is only background/session coordinates, not a fatigue measurement. Full-bout F5, physical ring/F6 and independent deployment remain outside this window-bank scope.')
        if family=='F5':
            evidence += ('|benchmarks/new_bank_v3/PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json'
                         '|feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json'
                         '|src/emgimu/feature_bank/personal_temporal_bouts_v1.py'
                         '|src/emgimu/feature_bank/complete_bout_window_adapter_v1.py'
                         '|tests/test_personal_temporal_unibo_v1_delivery.py')
            supported += (' Independent full-bout personal/current-session DTW and order1/2 signature fusion now retains84 blocks/924 arms/187044 native probabilities on4251 matched UniBo bouts. '
                          'Independent DP/medoids/signatures, recording-level calibration exclusion, equal-trial metrics, cost ledger and saved-profile state checks pass. '
                          'Eight-channel250Hz adapter and label-free CLI checks preserve raw-quality Unknown and complete-versus-estimated provenance.')
            unresolved += (' Day6 five-shot full F1 .7705 to.7763 but loss .3906 to.4558 and Brier worsen; only1/7 users improves loss. Primary guard fails. '
                           'This native result is inspected four-channel200Hz oracle-boundary evidence, not eight-channel accuracy or automatic-fusion efficacy. No default promotion.')
            evidence += ('|feature_bank/TEMPORAL_LIVE_GUI_V1_ACCEPTANCE.json'
                         '|src/emgimu/feature_bank/temporal_bout_live_v1.py'
                         '|src/emgimu/feature_bank/personal_session_stream_v5.py'
                         '|tests/test_temporal_live_gui_v1_delivery.py')
            supported += (' Independent V5 desktop entry now provides full-action cued registration, saved raw/filtered input, profile/session reload, manual and estimated-auto decisions. '
                          'Fixed detector uses only neutral calibration; window-off identity, chunk/direct-interval parity, quality Unknown and gap/overflow/censoring checks pass. '
                          'Actual Qt/subprocess acquisition lifecycle and whole collection regression total197 tests; desktop shortcut targets the new entry.')
            unresolved += (' Software-only deterministic eight-channel fixtures do not prove a native automatic-fusion gain or physical/live accuracy; automatic boundaries remain estimated and output waits for interval completion.')
        if family=='F5':
            evidence += ('|benchmarks/new_bank_v3/DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_PROTOCOL.json'
                         '|benchmarks/new_bank_v3/DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json'
                         '|feature_bank/DETECTED_PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V3.json'
                         '|tests/test_detected_personal_temporal_unibo_v3_delivery.py')
            supported += (' New continuous fusion preserves frozen detector/G5 outputs while excluding140 complete calibration recording regions and their resampling guard. '
                          'Independent source-clock/pair exclusion,616 arm cells/273328 probabilities and identical budget axes pass. '
                          'All596 supported references retain69 misses; full five-shot changes366 to367 correct.')
            unresolved += (' Conditional loss improves .8001 to.7005, but only2/7 user loss wins; u07 has no matched fist. '
                           'Complete-class primary remains ineligible and fails. No default/8-channel/physical efficacy claim.')
        if family in ('F5','F8'):
            evidence += ('|benchmarks/new_bank_v3/CALIBRATION_REST_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
                         '|benchmarks/new_bank_v3/CALIBRATION_REST_CONTINUOUS_UNIBO_V1_RESULTS.json'
                         '|feature_bank/CALIBRATION_REST_CONTINUOUS_UNIBO_ACCEPTANCE_V1.json'
                         '|tests/test_calibration_rest_continuous_unibo_v1_delivery.py')
            supported += (' Registered-neutral automatic thresholds now reuse exact long/current profiles and fixed596 supported references. '
                          'Independent energy/FSM boundaries, direct G5 arithmetic and308 arm cells/192764 probability values pass. '
                          'Five-shot full changes367 to407 correct;5/7 user success wins and all six frozen guards pass. '
                          'Supported misses fall69 to16; u07 now matches20/30 fist references.')
            unresolved += (' The matched subsets differ, so conditional classification scores are descriptive. '
                           'u07 fist classification still gets only9/30 references correct. This is previously inspected four-channel200Hz evidence; '
                           'eight-channel highpass/current-device efficacy and chronological onboarding remain unproven. No default promotion.')
        if family=='F9':
            evidence += ('|feature_bank/AVAILABLE_BANK_QUALITY_ACCEPTANCE_V2.json'
                         '|src/emgimu/feature_bank/available_bank_quality_fusion_v2.py'
                         '|tests/test_quality_mask_product_oracle_v2.py'
                         '|tests/test_available_bank_quality_v2_delivery.py')
            supported += (' A generic opt-in available-provider quality interface now retains strict trial/policy identity, unavailable observations, per-row weights and scoreable Unknown fallback. '
                          'Direct seven-factor source-quantile mask arithmetic and288 frozen-probability algebra fixtures pass;96 invalid contracts reject.')
            unresolved += (' The native probability algebra uses synthetic quality fixtures; measured quality-effect and hardware-fault efficacy remain unproved. No default promotion or full-bank representation claim.')
        paths = [ROOT / part for part in evidence.split("|")]
        if any(not path.is_file() for path in paths):
            raise AssertionError(f"missing F0–F9 public evidence: {family}")
        rows.append({"family": family, "public_evidence_scope": status,
                     "evidence": evidence,
                     "evidence_sha256_json": json.dumps({path.relative_to(ROOT).as_posix(): sha(path)
                                                         for path in paths}, sort_keys=True),
                     "supported_conclusion": supported, "remaining_boundary": unresolved})
    if [r["family"] for r in rows] != [f"F{i}" for i in range(10)]:
        raise AssertionError("F0–F9 family inventory changed")
    path = OUT / "FORMULA_FAMILY_SCOPE_AUDIT.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"completion_proven": False, "families": len(rows),
              "formula_review_sha256": sha(formula), "formula_review_rows": len(reviewed),
              "csv_sha256": sha(path),
              "scope_counts": dict(Counter(r["public_evidence_scope"] for r in rows)),
              "boundary": "Family-level public-data eligibility and limitation audit. It does not prove every subformula or every applicable native evaluation; measured own-device hardware/data claims remain separate."}
    (OUT / "FORMULA_FAMILY_SCOPE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n",
                                                         encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build()))
