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
        if family == "F1":
            evidence += "|tests/test_f1_scale_pattern_oracle.py"
            supported += (" An independent eight-channel RMS/global-RMS oracle "
                          "confirms exact pattern values, scale invariance and "
                          "exclusion of the log global scale from H. New fits "
                          "reject changed channel count or sampling rate.")
            unresolved += (" Historical X1-H equivalence was explicitly "
                           "superseded by the user's versioned replacement.")
        if family == "F4":
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
