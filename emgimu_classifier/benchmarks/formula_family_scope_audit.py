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
    ("F0", "public_screened", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json",
     "Rest-fitted eight-channel F0v2 has seven public-axis native screens and an exact six-block known-waveform test.",
     "Historical R0 extras are user-superseded; own-device independent cohort remains unavailable."),
    ("F1", "public_screened", "benchmarks/new_bank_v2/V1_CROSS_AXIS_DECISION_AUDIT.json",
     "Independent scale-pattern formula is tested across seven public axes and in the complete-bank LOFO.",
     "This is not the missing historical X1-H source or a universally winning addition."),
    ("F2", "public_axis_limited", "benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2B_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_RESULTS.json|benchmarks/new_bank_v2/F2_GRAB_CANDIDATES_RESULTS.json",
     "Trace covariance, document CSP and SPD tangent each have isolated source-only increments on matched native wearing, MANUS session and GRAB unseen-user trials.",
     "Three public axes compare all candidates: MANUS gains pooled F1 with worse validation log loss; wearing validation/final rankings reverse; all F2 additions hurt GRAB unseen-user F1/log loss. No universal default promotion."),
    ("F3", "public_screened", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|feature_bank/results/wearing_ring_session_interaction.json",
     "Independent ring lag and correlation spectrum have seven-axis screens, while a separate raw ring covariance has bounded wearing evidence.",
     "Historical RLCS/CES equivalence is superseded; real own-device electrode re-donning remains unavailable."),
    ("F4", "public_context_limited", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|benchmarks/song_real8/F4D_SESSION_SHIFT_AUDIT.json|benchmarks/new_bank_v2/F4D_MANUS_SESSION_RESULTS.json|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_RESULTS.json",
     "Frequency direction has seven-axis public screening and a four-tone known-signal check; personal spectral shift has Song and six-user MANUS repeated-session diagnostics and a matched predictive control.",
     "MANUS F4d session-centering helps pooled validation and final F1 but final minimum-user F1 declines; medium-speed calibration versus slow/fast evaluation is speed-confounded. No measured-fatigue validation."),
    ("F5", "public_bout_screened", "feature_bank/results/manifests/feature_bank_unibo_sequence_temporal_final_20260928__replay_audit.json|benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json",
     "Validated G5, complete-bout DTW and optional order-two F5c have frozen native UniBo evidence; F5c rejects short windows and has a known-path oracle.",
     "G5+F5c improves Day6 weighted macro F1 but worsens log loss; Days7-8 pooled F1 slips. Complete-bout oracle boundaries do not provide live onset/bout segmentation or eight-channel device transfer."),
    ("F6", "public_context_limited", "feature_bank/results/manifests/feature_bank_manus_full_fusion_final_20260915_v2__replay_audit.json|feature_bank/EXPERIMENT_CAPABILITIES.md",
     "Real public IMU and oracle posture contexts have bounded native results; the calibrated body-frame API has rotation, trial-exclusion and independent 15-output analytical checks.",
     "No native calibrated forward-axis/neutral-trial metadata support the proposed gravity-relative body-frame evaluation."),
    ("F7", "public_calibration_screened", "benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_VERIFICATION.json|feature_bank/results/epn_spd_anchor_trial_study.json",
     "Public 0/1/2/5-shot score- and feature-space anchors plus frozen-source SPD tangent prototypes have native trial-aware evidence; Euclidean, standardized and cosine coordinates have an independent two-dimensional oracle.",
     "Some high-dimensional classwise Mahalanobis options cannot be estimated from these shot counts; no universal gain."),
    ("F8", "public_session_limited", "feature_bank/results/family_specific_session_shift_audit.json|benchmarks/song_real8/F8_REST_NOISE_SHIFT.json",
     "Long-versus-current family-specific summaries are derived from source/calibration trials; public wearing and MANUS context controls plus one-person Song calibration-only Rest-noise shifts exist.",
     "Song is same-person/same-day, S03 readiness failed and the Rest-noise descriptor has no matched predictive-routing increment; true later-day/re-donning validation is unavailable."),
    ("F9", "one_person_quality_candidate", "benchmarks/new_bank_v2/ROAM_V1_QUALITY_ANALYSIS_AUDIT.json|feature_bank/results/quality_unknown_replay.json|benchmarks/song_real8/QUALITY_MASK_V1.json",
     "Availability-aware quality observations have direct-Fourier line/low-band, robust amplitude and neighbor-correlation oracles; explicit Unknown replay and a source-frozen Song mask with synthetic constant-channel corruption and frozen F0 trial-ID joining exist.",
     "The illustrative 0.5 gate rejects mostly correct F0 trials on both held-out sessions; there is no physical-fault ground truth, and safe live gating remains unresolved."),
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
