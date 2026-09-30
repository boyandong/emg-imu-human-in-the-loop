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
    ("F2", "public_axis_limited", "benchmarks/new_bank_v2/F2_AC_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2B_WEARING_RESULTS.json|benchmarks/new_bank_v2/F2_MANUS_CANDIDATES_RESULTS.json",
     "Trace covariance, document CSP and SPD tangent each have isolated source-only increments on matched native wearing and MANUS session trials.",
     "Two public axes now compare all three candidates; MANUS uses external ROAM Rest and same-user sessions. All additions worsen MANUS validation log loss, and wearing validation/final rankings reverse. No universal default promotion."),
    ("F3", "public_screened", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|feature_bank/results/wearing_ring_session_interaction.json",
     "Independent ring lag and correlation spectrum have seven-axis screens, while a separate raw ring covariance has bounded wearing evidence.",
     "Historical RLCS/CES equivalence is superseded; real own-device electrode re-donning remains unavailable."),
    ("F4", "public_context_limited", "benchmarks/new_bank_v2/FULL_V1_LOFO_AUDIT.json|benchmarks/song_real8/F4D_SESSION_SHIFT_AUDIT.json|benchmarks/new_bank_v2/F4D_MANUS_SESSION_RESULTS.json|benchmarks/new_bank_v2/F4D_MANUS_PREDICTIVE_RESULTS.json",
     "Frequency direction has seven-axis public screening and a four-tone known-signal check; personal spectral shift has Song and six-user MANUS repeated-session diagnostics and a matched predictive control.",
     "MANUS F4d session-centering helps pooled validation and final F1 but final minimum-user F1 declines; medium-speed calibration versus slow/fast evaluation is speed-confounded. No measured-fatigue validation."),
    ("F5", "public_bout_screened", "feature_bank/results/manifests/feature_bank_unibo_sequence_temporal_final_20260928__replay_audit.json",
     "Validated G5 and complete-bout DTW have frozen native UniBo replay; DTW cost/warp/path length has a known-sequence check.",
     "The optional low-order path signature now rejects unverified short windows and has an exact known-path oracle, but no native promotion evidence; live onset/bout segmentation is unavailable."),
    ("F6", "public_context_limited", "feature_bank/results/manifests/feature_bank_manus_full_fusion_final_20260915_v2__replay_audit.json|feature_bank/EXPERIMENT_CAPABILITIES.md",
     "Real public IMU and oracle posture contexts have bounded native results; calibrated body-frame API has a synthetic contract check.",
     "No native calibrated forward-axis/neutral-trial metadata support the proposed gravity-relative body-frame evaluation."),
    ("F7", "public_calibration_screened", "benchmarks/new_bank_v2/V1_FEATURE_ANCHOR_VERIFICATION.json|feature_bank/results/epn_spd_anchor_trial_study.json",
     "Public 0/1/2/5-shot score- and feature-space anchors plus frozen-source SPD tangent prototypes have native trial-aware evidence.",
     "Some high-dimensional classwise Mahalanobis options cannot be estimated from these shot counts; no universal gain."),
    ("F8", "public_session_limited", "feature_bank/results/family_specific_session_shift_audit.json",
     "Long-versus-current family-specific summaries are derived from source/calibration trials; public wearing and MANUS context controls exist.",
     "Own-device true later-day/re-donning validation and a universally beneficial routing rule are unavailable."),
    ("F9", "synthetic_quality_only", "benchmarks/new_bank_v2/ROAM_V1_QUALITY_ANALYSIS_AUDIT.json|feature_bank/results/quality_unknown_replay.json",
     "Availability-aware quality observations, synthetic corruption controls and explicit Unknown decision replay exist.",
     "Unknown ADC range and absent pre-highpass signal are masked; measured hardware-fault prevalence and safe live gating remain unverified."),
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
