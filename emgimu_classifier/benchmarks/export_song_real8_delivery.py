"""Export frozen Song trial scores into versioned Feature Bank source tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RUN_ID = "feature_bank_song_real8_spd_delivery_20260924"
DATASET = "song_real8_one_person_one_day"
CONDITION = "completed_valid_cued_stable_formal"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(calibration_audit: Path, spd_study: Path, output: Path) -> dict:
    audit = json.loads(calibration_audit.read_text(encoding="utf-8"))
    study = json.loads(spd_study.read_text(encoding="utf-8"))
    if audit["status"] != "held_out_cued_stable_trial_reliability_not_live_calibration":
        raise ValueError("Unexpected Song reliability audit status")
    families, increments, complements, curves = [], [], [], []
    for session, phase in (("S03", "validation"), ("S04", "final")):
        entry = audit["sessions"][session]
        if entry["source_hdf5_sha256"] != study["source_hdf5_sha256"][session]:
            raise ValueError(f"{session} source provenance mismatch")
        for model, study_key, width, family_name in (
            ("F0", "F0", 48, "F0"),
            ("F0+SPD", "F0_plus_SPD", 84, "F0+F2c_SPD"),
        ):
            metrics = entry["models"][model]
            reference = study["source_zero_shot"][phase][study_key]
            for scored, frozen in (("replayed_accuracy", "accuracy"),
                                   ("replayed_macro_f1", "macro_f1"),
                                   ("replayed_log_loss", "log_loss"),
                                   ("replayed_brier", "brier")):
                if abs(metrics[scored] - reference[frozen]) > 1e-5:
                    raise ValueError(f"{session}/{model} differs from frozen source-only study")
            common = {"dataset": DATASET, "phase": phase, "subject": "Song",
                      "session/domain": session, "condition": CONDITION,
                      "calibration_budget": 0, "shots_per_class": 0,
                      "evaluation_unit": "whole_native_trial_mean",
                      "evaluation_trials": metrics["formal_trial_count"]}
            families.append({**common, "feature_family": family_name,
                             "model": "source_only_logistic", "feature_dimension": width,
                             "macro_f1": metrics["replayed_macro_f1"],
                             "accuracy": metrics["replayed_accuracy"],
                             "log_loss": metrics["replayed_log_loss"],
                             "brier": metrics["replayed_brier"],
                             "ece": metrics["top_label_ece_10_bins"],
                             "scope": "one-person/day exploratory, S01/S02 source, S03 validation, S04 final",
                             "evaluation_boundary": "cued stable trial means; source model fixed; S04 previously inspected; not live accuracy"})
            curve_common = {key: value for key, value in common.items()
                            if key not in ("calibration_budget", "evaluation_unit")}
            curves.append({**curve_common, "feature_bank": family_name,
                           "method": "source_only_zero_shot",
                           "macro_f1": metrics["replayed_macro_f1"],
                           "log_loss": metrics["replayed_log_loss"]})
        base = entry["models"]["F0"]
        candidate = entry["models"]["F0+SPD"]
        paired = entry["paired_F0_vs_SPD"]
        count = int(paired["trials"])
        if count != base["formal_trial_count"] or count != candidate["formal_trial_count"]:
            raise ValueError("Paired Song trial counts differ")
        increments.append({"dataset": DATASET, "phase": phase, "subject": "Song",
                           "session/domain": session, "condition": CONDITION,
                           "calibration_budget": 0, "core_bank": "F0",
                           "added_family": "F2c_SPD",
                           "delta_logloss": candidate["replayed_log_loss"] - base["replayed_log_loss"],
                           "delta_macro_f1": candidate["replayed_macro_f1"] - base["replayed_macro_f1"],
                           "delta_brier": candidate["replayed_brier"] - base["replayed_brier"],
                           "evaluation_unit": "whole_native_trial_mean", "evaluation_trials": count})
        complements.append({"dataset": DATASET, "phase": phase, "subject": "Song",
                            "session/domain": session, "condition": CONDITION,
                            "family_a": "F0", "family_b": "F0+F2c_SPD",
                            "error_correlation": paired["error_correlation"],
                            "disagreement_rate": paired["prediction_disagreement_rate"],
                            "a_correct_b_wrong": paired["F0_correct_SPD_wrong"] / count,
                            "a_wrong_b_correct": paired["F0_wrong_SPD_correct"] / count,
                            "evaluation_unit": "whole_native_trial_mean", "evaluation_trials": count})
    # This is a separate curve: the zero-shot member is source F0, while
    # 1/2-shot members mix source F0 with calibration-only F7 SPD anchors.
    # It must not be presented as calibration of the source F0+SPD classifier.
    selected = study["personal_anchor"]["selected_weights"]
    for session, phase in (("S03", "validation"), ("S04", "final")):
        zero = study["source_zero_shot"][phase]["F0"]
        for budget in (0, 1, 2):
            if budget == 0:
                scored = zero
                weight = 0.0
                block_count = 0
            else:
                key = str(budget)
                weight = float(selected[key])
                block = study["personal_anchor"][phase][key]
                if block["audit"]["formal_trial_overlap"]:
                    raise ValueError("Personal SPD calibration overlaps formal evaluation")
                block_count = int(block["audit"]["calibration_blocks"])
                if block_count != budget * 4:
                    raise ValueError("Personal SPD shot count differs from four hand classes")
                if phase == "validation":
                    if float(block["selected_weight"]) != weight:
                        raise ValueError("S03 selected SPD weight changed")
                    scored = block["candidates"][str(weight)]
                else:
                    if float(block["weight_fixed_on_S03"]) != weight:
                        raise ValueError("S04 SPD weight differs from S03 selection")
                    scored = block["F0_plus_SPD_anchor"]
            if int(scored["trials"]) != int(study["source_zero_shot"][phase]["F0"]["trials"]):
                raise ValueError("Personal SPD evaluation trial count changed")
            curves.append({"dataset": DATASET, "phase": phase, "subject": "Song",
                           "session/domain": session, "condition": CONDITION,
                           "shots_per_class": budget,
                           "evaluation_trials": scored["trials"],
                           "feature_bank": "F0+F7_SPD_personal_anchor",
                           "method": "source_F0_plus_calibration_only_SPD_anchor",
                           "macro_f1": scored["macro_f1"],
                           "log_loss": scored["log_loss"]})
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "feature_family_results.csv", families)
    _write(output / "conditional_incremental.csv", increments)
    _write(output / "error_complementarity.csv", complements)
    _write(output / "calibration_curve.csv", curves)
    manifest = {
        "run_id": RUN_ID, "dataset": DATASET,
        "phase": "validation_and_final_exploratory",
        "source_sessions": ["S01", "S02"], "validation_session": "S03", "final_session": "S04",
        "participant_count": 1, "calendar_day_count": 1,
        "raw_data_redistributed": False,
        "source_hdf5_sha256": study["source_hdf5_sha256"],
        "model_sha256": {name: audit["sessions"]["S04"]["models"][name]["bundle_sha256"]
                         for name in ("F0", "F0+SPD")},
        "personal_anchor_selected_weights_on_S03": selected,
        "personal_anchor_max_blocks_per_class": 2,
        "reliability_audit_sha256": _sha(calibration_audit),
        "spd_increment_study_sha256": _sha(spd_study),
        "boundary": "Source-trained zero-shot 4-state F0/F0+SPD candidates, paired comparisons and a separate source-F0-plus-personal-SPD-anchor 0/1/2-shot curve. Only two calibration blocks per class exist; 5-shot is unsupported. S01-S03 whole sessions failed collection readiness; S04 was previously inspected. No full F0-F9 bank, cross-person/day or physical live accuracy claim.",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "family_rows": len(families),
                      "increment_rows": len(increments), "pair_rows": len(complements),
                      "curve_rows": len(curves)}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-audit", required=True, type=Path)
    parser.add_argument("--spd-study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.calibration_audit, args.spd_study, args.output)
