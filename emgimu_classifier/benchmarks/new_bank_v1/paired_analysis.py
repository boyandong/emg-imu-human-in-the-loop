"""Matched new-v1 family screen, conditional value, errors and interaction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.family_screen import _score

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "PAIRED_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / "TRIAL_PREDICTIONS.csv"
DATASET_NAMES = {"force": "libemg_contraction_intensity",
                 "wearing": "libemg_electrode_shift",
                 "manus": "semg_manus",
                 "grab_user": "grabmyo_forearm8",
                 "roam_posture": "roam_emg_static"}


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def analyze_group(dataset: str, phase: str, scope: str, subject: str,
                  condition: str, by_arm: dict[str, list[dict]], classes: list[str],
                  names: tuple[str, str, str, str]) -> tuple[list[dict], list[dict], dict, dict, dict]:
    indexed = {}
    for arm, rows in by_arm.items():
        index = {}
        for row in rows:
            key = (row["subject"], row["condition"], row["trial_id"])
            if key in index:
                raise AssertionError("duplicate native trial")
            index[key] = row
        indexed[arm] = index
    keys = sorted(indexed[names[0]])
    if not keys or any(set(index) != set(keys) for index in indexed.values()):
        raise AssertionError(f"native trial mismatch: {dataset}/{phase}/{scope}")
    labels = np.asarray([indexed[names[0]][key]["label"] for key in keys])
    calculated = {}
    common = {"dataset": DATASET_NAMES[dataset], "axis": dataset, "phase": phase,
              "scope": scope, "subject": subject, "condition": condition,
              "calibration_budget": 0, "evaluation_unit": "whole_native_trial",
              "evaluation_trials": len(keys)}
    screen = []
    for arm in names:
        ordered = [indexed[arm][key] for key in keys]
        if not np.array_equal(labels, [row["label"] for row in ordered]):
            raise AssertionError("matched trial has conflicting labels")
        values = _score(ordered, classes)
        probability = np.asarray([[float(row[f"p_{c}"]) for c in classes] for row in ordered])
        predicted = np.asarray(classes)[probability.argmax(axis=1)]
        calculated[arm] = {"score": values, "correct": predicted == labels, "predicted": predicted}
        screen.append({**common, "feature_family": arm,
                       **{name: values[name] for name in ("macro_f1", "accuracy", "log_loss",
                                                           "brier", "ece", "per_class_f1_json")}})
    base, aa, bb, joint = names
    conditional = []
    for core, added in ((bb, aa), (aa, bb)):
        old = calculated[core]["score"]
        new = calculated[joint]["score"]
        conditional.append({**common, "core_bank": core, "added_family": added.removeprefix(base + "+"),
                            "increment_bank": joint,
                            "delta_log_loss": old["log_loss"] - new["log_loss"],
                            "delta_brier": old["brier"] - new["brier"],
                            "delta_macro_f1": new["macro_f1"] - old["macro_f1"],
                            "core_per_class_f1_json": old["per_class_f1_json"],
                            "increment_per_class_f1_json": new["per_class_f1_json"]})
    a, b = calculated[aa], calculated[bb]
    ea, eb = (~a["correct"]).astype(int), (~b["correct"]).astype(int)
    corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else "N/A"
    acbw = int(np.sum(a["correct"] & ~b["correct"]))
    awbc = int(np.sum(~a["correct"] & b["correct"]))
    both_correct = int(np.sum(a["correct"] & b["correct"]))
    both_wrong = int(np.sum(~a["correct"] & ~b["correct"]))
    if acbw + awbc + both_correct + both_wrong != len(keys):
        raise AssertionError("correctness contingency does not sum to trial count")
    errors = {**common, "family_a": aa, "family_b": bb,
              "comparison_type": "matched_new_v1_single_additions",
              "error_correlation": corr,
              "prediction_disagreement_rate": float(np.mean(a["predicted"] != b["predicted"])),
              "a_correct_b_wrong": acbw, "a_wrong_b_correct": awbc,
              "both_correct": both_correct, "both_wrong": both_wrong}
    s0, sa, sb, sj = (calculated[arm]["score"] for arm in names)
    interaction = {**common, "base_arm": base, "family_a_arm": aa,
                   "family_b_arm": bb, "joint_arm": joint,
                   "interaction_neg_log_loss": -sj["log_loss"] + sa["log_loss"] +
                   sb["log_loss"] - s0["log_loss"],
                   "interaction_macro_f1": sj["macro_f1"] - sa["macro_f1"] -
                   sb["macro_f1"] + s0["macro_f1"],
                   "joint_minus_core_macro_f1": sj["macro_f1"] - s0["macro_f1"],
                   "joint_minus_core_neg_log_loss": s0["log_loss"] - sj["log_loss"]}
    return screen, conditional, errors, interaction, calculated


def build(verify: bool = False) -> dict:
    for name, field in (("PROTOCOL.json", "parent_protocol_sha256"),
                        ("TRIAL_PREDICTIONS.csv", "parent_prediction_sha256"),
                        ("RESULTS.json", "parent_result_sha256")):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != PROTOCOL[field]:
            raise AssertionError(f"frozen parent changed: {name}")
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != 5664:
        raise AssertionError("expected 5,664 frozen arm–trial predictions")
    parent = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    screen_rows, conditional_rows, error_rows, interaction_rows = [], [], [], []
    replayed = 0
    cell_counts = {}
    for dataset, specification in PROTOCOL["comparisons"].items():
        base = specification["baseline"]
        family_a, family_b = specification["family_a"], specification["family_b"]
        names = (base, base + "+" + family_a, base + "+" + family_b,
                 base + "+" + family_a + "+" + family_b)
        classes = [str(value) for value in specification["classes"]]
        before = len(interaction_rows)
        for phase in ("validation", "final"):
            phase_rows = [row for row in raw if row["dataset"] == dataset and row["phase"] == phase]
            base_rows = [row for row in phase_rows if row["arm"] == base]
            subjects = sorted({row["subject"] for row in base_rows}, key=int)
            conditions = sorted({row["condition"] for row in base_rows})
            groups = [("pooled", "ALL", "ALL")]
            groups += [("subject", subject, "ALL") for subject in subjects]
            groups += [("condition", "ALL", condition) for condition in conditions]
            for scope, subject, condition in groups:
                by_arm = {arm: [row for row in phase_rows if row["arm"] == arm and
                                (scope != "subject" or row["subject"] == subject) and
                                (scope != "condition" or row["condition"] == condition)]
                          for arm in names}
                screen, conditional, errors, interaction, computed = analyze_group(
                    dataset, phase, scope, subject, condition, by_arm, classes, names)
                screen_rows.extend(screen)
                conditional_rows.extend(conditional)
                error_rows.append(errors)
                interaction_rows.append(interaction)
                if scope == "pooled":
                    for arm in names:
                        saved = parent["results"][dataset][phase]["arms"][arm]["pooled"]
                        value = computed[arm]["score"]
                        for metric in ("trials", "macro_f1", "accuracy", "log_loss", "brier"):
                            if not np.isclose(value[metric], saved[metric], rtol=0, atol=1e-10):
                                raise AssertionError(f"parent pooled metric changed: {dataset}/{phase}/{arm}/{metric}")
                        replayed += 1
        cell_counts[dataset] = len(interaction_rows) - before
    if cell_counts != {"force": 28, "wearing": 16} or replayed != 16:
        raise AssertionError("unexpected matched study cells or pooled metric replays")
    outputs = {"PAIRED_SCREEN.csv": screen_rows, "PAIRED_CONDITIONAL.csv": conditional_rows,
               "PAIRED_COMPLEMENTARITY.csv": error_rows, "PAIRED_INTERACTION.csv": interaction_rows}
    if {name: len(rows) for name, rows in outputs.items()} != {
            "PAIRED_SCREEN.csv": 176, "PAIRED_CONDITIONAL.csv": 88,
            "PAIRED_COMPLEMENTARITY.csv": 44, "PAIRED_INTERACTION.csv": 44}:
        raise AssertionError("unexpected new-v1 analysis row count")
    for name, rows in outputs.items():
        content = csv_text(rows)
        path = ROOT / name
        if verify:
            if path.read_bytes() != content.encode("utf-8"):
                raise AssertionError(f"derived table changed: {name}")
        else:
            path.write_bytes(content.encode("utf-8"))
    audit = {"status": "ok", "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
             "source_predictions_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "saved_pooled_metric_groups_replayed": replayed, "matched_cells_by_axis": cell_counts,
             "rows": {name: len(rows) for name, rows in outputs.items()},
             "boundary": PROTOCOL["boundary"]}
    path = ROOT / "PAIRED_AUDIT.json"
    content = json.dumps(audit, indent=2) + "\n"
    if verify:
        if path.read_text(encoding="utf-8") != content:
            raise AssertionError("paired analysis audit changed")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "ok", "verify": verify, "rows": audit["rows"]}))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    build(parser.parse_args().verify)
