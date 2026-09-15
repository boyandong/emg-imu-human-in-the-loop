from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.electrode_shift import CLASS_NAMES, ShiftWindows, load_electrode_shift_windows

from .screening import FAMILY_FACTORIES, SEED, expected_calibration_error


def _aggregate(features: np.ndarray, data: ShiftWindows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials = np.unique(data.trials); output = []; labels = []; domains = []
    for trial in trials:
        selected = data.trials == trial; output.append(features[selected].mean(axis=0))
        labels.append(np.unique(data.labels[selected]).item()); domains.append(np.unique(data.domains[selected]).item())
    return np.stack(output), np.asarray(labels), np.asarray(domains), trials


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(1); one_hot = np.eye(5)[y]; weights = np.ones(len(y))
    per_class = f1_score(y, prediction, labels=np.arange(5), average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y, prediction, labels=np.arange(5), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)), "log_loss": float(log_loss(y, probability, labels=np.arange(5))),
        "brier": float(np.mean((probability-one_hot)**2)), "ece": expected_calibration_error(y, probability, weights),
        "per_class_f1_json": json.dumps({name: float(value) for name, value in zip(CLASS_NAMES, per_class)}, separators=(",", ":"))}


def run(archive: Path, output: Path, phase: str) -> None:
    subjects = (15, 16, 17) if phase == "validation" else (18, 19, 20)
    active = FAMILY_FACTORIES if phase == "validation" else {"F0": FAMILY_FACTORIES["F0"], "F3_Ring": FAMILY_FACTORIES["F3_Ring"]}
    specs = {"B_F0": ("F0",), **{f"B_plus_{name}": ("F0", name) for name in active if name != "F0"}}
    collected = {name: {"y": [], "p": [], "domain": [], "subject": []} for name in specs}
    print(f"[1/4] loading Electrode Shift {phase} subjects {subjects}", flush=True)
    for subject_index, subject in enumerate(subjects, 1):
        train = load_electrode_shift_windows(archive, subjects=(subject,), domains=("training",))
        target = load_electrode_shift_windows(archive, subjects=(subject,), domains=("trial_1", "trial_2", "trial_3", "trial_4"))
        train_features, target_features = {}, {}
        for feature_index, (name, factory) in enumerate(active.items(), 1):
            print(f"[2/4] subject {subject_index}/{len(subjects)} feature {feature_index}/{len(active)} {name}", flush=True)
            family = factory(); a = family.fit_transform(train.batch, train.labels); b = family.transform(target.batch)
            train_features[name], train_y, _, _ = _aggregate(a, train)
            target_features[name], target_y, target_domain, _ = _aggregate(b, target)
        for model_index, (model_name, members) in enumerate(specs.items(), 1):
            print(f"[3/4] subject {subject_index}/{len(subjects)} model {model_index}/{len(specs)} {model_name}", flush=True)
            a = np.concatenate([train_features[x] for x in members], axis=1); b = np.concatenate([target_features[x] for x in members], axis=1)
            scaler = StandardScaler().fit(a); model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
            probability = model.fit(scaler.transform(a), train_y).predict_proba(scaler.transform(b))
            collected[model_name]["y"].append(target_y); collected[model_name]["p"].append(probability)
            collected[model_name]["domain"].append(target_domain); collected[model_name]["subject"].append(np.full(len(target_y), subject))
    rows = []
    for model_name, members in specs.items():
        y = np.concatenate(collected[model_name]["y"]); probability = np.concatenate(collected[model_name]["p"])
        domains = np.concatenate(collected[model_name]["domain"]); subject_values = np.concatenate(collected[model_name]["subject"])
        for subject in ("ALL", *subjects):
            for domain in ("ALL", "trial_1", "trial_2", "trial_3", "trial_4"):
                selected = np.ones(len(y), dtype=bool)
                if subject != "ALL": selected &= subject_values == subject
                if domain != "ALL": selected &= domains == domain
                rows.append({"phase": phase, "dataset": "libemg_electrode_shift", "subject": subject, "condition": domain,
                    "feature_family": "+".join(members), "calibration_budget": 0, **_metrics(y[selected], probability[selected])})
    print("[4/4] writing electrode-shift evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "feature_family_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "run_manifest.json").write_text(json.dumps({"phase": phase, "seed": SEED, "subjects": list(subjects),
        "training_domain": "training", "target_domains": ["trial_1", "trial_2", "trial_3", "trial_4"],
        "evaluation_unit": "trial", "feature_candidates": list(active)}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "phase": phase, "subjects": len(subjects), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Electrode Shift Feature Bank study")
    parser.add_argument("archive", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    args = parser.parse_args(); run(args.archive, args.output, args.phase)


if __name__ == "__main__": main()
