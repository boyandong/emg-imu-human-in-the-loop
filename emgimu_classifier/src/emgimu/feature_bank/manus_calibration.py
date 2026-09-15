from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.semg_manus import load_semg_manus_windows

from .manus_study import GESTURES, _aggregate, _metrics
from .screening import FAMILY_FACTORIES, SEED


FROZEN_FAMILIES = ("F0", "F2c_SPD")


def run(archive: Path, output: Path, phase: str) -> None:
    users = (3, 4, 5, 6, 7, 8); target_session = 2 if phase == "validation" else 3
    print(f"[1/5] loading sEMG-MANUS session calibration {phase}", flush=True)
    train = load_semg_manus_windows(archive, users=users, sessions=(1,), gestures=GESTURES)
    target = load_semg_manus_windows(archive, users=users, sessions=(target_session,), gestures=GESTURES)
    train_parts, target_parts, dimensions = [], [], {}
    train_y = target_y = target_users = target_trials = None
    for index, name in enumerate(FROZEN_FAMILIES, 1):
        print(f"[2/5] feature {index}/2 {name}", flush=True)
        family = FAMILY_FACTORIES[name](); a = family.fit_transform(train.batch, train.labels); b = family.transform(target.batch)
        a, train_y, _, _, _, _ = _aggregate(a, train); b, target_y, target_users, _, _, target_trials = _aggregate(b, target)
        train_parts.append(a); target_parts.append(b); dimensions[name] = a.shape[1]
    train_x, target_x = np.concatenate(train_parts, axis=1), np.concatenate(target_parts, axis=1)
    scaler = StandardScaler().fit(train_x); a = scaler.transform(train_x); b = scaler.transform(target_x)
    base_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(a, train_y)
    base_probability = base_model.predict_proba(b)
    rng = np.random.default_rng(SEED); rows = []; selected_rows = []
    for index, (user, shots) in enumerate(((u, k) for u in users for k in (0, 1, 2, 5)), 1):
        print(f"[3/5] user/session {index}/{len(users)*4} user{user} {shots}-shot", flush=True)
        user_mask = target_users == user; chosen = []
        if shots < 3 and shots:
            for label in range(len(GESTURES)):
                trials = np.unique(target_trials[user_mask & (target_y == label)])
                chosen.extend(rng.permutation(trials)[:shots].tolist())
        if shots >= 3:
            rows.append({"phase": phase, "dataset": "semg_manus", "subject": user, "condition": f"session_{target_session}",
                "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES), "mode": "unsupported",
                "reason": "only 3 speed trials/class and one must remain for evaluation", "evaluation_trials": 0,
                "macro_f1": "", "accuracy": "", "log_loss": "", "brier": "", "ece": "", "per_class_f1_json": ""})
            continue
        calibration = user_mask & np.isin(target_trials, chosen); evaluation = user_mask & ~calibration
        if shots:
            augmented_x = np.concatenate((a, b[calibration])); augmented_y = np.concatenate((train_y, target_y[calibration]))
            weights = np.concatenate((np.ones(len(train_y)), np.full(calibration.sum(), 6.0 / shots)))
            probability = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED).fit(
                augmented_x, augmented_y, sample_weight=weights).predict_proba(b[evaluation])
        else:
            probability = base_probability[evaluation]
        rows.append({"phase": phase, "dataset": "semg_manus", "subject": user, "condition": f"session_{target_session}",
            "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
            "mode": "population_zero_shot" if shots == 0 else "session_weighted_finetune", "reason": "",
            "evaluation_trials": int(evaluation.sum()), **_metrics(target_y[evaluation], probability)})
        selected_rows.extend({"phase": phase, "subject": user, "shots_per_class": shots, "trial_id": trial} for trial in chosen)
    print("[4/5] aggregating supported budgets", flush=True)
    for shots in (0, 1, 2):
        selected = [row for row in rows if row["shots_per_class"] == shots and row["mode"] != "unsupported"]
        numeric = ("macro_f1", "accuracy", "log_loss", "brier", "ece")
        rows.append({"phase": phase, "dataset": "semg_manus", "subject": "ALL", "condition": f"session_{target_session}",
            "shots_per_class": shots, "feature_bank": "+".join(FROZEN_FAMILIES),
            "mode": "population_zero_shot" if shots == 0 else "session_weighted_finetune", "reason": "",
            "evaluation_trials": sum(int(row["evaluation_trials"]) for row in selected),
            **{key: float(np.mean([float(row[key]) for row in selected])) for key in numeric}, "per_class_f1_json": "per-user macro aggregation"})
    rows.append({"phase": phase, "dataset": "semg_manus", "subject": "ALL", "condition": f"session_{target_session}",
        "shots_per_class": 5, "feature_bank": "+".join(FROZEN_FAMILIES), "mode": "unsupported",
        "reason": "only 3 speed trials/class and one must remain for evaluation", "evaluation_trials": 0,
        "macro_f1": "", "accuracy": "", "log_loss": "", "brier": "", "ece": "", "per_class_f1_json": ""})
    print("[5/5] writing session calibration evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "calibration_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "calibration_trial_ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "subject", "shots_per_class", "trial_id"]); writer.writeheader(); writer.writerows(selected_rows)
    (output / "run_manifest.json").write_text(json.dumps({"phase": phase, "seed": SEED, "users": list(users), "train_session": 1,
        "target_session": target_session, "families": list(FROZEN_FAMILIES), "dimensions": dimensions,
        "finetune": {"C": 0.1, "calibration_sample_weight": "6/shots"}, "trial_disjoint": True}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "phase": phase, "rows": len(rows), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="sEMG-MANUS 0/1/2-shot session calibration")
    parser.add_argument("archive", type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    args = parser.parse_args(); run(args.archive, args.output, args.phase)


if __name__ == "__main__": main()
