from __future__ import annotations

import argparse
import csv
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

from emgimu.datasets.libemg_force import load_libemg_force_windows

from .screening import FAMILY_FACTORIES, fit_predict, metrics


CANDIDATES = ("F2b_CSP", "F9_Quality", "F1_X1H", "F4_Spectral")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output: Path) -> None:
    print("[1/4] loading frozen development split", flush=True)
    train = load_libemg_force_windows(root, subjects=range(1, 7), conditions=("Ramp",))
    validation = load_libemg_force_windows(
        root, subjects=(7, 8),
        conditions=("20P", "30P", "40P", "50P", "60P", "70P", "80P", "MVC", "Light", "Medium", "Hard"),
    )
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    states = {}
    for index, name in enumerate(("F0", *CANDIDATES), 1):
        print(f"[2/4] feature {index}/5 {name}", flush=True)
        family = FAMILY_FACTORIES[name]()
        features[name] = (family.fit_transform(train.batch, train.labels), family.transform(validation.batch))
        states[name] = family

    specs: list[tuple[str, tuple[str, ...]]] = [("B_F0", ("F0",))]
    specs += [(f"single_{item}", ("F0", item)) for item in CANDIDATES]
    specs += [(f"pair_{a}_{b}", ("F0", a, b)) for a, b in combinations(CANDIDATES, 2)]
    full = ("F0", *CANDIDATES)
    specs.append(("full_shortlist", full))
    specs += [(f"leave_out_{item}", tuple(name for name in full if name != item)) for item in full]

    predictions: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for index, (name, members) in enumerate(specs, 1):
        print(f"[3/4] model {index}/{len(specs)} {name}", flush=True)
        train_x = np.concatenate([features[item][0] for item in members], axis=1)
        validation_x = np.concatenate([features[item][1] for item in members], axis=1)
        probability, seconds = fit_predict(train_x, train, validation_x)
        predictions[name] = probability
        values = metrics(validation.labels, probability, validation.sample_weight)
        rows.append({"model": name, "families": "+".join(members), **values, "training_seconds": seconds,
                     "feature_dimension":train_x.shape[1]})

    baseline = next(row for row in rows if row["model"] == "B_F0")
    singles = {row["model"].removeprefix("single_"): row for row in rows if row["model"].startswith("single_")}
    interactions = []
    for row in rows:
        if not row["model"].startswith("pair_"):
            continue
        a, b = [item for item in row["families"].split("+") if item != "F0"]
        expected = float(singles[a]["macro_f1"]) + float(singles[b]["macro_f1"]) - float(baseline["macro_f1"])
        interactions.append({
            "family_a": a, "family_b": b, "macro_f1": row["macro_f1"],
            "interaction_macro_f1": float(row["macro_f1"]) - expected,
            "log_loss": row["log_loss"], "brier": row["brier"], "ece": row["ece"],
        })

    full_row = next(row for row in rows if row["model"] == "full_shortlist")
    ablation=[]
    for condition in ['ALL',*sorted(np.unique(validation.conditions))]:
        selected=np.ones(len(validation.labels),dtype=bool) if condition=='ALL' else validation.conditions==condition
        reference=metrics(validation.labels[selected],predictions['full_shortlist'][selected],validation.sample_weight[selected])
        for row in rows:
            if row['model']!='full_shortlist' and not row['model'].startswith('leave_out_'):continue
            values=metrics(validation.labels[selected],predictions[row['model']][selected],validation.sample_weight[selected])
            ablation.append({'dataset':'libemg_contraction_intensity','condition':condition,
                'removed_family':'NONE' if row['model']=='full_shortlist' else row['model'].removeprefix('leave_out_'),
                **values,'delta_macro_f1_vs_full':values['macro_f1']-reference['macro_f1'],
                'feature_dimension':row['feature_dimension']})

    print("[4/4] writing selection evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'heldout_predictions.npz',**predictions,labels=validation.labels,
                        trials=validation.trials,weights=validation.sample_weight,conditions=validation.conditions)
    with (output/'family_states.pkl').open('wb') as handle:pickle.dump(states,handle)
    (output/'split_trial_ids.json').write_text(json.dumps({'train':sorted(set(train.trials.tolist())),
        'validation':sorted(set(validation.trials.tolist()))},indent=2),encoding='utf-8')
    _write(output / "selection_models.csv", rows)
    _write(output / "interaction_screen.csv", interactions)
    _write(output / "ablation_full_bank.csv", ablation)
    (output / "shortlist.json").write_text(json.dumps({
        "selection_only": True, "validation_subjects": [7, 8], "test_subjects_unopened": [9, 10],
        "candidates": list(CANDIDATES), "full_families": list(full),
        "selected_model": max(rows, key=lambda row: float(row["macro_f1"]))["model"],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "models": len(specs), "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair interaction and leave-family-out development study")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
