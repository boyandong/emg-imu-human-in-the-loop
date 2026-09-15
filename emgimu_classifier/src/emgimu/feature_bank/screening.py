from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from emgimu.datasets.libemg_force import ForceWindows, load_libemg_force_windows

from .families import (
    CspSpatialFamily, LocalDetailFamily, QualityFamily, RingGeometryFamily,
    ScalePatternFamily, SpectralStateFamily, SpdTangentFamily,
    TemporalFormFamily, TraceCovarianceFamily,
)


SEED = 20260915
CLASSES = np.arange(7)
FAMILY_FACTORIES = {
    "F0": LocalDetailFamily,
    "F1_X1H": ScalePatternFamily,
    "F2a_Cov": TraceCovarianceFamily,
    "F2b_CSP": CspSpatialFamily,
    "F2c_SPD": SpdTangentFamily,
    "F3_Ring": RingGeometryFamily,
    "F4_Spectral": SpectralStateFamily,
    "F5_Temporal": TemporalFormFamily,
    "F9_Quality": QualityFamily,
}


def expected_calibration_error(truth: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if np.any(selected):
            bin_weight = float(weights[selected].sum() / weights.sum())
            result += bin_weight * abs(float(np.average(correct[selected], weights=weights[selected])) - float(np.average(confidence[selected], weights=weights[selected])))
    return float(result)


def metrics(truth: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[truth]
    per_class = f1_score(truth, prediction, labels=CLASSES, average=None, sample_weight=weights, zero_division=0)
    return {
        "macro_f1": float(f1_score(truth, prediction, labels=CLASSES, average="macro", sample_weight=weights, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction, sample_weight=weights)),
        "log_loss": float(log_loss(truth, probabilities, labels=CLASSES, sample_weight=weights)),
        "brier": float(np.average(np.mean((probabilities - one_hot) ** 2, axis=1), weights=weights)),
        "ece": expected_calibration_error(truth, probabilities, weights),
        "per_class_f1_json": json.dumps({str(index): float(value) for index, value in enumerate(per_class)}, separators=(",", ":")),
    }


def fit_predict(train_x: np.ndarray, train: ForceWindows, validation_x: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    scaler = StandardScaler().fit(train_x, sample_weight=train.sample_weight)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    model.fit(scaler.transform(train_x), train.labels, sample_weight=train.sample_weight)
    return model.predict_proba(scaler.transform(validation_x)), time.perf_counter() - started


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output: Path) -> None:
    print("[1/4] loading frozen train and validation trials", flush=True)
    train = load_libemg_force_windows(root, subjects=range(1, 7), conditions=("Ramp",))
    validation = load_libemg_force_windows(root, subjects=(7, 8), conditions=("20P", "30P", "40P", "50P", "60P", "70P", "80P", "MVC", "Light", "Medium", "Hard"))
    factories = FAMILY_FACTORIES
    train_features: dict[str, np.ndarray] = {}
    validation_features: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    fitted_families = {}
    diagnostics = []
    for index, (name, factory) in enumerate(factories.items(), 1):
        print(f"[2/4] feature {index}/{len(factories)} {name}", flush=True)
        family = factory()
        train_features[name] = family.fit_transform(train.batch, train.labels)
        validation_features[name] = family.transform(validation.batch)
        dimensions[name] = len(family.feature_names)
        fitted_families[name] = family
        # Standardization is fit on source trials, never on target variance.
        diagnostic_scaler = StandardScaler().fit(train_features[name], sample_weight=train.sample_weight)
        target_z = diagnostic_scaler.transform(validation_features[name])
        nuisance, separation = [], []
        for subject in np.unique(validation.subjects):
            cells = {}
            for condition in np.unique(validation.conditions):
                for label in CLASSES:
                    mask = (validation.subjects == subject) & (validation.conditions == condition) & (validation.labels == label)
                    if np.any(mask):
                        cells[(condition, label)] = target_z[mask].mean(0)
            conditions = sorted(np.unique(validation.conditions))
            for label in CLASSES:
                centers = [cells[(c, label)] for c in conditions if (c, label) in cells]
                nuisance.extend(np.linalg.norm(a-b) for i,a in enumerate(centers) for b in centers[i+1:])
            for condition in conditions:
                centers = [cells[(condition, label)] for label in CLASSES if (condition, label) in cells]
                separation.extend(np.linalg.norm(a-b) for i,a in enumerate(centers) for b in centers[i+1:])
        dn, dg = float(np.mean(nuisance)), float(np.mean(separation))
        diagnostics.append({'family':name,'dimension':dimensions[name], 'D_nuisance':dn,
                            'D_gesture':dg,'J':dg/(dn+1e-12), 'definition':'within-user class-matched condition centroid Euclidean distance; train-standardized'})

    models = {"B_F0": ("F0",)}
    models.update({f"B_plus_{name}": ("F0", name) for name in factories if name != "F0"})
    predictions: dict[str, np.ndarray] = {}
    training_seconds: dict[str, float] = {}
    for index, (model_name, members) in enumerate(models.items(), 1):
        print(f"[3/4] classifier {index}/{len(models)} {model_name}", flush=True)
        train_x = np.concatenate([train_features[item] for item in members], axis=1)
        validation_x = np.concatenate([validation_features[item] for item in members], axis=1)
        predictions[model_name], training_seconds[model_name] = fit_predict(train_x, train, validation_x)

    result_rows: list[dict] = []
    for model_name, probability in predictions.items():
        members = models[model_name]
        for subject in ["ALL", *sorted(np.unique(validation.subjects))]:
            for condition in ["ALL", *sorted(np.unique(validation.conditions))]:
                if subject == "ALL" and condition == "ALL":
                    selected = np.ones(validation.batch.windows, dtype=bool)
                elif subject == "ALL":
                    selected = validation.conditions == condition
                elif condition == "ALL":
                    selected = validation.subjects == subject
                else:
                    selected = (validation.subjects == subject) & (validation.conditions == condition)
                values = metrics(validation.labels[selected], probability[selected], validation.sample_weight[selected])
                result_rows.append({
                    "dataset": "libemg_contraction_intensity", "subject": subject,
                    "session/domain": "fixed_intensity", "feature_family": "+".join(members),
                    "calibration_budget": 0, "condition": condition, **values,
                    "feature_dimension": sum(dimensions[item] for item in members),
                    "training_seconds": training_seconds[model_name],
                })

    baseline = predictions["B_F0"]
    baseline_metrics = metrics(validation.labels, baseline, validation.sample_weight)
    incremental = []
    for name in factories:
        if name == "F0":
            continue
        for condition in ["ALL", *sorted(np.unique(validation.conditions))]:
            selected = np.ones(validation.batch.windows, dtype=bool) if condition == "ALL" else validation.conditions == condition
            base_cell = metrics(validation.labels[selected], baseline[selected], validation.sample_weight[selected])
            candidate = metrics(validation.labels[selected], predictions[f"B_plus_{name}"][selected], validation.sample_weight[selected])
            incremental.append({
                "core_bank": "F0", "added_family": name, "condition": condition,
                "delta_logloss": base_cell["log_loss"] - candidate["log_loss"],
                "delta_macro_f1": candidate["macro_f1"] - base_cell["macro_f1"],
                "delta_brier": base_cell["brier"] - candidate["brier"],
            })

    complementarity = []
    names = list(predictions)
    truth = validation.labels
    for index, first in enumerate(names):
        error_a = predictions[first].argmax(axis=1) != truth
        for second in names[index + 1:]:
            error_b = predictions[second].argmax(axis=1) != truth
            weight = validation.sample_weight / validation.sample_weight.sum()
            mean_a, mean_b = np.sum(weight * error_a), np.sum(weight * error_b)
            covariance = np.sum(weight * (error_a - mean_a) * (error_b - mean_b))
            variance_a = np.sum(weight * (error_a - mean_a) ** 2)
            variance_b = np.sum(weight * (error_b - mean_b) ** 2)
            if variance_a <= 0 or variance_b <= 0:
                correlation = 0.0
            else:
                correlation = float(covariance / np.sqrt(variance_a * variance_b))
            complementarity.append({
                "family_a": first, "family_b": second,
                "error_correlation": correlation,
                "disagreement": float(np.sum(weight * (error_a != error_b))),
                "a_correct_b_wrong": float(np.sum(weight * (~error_a & error_b))),
                "a_wrong_b_correct": float(np.sum(weight * (error_a & ~error_b))),
            })

    print("[4/4] writing validation-only evidence", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'family_states.pkl').open('wb') as handle:
        pickle.dump(fitted_families, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(output / 'heldout_predictions.npz', **predictions,
                        labels=validation.labels, weights=validation.sample_weight,
                        trials=validation.trials, subjects=validation.subjects, conditions=validation.conditions)
    (output / 'split_trial_ids.json').write_text(json.dumps({
        'train':sorted(set(train.trials.tolist())), 'validation':sorted(set(validation.trials.tolist()))},indent=2),encoding='utf-8')
    write_csv(output / 'family_diagnostics.csv', diagnostics, list(diagnostics[0]))
    write_csv(output / "feature_family_results.csv", result_rows, list(result_rows[0]))
    write_csv(output / "conditional_incremental.csv", incremental, list(incremental[0]))
    write_csv(output / "error_complementarity.csv", complementarity, list(complementarity[0]))
    write_csv(output / "calibration_curve.csv", [], ["dataset", "condition", "shots_per_class", "feature_bank", "macro_f1", "log_loss"])
    write_csv(output / "ablation_full_bank.csv", [], ["dataset", "condition", "removed_family", "macro_f1", "log_loss", "brier"])
    (output / "run_manifest.json").write_text(json.dumps({
        "run_id": output.name, "seed": SEED, "train_subjects": list(range(1, 7)),
        "validation_subjects": [7, 8], "test_subjects_unopened": [9, 10],
        "train_conditions": ["Ramp"], "validation_conditions": sorted(set(validation.conditions)),
        "window_ms": 200, "hop_ms": 200, "maximum_windows_per_trial": 8,
        "models": {name: {"families": members, "dimension": sum(dimensions[item] for item in members)} for name, members in models.items()},
        "baseline_validation": baseline_metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "train_windows": train.batch.windows, "validation_windows": validation.batch.windows, "models": len(models), "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe Feature Bank screening on LibEMG force")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)


if __name__ == "__main__":
    main()
