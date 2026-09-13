import csv
import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "tools" / "aggregate_hla_screens.py"
    spec = importlib.util.spec_from_file_location("hla_screen_aggregate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _screen(root: Path, seed: int, values: dict[tuple[str, str], float]) -> None:
    subjects = sorted({subject for _, subject in values})
    root.mkdir()
    (root / "screen_summary.json").write_text(json.dumps({
        "status": "complete", "seed": seed, "sensor_view": "ring", "targets": subjects,
    }))
    for (representation, subject), score in values.items():
        run = root / f"{representation}_{subject}_s{seed}"
        run.mkdir()
        (run / "run_manifest.json").write_text(json.dumps({
            "representation": representation, "target_subject": subject,
            "source_commit": "abc", "metrics": {"active_macro_f1": score},
            "per_class": [{"recall": score}, {"recall": score / 2}],
        }))
        with (run / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "trial_id", "start_timestamp_ms", "actual_label", "predicted_label", "confidence",
            ))
            writer.writeheader()
            writer.writerow({
                "trial_id": f"{subject}-t", "start_timestamp_ms": "0.0", "actual_label": "1",
                "predicted_label": "1", "confidence": "0.9",
            })


def test_aggregate_averages_seeds_before_subject_bootstrap(tmp_path: Path) -> None:
    module = _load_module()
    roots = []
    for seed, offset in ((42, 0.0), (43, 0.1)):
        root = tmp_path / f"s{seed}"
        _screen(root, seed, {
            ("R1-core", "s1"): 0.5 + offset,
            ("R3", "s1"): 0.6 + offset,
            ("R1-core", "s2"): 0.7 + offset,
            ("R3", "s2"): 0.75 + offset,
        })
        roots.append(root)
    result = module.aggregate(roots, baseline="R1-core", challenger="R3", bootstrap_iterations=100)
    paired = result["paired_challenger_minus_baseline"]
    assert paired["subject_count"] == 2
    assert abs(paired["mean"] - 0.075) < 1e-9
    assert result["evaluation_pairing_verified"] is True
