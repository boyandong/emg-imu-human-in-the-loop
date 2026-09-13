import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch neural extra is not installed",
)


def test_active_macro_f1_uses_dataset_specific_neutral_label() -> None:
    from emgimu.datasets.hla_neural_training import _metrics

    actual = np.asarray([0, 16])
    predicted = np.asarray([16, 16])
    assert _metrics(actual, predicted, neutral_label=16)["active_macro_f1"] == 0.0
    assert _metrics(actual, predicted, neutral_label=0)["active_macro_f1"] == pytest.approx(2 / 3)


def test_neural_fold_has_three_disjoint_subject_roles(tmp_path: Path) -> None:
    from emgimu.datasets.hla_neural_training import (
        HLANeuralRunConfig, _representation_view, prepare_hla_screen_examples,
        run_neural_subject_fold,
    )
    from emgimu.datasets.hla_schema import (
        ChannelSpec, DatasetManifestV2, EMGTrial, OntologyEntry, OntologyRelation,
        write_hla_manifest, write_hla_trial,
    )
    from emgimu.state import Gesture

    dataset = tmp_path / "data"
    (dataset / "trials").mkdir(parents=True)
    manifest = DatasetManifestV2(
        dataset_id="tiny", dataset_version="1", adapter_version="1",
        source_url="https://example.invalid", license_id="test", native_sample_rate_hz=200,
        has_imu=False, channel_layouts={"two": (ChannelSpec("a"), ChannelSpec("b"))},
        ontology=(
            OntologyEntry("0", "rest", 0, OntologyRelation.EXACT, Gesture.NEUTRAL),
            OntologyEntry("1", "fist", 1, OntologyRelation.EXACT, Gesture.FIST),
        ),
    )
    write_hla_manifest(dataset / "manifest.json", manifest)
    rng = np.random.default_rng(2)
    for subject in range(4):
        for label in (0, 1):
            values = rng.normal(label * 2, 0.3, (120, 2))
            path = dataset / "trials" / f"s{subject}-g{label}.npz"
            write_hla_trial(path, EMGTrial(
                path=path, dataset_id="tiny", subject_id=f"s{subject}", session_id="d1",
                trial_id=f"s{subject}-g{label}", channel_layout_id="two", sample_rate_hz=200,
                timestamp_ms=np.arange(120) * 5.0, emg=values,
                source_label=np.full(120, str(label)), task_label=np.full(120, label),
                canonical_label=np.full(120, label), stable_mask=np.ones(120, dtype=bool),
            ))
    test_log = tmp_path / "pre-run-tests.txt"
    test_log.write_text("114 passed, 1 skipped\n", encoding="utf-8")
    config = HLANeuralRunConfig(
        representation="R3", sensor_view="two", target_subject="s0",
        validation_subject="s1", maximum_windows_per_trial=2, batch_size=4,
        maximum_epochs=2, patience=2, device="cpu",
        test_log_path=str(test_log),
    )
    prepared = prepare_hla_screen_examples(dataset, config)
    assert _representation_view(prepared[0], "R0").features.shape[-1] == 6
    assert _representation_view(prepared[0], "R2").features is None
    output = run_neural_subject_fold(dataset, tmp_path / "run", config, prepared=prepared)
    import json
    report = json.loads((output / "run_manifest.json").read_text())
    assert report["target_subject"] == "s0"
    assert report["validation_subject"] == "s1"
    assert set(report["train_subjects"]) == {"s2", "s3"}
    assert (output / "model.pt").is_file()
    assert (output / "test_log.txt").read_text(encoding="utf-8") == "114 passed, 1 skipped\n"
    assert report["pre_run_test_log_sha256"] is not None
    for name in (
        "metrics.csv", "per_class.csv", "calibration_curve.csv", "predictions.csv",
        "confusion_matrix.csv", "risk_coverage.csv", "model.sha256",
        "environment.json", "test_log.txt",
    ):
        assert (output / name).is_file()
