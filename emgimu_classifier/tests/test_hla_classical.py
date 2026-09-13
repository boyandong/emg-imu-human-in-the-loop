from pathlib import Path

import numpy as np

from emgimu.datasets.hla_classical import ClassicalRunConfig, run_loso_svm
from emgimu.datasets.hla_schema import (
    ChannelSpec, DatasetManifestV2, EMGTrial, OntologyEntry, OntologyRelation,
    write_hla_manifest, write_hla_trial,
)
from emgimu.state import Gesture


def test_loso_runner_emits_required_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "trials").mkdir(parents=True)
    manifest = DatasetManifestV2(
        dataset_id="synthetic", dataset_version="1", adapter_version="1",
        source_url="https://example.invalid", license_id="test", native_sample_rate_hz=200,
        has_imu=False, channel_layouts={"two": (ChannelSpec("a"), ChannelSpec("b"))},
        ontology=(
            OntologyEntry("0", "rest", 0, OntologyRelation.EXACT, Gesture.NEUTRAL),
            OntologyEntry("1", "fist", 1, OntologyRelation.EXACT, Gesture.FIST),
        ),
    )
    write_hla_manifest(dataset / "manifest.json", manifest)
    rng = np.random.default_rng(42)
    for subject_index in range(3):
        for label in (0, 1):
            n = 120
            emg = rng.normal(label * 3.0, 0.2, size=(n, 2))
            path = dataset / "trials" / f"s{subject_index}-g{label}.npz"
            write_hla_trial(path, EMGTrial(
                path=path, dataset_id="synthetic", subject_id=f"s{subject_index}",
                session_id="d1", trial_id=f"t{subject_index}-{label}", channel_layout_id="two",
                sample_rate_hz=200, timestamp_ms=np.arange(n) * 5.0, emg=emg,
                source_label=np.full(n, str(label)), task_label=np.full(n, label),
                canonical_label=np.full(n, label), stable_mask=np.ones(n, dtype=bool),
            ))
    output = run_loso_svm(dataset, tmp_path / "run", ClassicalRunConfig(maximum_windows_per_trial=3))
    for name in (
        "run_manifest.json", "config.json", "metrics.csv", "per_class.csv",
        "calibration_curve.csv", "predictions.csv", "confusion_matrix.csv",
        "risk_coverage.csv", "model.sha256", "environment.json", "test_log.txt",
    ):
        assert (output / name).is_file()
