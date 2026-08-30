from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from emgforce.processing.meta_corpus import (
    CORPUS_FILENAME, MANIFEST_FILENAME, resolve_dataset_split,
    upsert_corpus_entry, validate_training_export,
)
from emgforce.processing.training_preprocessing import (
    PREPROCESSING_VERSION, TARGET_MEDIAN_ABS, preprocess_training_emg,
)


def test_training_preprocessing_removes_low_frequency_without_mutating_raw() -> None:
    rate = 2000.0
    time = np.arange(int(rate * 4.0)) / rate
    low = 1000.0 * np.sin(2 * np.pi * 10.0 * time)
    carrier = 200.0 * np.sin(2 * np.pi * 173.0 * time)
    raw = np.column_stack([
        low + carrier * (1.0 + channel / 10.0) for channel in range(8)
    ]).astype(np.int32)
    original = raw.copy()

    result = preprocess_training_emg(raw, rate)

    np.testing.assert_array_equal(raw, original)
    assert result.signal.shape == raw.shape
    assert result.signal.dtype == np.float32
    assert np.isclose(np.median(np.abs(result.signal)), TARGET_MEDIAN_ABS, rtol=0.02)
    stable = result.signal[int(rate):int(3 * rate), 0]
    spectrum = np.abs(np.fft.rfft(stable))
    frequencies = np.fft.rfftfreq(len(stable), 1.0 / rate)
    low_power = spectrum[np.argmin(np.abs(frequencies - 10.0))]
    carrier_power = spectrum[np.argmin(np.abs(frequencies - 173.0))]
    assert low_power < carrier_power * 0.01


def test_session_split_policy() -> None:
    assert resolve_dataset_split("S01") == "train"
    assert resolve_dataset_split("participant_S08") == "train"
    assert resolve_dataset_split("S09") == "train"
    assert resolve_dataset_split("S10") == "train"
    assert resolve_dataset_split("S11") == "val"
    assert resolve_dataset_split("S12") == "val"
    assert resolve_dataset_split("S13") == "test"
    assert resolve_dataset_split("S15") == "test"
    assert resolve_dataset_split("custom", "val") == "val"


def _write_aligned(path: Path, *, prompt_count: int = 2,
                   preprocessing_version: str = PREPROCESSING_VERSION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = 1_700_000_000.0 + np.arange(4000) / 2000.0
    dtype = np.dtype([("emg", "f4", (8,)), ("time", "f8")])
    data = np.zeros(len(times), dtype=dtype)
    data["time"] = times
    with h5py.File(path, "x") as handle:
        dataset = handle.create_dataset("data", data=data)
        dataset.attrs["task"] = "discrete_gestures"
        dataset.attrs["sample_rate"] = 2000.0
        dataset.attrs["num_emg_channels"] = 8
        dataset.attrs["preprocessing_version"] = preprocessing_version
        meta = handle.create_group("meta")
        meta.attrs["participant_id"] = "P001"
        meta.attrs["session_id"] = "S09"
        meta.attrs["protocol_name"] = "meta_discrete_7_short_v2"
        meta.attrs["dataset_split"] = "auto"
    names = ["index_press", "index_release"][:prompt_count]
    prompts = pd.DataFrame({
        "name": names,
        "time": times[100:100 + len(names)],
    })
    prompts.to_hdf(path, key="prompts", mode="a", format="fixed")
    pd.DataFrame({
        "start": [times[0]], "end": [times[-1]], "name": ["default"],
    }).to_hdf(path, key="stages", mode="a", format="fixed")


def test_corpus_upsert_is_relative_atomic_and_deduplicated(tmp_path) -> None:
    data_root = tmp_path / "data"
    aligned = data_root / "P001" / "2026-01-01_S09" / "session_meta_aligned.hdf5"
    _write_aligned(aligned)

    first = upsert_corpus_entry(aligned)
    second = upsert_corpus_entry(aligned)

    assert first == second == data_root / CORPUS_FILENAME
    corpus = pd.read_csv(first)
    assert len(corpus) == 1
    assert corpus.iloc[0]["dataset"] == \
        "P001/2026-01-01_S09/session_meta_aligned.hdf5"
    assert corpus.iloc[0]["split"] == "train"
    assert corpus.iloc[0]["prompt_count"] == 2
    assert corpus.iloc[0]["preprocessing_version"] == PREPROCESSING_VERSION
    assert (data_root / MANIFEST_FILENAME).exists()
    assert not list(data_root.glob(".*.tmp"))


def test_training_readiness_rejects_legacy_unfiltered_export(tmp_path) -> None:
    aligned = tmp_path / "legacy.hdf5"
    _write_aligned(aligned, preprocessing_version="")

    result = validate_training_export(aligned)

    assert not result.ready
    assert any("未预处理" in problem for problem in result.problems)
