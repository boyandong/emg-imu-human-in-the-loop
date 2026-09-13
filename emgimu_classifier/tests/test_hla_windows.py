from pathlib import Path

import numpy as np

from emgimu.datasets.hla_schema import EMGTrial
from emgimu.datasets.hla_windows import HLA_MAIN_PROTOCOL, window_trial


def _trial() -> EMGTrial:
    samples = 200
    labels = np.zeros(samples, dtype=np.int16)
    labels[100:] = 1
    stable = np.ones(samples, dtype=bool)
    stable[75] = False
    return EMGTrial(
        path=Path("trial.npz"), dataset_id="demo", subject_id="s1", session_id="d1",
        trial_id="t1", channel_layout_id="four", sample_rate_hz=200.0,
        timestamp_ms=np.arange(samples) * 5.0,
        emg=np.arange(samples * 4, dtype=np.float64).reshape(samples, 4),
        source_label=labels.astype(str), task_label=labels, canonical_label=labels,
        stable_mask=stable,
    )


def test_main_protocol_uses_actual_trial_rate() -> None:
    assert HLA_MAIN_PROTOCOL.sample_counts(200.0) == (50, 10)
    assert HLA_MAIN_PROTOCOL.sample_counts(2048.0) == (512, 102)


def test_windowing_never_crosses_label_or_unstable_samples() -> None:
    source = _trial()
    result = window_trial(source)
    assert result.emg.shape[1:] == (50, 4)
    for start, label in zip(result.start_sample, result.labels):
        stop = start + 50
        assert source.stable_mask[start:stop].all()
        assert np.all(source.task_label[start:stop] == label)


def test_empty_result_has_stable_shape() -> None:
    source = _trial()
    source.stable_mask[:] = False
    result = window_trial(source)
    assert result.emg.shape == (0, 50, 4)
    assert len(result) == 0
