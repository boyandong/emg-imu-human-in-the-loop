from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import torch

from emgforce.inference.model_bundle import create_gesture_model, load_model_bundle
from mpf_tds.features import MPFConfig, MultiBandMatrixPowerFeatures
from mpf_tds.model import MPFTDSConfig, MPFTDSNetwork
from mpf_tds.metrics import _events_from_probabilities, evaluate_paper_fnr
from mpf_tds.data import _validate_training_dataset
from mpf_tds.train import paper_learning_rate

OFFICIAL_ROOT = Path(__file__).parents[1] / "third_party" / "generic-neuromotor-interface"
sys.path.insert(0, str(OFFICIAL_ROOT))
from generic_neuromotor_interface.networks import MultivariatePowerFrequencyFeatures


def test_mpf_shape_frequency_bins_and_finite_values() -> None:
    config = MPFConfig()
    extractor = MultiBandMatrixPowerFeatures(config)
    t = torch.arange(1000) / 2000
    signal = torch.stack([torch.sin(2 * torch.pi * (50 + 20 * channel) * t) for channel in range(8)])[None]
    features = extractor(signal)
    assert features.shape == (1, 19, 192)
    assert torch.isfinite(features).all()
    frequencies = torch.fft.rfftfreq(64, 1 / 2000)
    counts = [int(((frequencies >= low) & (frequencies < high if high < 1000 else frequencies <= high)).sum())
              for low, high in config.frequency_bins]
    assert all(count > 0 for count in counts)


def test_mpf_batch_and_single_window_parity_and_spd_stability() -> None:
    torch.manual_seed(3)
    extractor = MultiBandMatrixPowerFeatures()
    signal = torch.randn(1, 8, 654)
    batch = extractor(signal)
    single = extractor(signal[..., :254])
    torch.testing.assert_close(batch[:, :1], single)
    zero_features = extractor(torch.zeros(1, 8, 254))
    assert torch.isfinite(zero_features).all()
    assert not torch.equal(zero_features, torch.zeros_like(zero_features))


def test_mpf_training_dataset_contract_is_enforced(tmp_path) -> None:
    path = tmp_path / "aligned.hdf5"
    dtype = np.dtype([("emg", "f4", (8,)), ("time", "f8")])
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=np.zeros(32, dtype=dtype))
        dataset.attrs["sample_rate"] = 2000.0
        dataset.attrs["preprocessing_version"] = ""
        try:
            _validate_training_dataset(dataset, path, MPFConfig())
        except ValueError as exc:
            assert "meta_8ch_v1" in str(exc)
        else:
            raise AssertionError("未预处理训练文件被接受")


def test_mpf_is_numerically_equivalent_to_meta_official_implementation() -> None:
    torch.manual_seed(17)
    config = MPFConfig(matrix_log_mode="legacy_nan_to_zero")
    signal = torch.randn(2, 8, 1054)
    actual = MultiBandMatrixPowerFeatures(config)(signal)
    official = MultivariatePowerFrequencyFeatures(
        window_length=200, stride=40, n_fft=64, fft_stride=10,
        frequency_bins=config.frequency_bins,
    )(signal).permute(0, 4, 1, 2, 3)
    channels = torch.arange(8)
    expected = torch.cat([
        torch.cat([official[:, :, band, channels, (channels + offset) % 8]
                   for offset in range(4)], dim=-1)
        for band in range(6)
    ], dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_mpf_contains_cross_channel_covariance_and_rotates_consistently() -> None:
    torch.manual_seed(9)
    extractor = MultiBandMatrixPowerFeatures()
    base = torch.randn(1, 1, 254)
    signal = torch.cat([base, base * 0.8 + 0.1 * torch.randn_like(base),
                        torch.randn(1, 6, 254)], dim=1)
    features = extractor(signal).reshape(1, 1, 6, 4, 8)
    assert features[..., 1, :].abs().max() > 1e-3
    rotated = extractor(signal.roll(1, dims=1)).reshape(1, 1, 6, 4, 8)
    # Eigendecomposition of nearly repeated eigenvalues has small platform-
    # dependent differences, while the circular feature ordering is preserved.
    torch.testing.assert_close(rotated, features.roll(1, dims=-1), atol=2e-2, rtol=3e-3)


def test_tds_is_causal_has_gradients_and_correct_shape() -> None:
    torch.manual_seed(4)
    config = MPFTDSConfig(feature_dim=32, num_scales=3, blocks_per_scale=2, dropout=0.0)
    model = MPFTDSNetwork(config).eval()
    before = torch.randn(2, 30, 192, requires_grad=True)
    after = before.detach().clone(); after[:, 20:] += 100
    first = model(before); second = model(after)
    assert first.shape == (2, 30, 9)
    torch.testing.assert_close(first[:, :20], second[:, :20], atol=1e-5, rtol=1e-5)
    first.sum().backward()
    assert before.grad is not None and torch.isfinite(before.grad).all()


def test_tds_can_overfit_one_batch() -> None:
    torch.manual_seed(12)
    config = MPFTDSConfig(feature_dim=16, num_scales=2, blocks_per_scale=2, dropout=0.0)
    model = MPFTDSNetwork(config)
    features = torch.randn(1, 8, 192)
    target = torch.zeros(1, 8, 9); target[..., 4] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial = torch.nn.functional.binary_cross_entropy_with_logits(model(features), target)
    for _ in range(20):
        optimizer.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), target)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.binary_cross_entropy_with_logits(model(features), target)
    assert final < initial * 0.25


def test_mpf_tds_cpu_latency_is_bounded() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    extractor = MultiBandMatrixPowerFeatures(); model = MPFTDSNetwork(
        MPFTDSConfig(feature_dim=32, num_scales=3, blocks_per_scale=2)).eval()
    signal = torch.randn(1, 8, 4000)
    with torch.inference_mode():
        model(extractor(signal)); started = time.perf_counter(); model(extractor(signal))
    assert (time.perf_counter() - started) < 1.0


def test_mpf_tds_bundle_load_and_predict(tmp_path) -> None:
    config = MPFTDSConfig(feature_dim=16, num_scales=2, blocks_per_scale=2, dropout=0.0)
    model = MPFTDSNetwork(config)
    root = tmp_path / "mpf"; root.mkdir(); artifact = root / "model.pt"
    legacy_features = MPFConfig().to_dict()
    legacy_features.pop("matrix_log_mode")
    torch.save({"state_dict": model.state_dict(), "feature_config": legacy_features,
                "implementation_revision": 2,
                "network_config": config.to_dict()}, artifact)
    labels = [f"c{i}" for i in range(9)]
    (root / "labels.json").write_text(json.dumps({"model_outputs": labels, "display_names": {}}))
    (root / "preprocessing.json").write_text("{}")
    manifest = {"bundle_version": 2, "model_id": "mpf-test", "algorithm_id": "personal_mpf_tds_v1",
                "runtime_backend": "mpf_tds", "artifact": {"filename": "model.pt", "format": "mpf_tds_state_dict",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
                "signal": {"input_channels": 8, "sample_rate_hz": 2000},
                "network": {"input_channels": 8, "output_channels": 9, "sample_rate_hz": 2000,
                            "left_context_samples": 199, "stride": 40},
                "labels_file": "labels.json", "preprocessing_file": "preprocessing.json"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    bundle = load_model_bundle(root); runtime = create_gesture_model(bundle)
    assert runtime.feature_config.matrix_log_mode == "legacy_nan_to_zero"
    output = runtime.predict(np.random.default_rng(2).normal(size=(1000, 8)).astype(np.float32))
    assert output.shape == (9, 19)
    assert np.isfinite(output).all()


def test_paper_learning_rate_schedule() -> None:
    assert np.isclose(paper_learning_rate(1), 2e-4)
    assert np.isclose(paper_learning_rate(5), 1e-3)
    assert np.isclose(paper_learning_rate(25), 1e-3)
    assert np.isclose(paper_learning_rate(26), 5e-4)
    assert np.isclose(paper_learning_rate(300), 5e-4)


def test_paper_fnr_uses_fixed_threshold_and_counts_missing_events() -> None:
    labels = ("index_press", "index_release", "middle_press", "middle_release",
              "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up")
    targets = np.zeros((1, 80, 9), dtype=np.float32)
    probabilities = np.zeros_like(targets)
    targets[0, 10:12, 4] = 1; targets[0, 40:42, 5] = 1
    probabilities[0, 11, 4] = 0.8
    metrics = evaluate_paper_fnr(probabilities, targets, labels, threshold=0.35)
    assert metrics.correct["thumb_click"] == 1
    assert metrics.per_class_fnr["thumb_click"] == 0.0
    assert metrics.per_class_fnr["thumb_down"] == 1.0
    assert np.isclose(metrics.mean_fnr, 0.5)


def test_paper_debounce_suppresses_nonrelease_after_release() -> None:
    labels = ("index_press", "index_release", "middle_press", "middle_release",
              "thumb_click", "thumb_down", "thumb_in", "thumb_out", "thumb_up")
    probabilities = np.zeros((100, 9), dtype=np.float32)
    probabilities[10, labels.index("index_release")] = 0.8
    probabilities[20, labels.index("thumb_click")] = 0.9

    events = _events_from_probabilities(
        probabilities, labels, threshold=0.35, frame_rate=1000.0)

    assert [event["name"] for event in events] == ["index_release"]
