from __future__ import annotations

import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from mpf_tds.data import LABEL_HALF_WIDTH_SEC, LABEL_SHIFT_SEC


REPOSITORY_ROOT = Path(__file__).parents[1]
OFFICIAL_PACKAGE_ROOT = REPOSITORY_ROOT / "third_party" / "generic-neuromotor-interface"
sys.path.insert(0, str(OFFICIAL_PACKAGE_ROOT))


def test_local_discrete_model_restores_the_best_8_channel_architecture() -> None:
    config_path = (
        OFFICIAL_PACKAGE_ROOT
        / "config"
        / "lightning_module"
        / "discrete_gestures_module.yaml"
    )
    config = OmegaConf.load(config_path)

    network = instantiate(config.network)

    assert network.conv_layer.in_channels == 8
    assert network.conv_layer.out_channels == 512
    assert network.lstm.input_size == 512
    assert network.lstm.hidden_size == 512
    assert network.lstm.num_layers == 3
    assert network.projection.out_features == 9
    assert config.positive_class_weight == 1.0


def test_lstm_and_mpf_tds_share_one_100_ms_label_shift() -> None:
    config = OmegaConf.load(
        OFFICIAL_PACKAGE_ROOT / "config" / "data_module"
        / "discrete_gestures_data_module.yaml")
    expected = [
        LABEL_SHIFT_SEC - LABEL_HALF_WIDTH_SEC,
        LABEL_SHIFT_SEC + LABEL_HALF_WIDTH_SEC,
    ]

    assert list(config.transform.pulse_window) == [0.08, 0.12]
    assert all(abs(actual - wanted) < 1e-12 for actual, wanted in zip(
        config.transform.pulse_window, expected))
