from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).parents[1]
OFFICIAL_PACKAGE_ROOT = REPOSITORY_ROOT / "third_party" / "generic-neuromotor-interface"
sys.path.insert(0, str(OFFICIAL_PACKAGE_ROOT))

from generic_neuromotor_interface.augmentation import RotationAugmentation


def test_discrete_rotation_moves_channels_without_shifting_time(monkeypatch) -> None:
    emg = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    monkeypatch.setattr(np.random, "choice", lambda _choices: 1)

    augmented = RotationAugmentation(rotation=1, channel_dim=0)(emg)

    torch.testing.assert_close(augmented, torch.roll(emg, 1, dims=0))
    for output_channel, input_channel in enumerate((3, 0, 1, 2)):
        torch.testing.assert_close(augmented[output_channel], emg[input_channel])


def test_default_rotation_keeps_time_channel_tasks_compatible(monkeypatch) -> None:
    emg = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    monkeypatch.setattr(np.random, "choice", lambda _choices: -1)

    augmented = RotationAugmentation(rotation=1)(emg)

    torch.testing.assert_close(augmented, torch.roll(emg, -1, dims=-1))
