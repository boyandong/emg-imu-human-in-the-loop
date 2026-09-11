from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).parents[1]
OFFICIAL_PACKAGE_ROOT = REPOSITORY_ROOT / "third_party" / "generic-neuromotor-interface"
sys.path.insert(0, str(OFFICIAL_PACKAGE_ROOT))

from generic_neuromotor_interface.lightning import DiscreteGesturesModule


def _module(positive_class_weight: float) -> DiscreteGesturesModule:
    return DiscreteGesturesModule(
        network=torch.nn.Linear(1, 1),
        optimizer=torch.optim.Adam,
        learning_rate=5e-4,
        lr_scheduler_milestones=[25],
        lr_scheduler_factor=0.5,
        warmup_start_factor=0.001,
        warmup_end_factor=1.0,
        warmup_total_epochs=5,
        gradient_clip_val=0.5,
        positive_class_weight=positive_class_weight,
    )


def test_positive_class_weight_only_scales_positive_bce_targets() -> None:
    module = _module(5.0)
    logits = torch.zeros(2)
    targets = torch.tensor([0.0, 1.0])

    losses = module.weighted_bce_loss(logits, targets)

    torch.testing.assert_close(losses[1], losses[0] * 5.0)


def test_positive_class_weight_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        _module(0.0)


def test_release_mask_stays_active_when_release_is_outside_window() -> None:
    module = _module(1.0)
    labels = torch.zeros(1, 9, 20)
    labels[0, 0, 5:8] = 1.0

    mask = module.mask_generator(labels)

    torch.testing.assert_close(mask[0, 0, :5], torch.zeros(5))
    torch.testing.assert_close(mask[0, 0, 5:], torch.ones(15))


def test_gradient_clipping_is_not_returned_as_optimizer_metadata() -> None:
    configured = _module(1.0).configure_optimizers()
    assert "gradient_clip_val" not in configured
