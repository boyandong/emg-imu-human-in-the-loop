import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch neural extra is not installed",
)


def _inputs(batch: int = 2, channels: int = 8, feature_dim: int = 11):
    import torch

    return {
        "streams": {
            200: torch.randn(batch, channels, 50),
            500: torch.randn(batch, channels, 125),
        },
        "stream_availability": torch.tensor([[1, 1, 0]] * batch, dtype=torch.float32),
        "metadata": torch.randn(batch, channels, 11),
        "quality": torch.ones(batch, channels),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "feature_tokens": torch.randn(batch, channels, feature_dim),
    }


@pytest.mark.parametrize("representation", ["R0", "R1-core", "R2", "R3", "R2-wide"])
def test_all_representations_support_variable_channels_and_missing_stream(representation: str) -> None:
    import torch
    from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAMultiDatasetModel

    torch.manual_seed(42)
    model = HLAMultiDatasetModel({"myo": 7}, HLAEncoderConfig(representation=representation)).eval()
    inputs = _inputs(channels=5, feature_dim=6 if representation == "R0" else 11)
    output = model("myo", **inputs)
    assert output["representation"].shape == (2, 96)
    assert output["logits"].shape == (2, 7)
    assert torch.isfinite(output["logits"]).all()


def test_joint_signal_metadata_permutation_is_invariant() -> None:
    import torch
    from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAHardwareFlexibleEncoder

    torch.manual_seed(43)
    model = HLAHardwareFlexibleEncoder(HLAEncoderConfig(representation="R3", dropout=0)).eval()
    inputs = _inputs()
    original = model(**inputs)
    order = torch.tensor([3, 7, 0, 5, 2, 6, 1, 4])
    permuted = dict(inputs)
    permuted["streams"] = {rate: value[:, order] for rate, value in inputs["streams"].items()}
    for name in ("metadata", "quality", "channel_mask", "feature_tokens"):
        permuted[name] = inputs[name][:, order]
    torch.testing.assert_close(original, model(**permuted), rtol=1e-5, atol=1e-6)


def test_signal_only_permutation_changes_geometry_conditioned_output() -> None:
    import torch
    from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAHardwareFlexibleEncoder

    torch.manual_seed(44)
    model = HLAHardwareFlexibleEncoder(HLAEncoderConfig(representation="R2", dropout=0)).eval()
    inputs = _inputs()
    original = model(**inputs)
    order = torch.tensor([3, 7, 0, 5, 2, 6, 1, 4])
    changed = dict(inputs)
    changed["streams"] = {rate: value[:, order] for rate, value in inputs["streams"].items()}
    assert not torch.allclose(original, model(**changed))


def test_unavailable_stream_cannot_affect_output() -> None:
    import torch
    from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAHardwareFlexibleEncoder

    torch.manual_seed(45)
    model = HLAHardwareFlexibleEncoder(HLAEncoderConfig(representation="R2", dropout=0)).eval()
    inputs = _inputs()
    first = model(**inputs)
    altered = dict(inputs)
    altered["streams"] = dict(inputs["streams"])
    altered["streams"][1000] = torch.randn(2, 8, 250) * 1e6
    torch.testing.assert_close(first, model(**altered), rtol=1e-5, atol=1e-6)


def test_zero_input_and_bad_channel_mask_behavior() -> None:
    import torch
    from emgimu.datasets.hla_neural import HLAEncoderConfig, HLAHardwareFlexibleEncoder

    model = HLAHardwareFlexibleEncoder(HLAEncoderConfig(representation="R3", dropout=0)).eval()
    inputs = _inputs(batch=1, channels=4)
    for rate in inputs["streams"]:
        inputs["streams"][rate].zero_()
    inputs["metadata"].zero_()
    inputs["feature_tokens"].zero_()
    assert torch.isfinite(model(**inputs)).all()
    inputs["channel_mask"].zero_()
    with pytest.raises(ValueError, match="at least one usable channel"):
        model(**inputs)
