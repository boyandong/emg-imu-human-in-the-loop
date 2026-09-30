"""Source-only quality thresholds and explicit missing-observation masks."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily
from benchmarks.song_quality_mask_v1 import frozen_f0_gate_control


def test_constant_channel_is_rejected_without_target_refit():
    rng = np.random.default_rng(220)
    source = FeatureBatch(rng.normal(size=(40, 200, 8)), 1000.)
    family = QualityObservabilityFamily(ring_topology=True).fit(source)
    source_features = family.transform(source)
    mask = SourceCalibratedQualityMask().fit(source_features, family.feature_names)
    frozen = pickle.dumps(mask)
    clean = source.emg[:5].copy()
    fault = clean.copy()
    fault[:, :, 0] = fault[:, :1, 0]
    clean_quality = mask.transform(family.transform(FeatureBatch(clean, 1000.)),
                                   family.feature_names)
    fault_quality = mask.transform(family.transform(FeatureBatch(fault, 1000.)),
                                   family.feature_names)
    assert clean_quality.shape == fault_quality.shape == (5, 12)
    assert np.all(fault_quality[:, 0] == 0.)
    assert np.all(fault_quality[:, -4] >= 1.)
    assert np.all(fault_quality[:, -2] == 0.)
    assert float(np.mean(clean_quality[:, 0])) > float(np.mean(fault_quality[:, 0]))
    assert mask.available_ == {"adc_clipping": False, "line_noise": True,
                               "low_frequency_pre_highpass": False}
    assert not np.any(mask.structural_invalid(family.transform(FeatureBatch(clean, 1000.)),
                                              family.feature_names))
    assert np.all(mask.structural_invalid(family.transform(FeatureBatch(fault, 1000.)),
                                          family.feature_names)[:, 0])
    assert frozen == pickle.dumps(mask)


def test_known_adc_clip_and_unknown_adc_are_not_confused():
    rng = np.random.default_rng(221)
    source = FeatureBatch(rng.normal(scale=.1, size=(35, 200, 8)), 1000.)
    family = QualityObservabilityFamily(adc_min=-1., adc_max=1.).fit(source)
    mask = SourceCalibratedQualityMask().fit(family.transform(source), family.feature_names)
    target = source.emg[:3].copy()
    target[:, :100, 2] = 1.
    scored = mask.transform(family.transform(FeatureBatch(target, 1000.)),
                            family.feature_names)
    assert mask.available_["adc_clipping"] is True
    assert np.all(scored[:, 2] == 0.)
    assert np.all(mask.structural_invalid(family.transform(FeatureBatch(target, 1000.)),
                                          family.feature_names)[:, 2])
    unknown_family = QualityObservabilityFamily().fit(source)
    unknown_mask = SourceCalibratedQualityMask().fit(unknown_family.transform(source),
                                                     unknown_family.feature_names)
    assert unknown_mask.available_["adc_clipping"] is False
    with pytest.raises(ValueError, match="availability"):
        mask.transform(unknown_family.transform(source), family.feature_names)
    with pytest.raises(ValueError, match="availability"):
        mask.structural_invalid(unknown_family.transform(source), family.feature_names)


def test_degraded_source_cannot_define_safe_flatline_threshold():
    rng = np.random.default_rng(222)
    source = rng.normal(size=(20, 200, 8))
    source[:, :, 0] = 0.
    batch = FeatureBatch(source, 1000.)
    family = QualityObservabilityFamily().fit(batch)
    with pytest.raises(ValueError, match="too degraded"):
        SourceCalibratedQualityMask().fit(family.transform(batch), family.feature_names)


def test_frozen_trial_gate_counts_errors_and_correct_rejections():
    rows = [
        {"session": "S03", "arm": "F0", "trial_id": "S03:1", "truth": "fist",
         "p_fist": ".9", "p_index_pinch": ".05", "p_neutral": ".03", "p_open_hand": ".02"},
        {"session": "S03", "arm": "F0", "trial_id": "S03:2", "truth": "neutral",
         "p_fist": ".7", "p_index_pinch": ".1", "p_neutral": ".1", "p_open_hand": ".1"},
    ]
    result = frozen_f0_gate_control(
        np.array(["S03:1", "S03:1", "S03:2"]),
        np.array(["fist", "fist", "neutral"]), np.array([.9, .4, .8]), rows, "S03")
    assert result["trials"] == 2
    assert result["accepted"] == 1
    assert result["correct_rejected"] == 1
    assert result["errors_rejected"] == 0
    assert result["accepted_error_rate"] == 1.
    with pytest.raises(ValueError, match="identities"):
        frozen_f0_gate_control(np.array(["S03:1"]), np.array(["fist"]),
                               np.array([.9]), rows, "S03")
