"""Source-only quality thresholds and explicit missing-observation masks."""
import pickle

import numpy as np
import pytest

from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


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
    unknown_family = QualityObservabilityFamily().fit(source)
    unknown_mask = SourceCalibratedQualityMask().fit(unknown_family.transform(source),
                                                     unknown_family.feature_names)
    assert unknown_mask.available_["adc_clipping"] is False
    with pytest.raises(ValueError, match="availability"):
        mask.transform(unknown_family.transform(source), family.feature_names)


def test_degraded_source_cannot_define_safe_flatline_threshold():
    rng = np.random.default_rng(222)
    source = rng.normal(size=(20, 200, 8))
    source[:, :, 0] = 0.
    batch = FeatureBatch(source, 1000.)
    family = QualityObservabilityFamily().fit(batch)
    with pytest.raises(ValueError, match="too degraded"):
        SourceCalibratedQualityMask().fit(family.transform(batch), family.feature_names)
