import copy
import pickle
import numpy as np
import pytest
from emgimu.feature_bank.document_session_v3 import DocumentSessionDescriptorV3
from emgimu.feature_bank.session_router_v1 import DocumentSessionRouterV1


def test_source_normalizers_and_weighted_probability_known_geometry():
    # Exact reviewed geometry fixture, independent of native window estimates.
    summary = DocumentSessionDescriptorV3()
    summary.user_id_ = 'user'; summary.source_session_ids_ = frozenset(['source'])
    summary.channels_ = 8; summary.rate_ = 200.
    summary.reference_ = {
        0: {'pattern': np.array([1., 0.]), 'log_bands': np.zeros(8),
            'log_scale': 0., 'covariance': np.eye(2)},
        1: {'pattern': np.array([0., 1.]), 'log_bands': np.full(8, 2.),
            'log_scale': 1., 'covariance': np.eye(2)*2},
    }
    router = DocumentSessionRouterV1().fit_source(summary)
    np.testing.assert_allclose(list(router.scales_.values()), [np.sqrt(2), np.sqrt(2)*np.log(2), 2., 1.])
    entry = {'log_global_activation_shift': 0., 'channel_quality_score_shift': [0.]*8,
             'scale_pattern_residual_norm': np.sqrt(2),
             'affine_invariant_trace_covariance_distance': 2*np.sqrt(2)*np.log(2),
             'mean_absolute_log_band_residual': 6.}
    d = {'version': 'document-f8-v3', 'user_id': 'user', 'source_sessions': ['source'],
         'sensor_channels': 8, 'sample_rate_hz': 200., 'classes': {'0': entry, '1': entry}}
    before = pickle.dumps(router)
    probabilities = np.array([[[1.,0.]], [[0.,1.]], [[.5,.5]], [[.25,.75]]])
    fused, weights, risks = router.fuse(probabilities, d)
    expected = np.exp(-np.arange(4)); expected = np.maximum(expected, .05); expected /= expected.sum()
    np.testing.assert_allclose(risks, [0.,1.,2.,3.])
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_allclose(fused[0], [expected[0]+.5*expected[2]+.25*expected[3],
                                       expected[1]+.5*expected[2]+.75*expected[3]])
    assert pickle.dumps(router) == before
    bad = copy.deepcopy(d); bad['user_id'] = 'other'
    with pytest.raises(ValueError, match='contract'):
        router.weights(bad)
    with pytest.raises(ValueError, match='normalized'):
        router.fuse(probabilities*2, d)
