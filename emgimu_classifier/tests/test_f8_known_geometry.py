"""Hand-computable F8 residual, cosine and pair-geometry coordinates."""
import pickle

import numpy as np

from emgimu.feature_bank.calibration import SessionSignature


def test_f8_three_class_geometry_and_source_immutability():
    labels = np.array([0, 0, 1, 1, 2, 2])
    long = np.repeat(np.array([[1., 0.], [0., 1.], [1., 1.]]), 2, axis=0)
    current = np.repeat(np.array([[2., 0.], [0., 2.], [1., 2.]]), 2, axis=0)
    signature = SessionSignature().fit_long_term(long, labels)
    frozen = pickle.dumps(signature)
    output = signature.from_session_calibration(current, labels)
    expected_norms = np.array([1., 1., 1.])
    expected_cosines = np.array([1., 1., 3/np.sqrt(10)])
    expected_geometry = np.array([
        np.sqrt(8)-np.sqrt(2),
        np.sqrt(5)-1,
        0.,
    ])
    np.testing.assert_allclose(output, np.r_[expected_norms, expected_cosines, expected_geometry],
                               atol=1e-7, rtol=0)
    assert output.shape == (2*3 + 3*(3-1)//2,)
    assert signature.feature_names == (
        "F8.residual_norm.0", "F8.residual_norm.1", "F8.residual_norm.2",
        "F8.cosine_agreement.0", "F8.cosine_agreement.1", "F8.cosine_agreement.2",
        "F8.geometry_delta.0.1", "F8.geometry_delta.0.2", "F8.geometry_delta.1.2",
    )
    assert frozen == pickle.dumps(signature)
