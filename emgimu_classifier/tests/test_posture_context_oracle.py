import pickle
import numpy as np
import pytest
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.families import BodyContextFamily


def test_source_fixed_posture_one_hot_has_known_order_and_no_fabricated_imu():
    source=FeatureBatch(np.ones((3,40,8)),200.,posture=np.array(['up','down','up']))
    model=BodyContextFamily().fit(source);before=pickle.dumps(model)
    query=FeatureBatch(np.ones((3,40,8)),200.,posture=np.array(['down','up','down']))
    np.testing.assert_array_equal(model.transform(query),[[1,0],[0,1],[1,0]])
    assert model.feature_names==('F6.posture.down','F6.posture.up')
    assert not model.use_imu_
    assert pickle.dumps(model)==before
    with pytest.raises(ValueError,match='unknown posture'):
        model.transform(FeatureBatch(np.ones((1,40,8)),200.,posture=np.array(['side'])))
