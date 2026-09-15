import pickle
import unittest
import numpy as np
from emgimu.datasets.unibo_physiology import PhysiologyFeatureTransformer
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.validated_unibo import ValidatedUniBoFamily


class LegacyContractTests(unittest.TestCase):
    def test_existing_formula_and_schema_are_preserved_exactly(self):
        rng=np.random.default_rng(17);train=FeatureBatch(rng.normal(size=(16,40,4)),200)
        held=FeatureBatch(rng.normal(size=(7,40,4)),200);y=np.arange(16)%4;w=np.linspace(.5,2,16)
        for group in ('G0','G5'):
            old=PhysiologyFeatureTransformer((group,)).fit(train.emg,y,w)
            new=ValidatedUniBoFamily(group).fit(train,y,w);before=pickle.dumps(new)
            np.testing.assert_array_equal(old.transform(held.emg,np.ones(7)),new.transform(held))
            self.assertEqual(old.feature_names_,new.feature_names)
            self.assertEqual(before,pickle.dumps(new))

    def test_eight_channels_are_rejected_without_projection(self):
        with self.assertRaisesRegex(ValueError,'native four'):
            ValidatedUniBoFamily().fit(FeatureBatch(np.ones((8,40,8)),200),np.arange(8)%4)
