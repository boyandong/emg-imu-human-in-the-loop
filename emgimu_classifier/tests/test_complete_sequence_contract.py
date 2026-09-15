import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.temporal import CompleteSequenceBatch, TemporalTemplateFamily
from emgimu.feature_bank.template_study import run
from pathlib import Path


class CompleteSequenceContractTests(unittest.TestCase):
    def test_short_and_sparse_batches_cannot_fit_or_predict(self):
        x = np.random.default_rng(2).normal(size=(4,8,4))
        for rate in (200., 1.):
            with self.assertRaisesRegex(ValueError,'complete sequences'):
                TemporalTemplateFamily().fit(FeatureBatch(x,rate),np.array([0,0,1,1]))
        batch = CompleteSequenceBatch(x,1.,durations_seconds=np.ones(4),full_coverage=True)
        fitted = TemporalTemplateFamily().fit(batch,np.array([0,0,1,1]))
        with self.assertRaisesRegex(ValueError,'complete sequences'):
            fitted.transform(FeatureBatch(x,1.))
        with self.assertRaisesRegex(ValueError,'ineligible'):
            run(Path('absent.zip'),Path('must_not_be_created'),'final')

    def test_native_duration_and_coverage_are_not_inferred_from_bin_rate(self):
        x = np.ones((4,32,4))
        for durations,coverage in ((np.full(4,.2),True),(np.full(4,np.nan),True),(np.ones(4),False)):
            with self.assertRaises(ValueError):
                CompleteSequenceBatch(x,1.,durations_seconds=durations,full_coverage=coverage)
        batch = CompleteSequenceBatch(x,1.,durations_seconds=np.array([1.,2.,3.,4.]),full_coverage=True)
        selected = batch.take(np.array([3,1]))
        np.testing.assert_array_equal(selected.durations_seconds,[4.,2.])
        fitted = TemporalTemplateFamily().fit(batch,np.array([0,0,1,1]))
        batch.durations_seconds[0] = .2
        with self.assertRaisesRegex(ValueError,'native duration'):
            fitted.transform(batch)
