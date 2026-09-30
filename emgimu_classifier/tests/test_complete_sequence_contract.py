import unittest
import numpy as np
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.temporal import (
    CompleteSequenceBatch, CuedSequenceAssembler, TemporalTemplateFamily, dtw_distance,
)
from emgimu.feature_bank.template_study import run
from pathlib import Path


class CompleteSequenceContractTests(unittest.TestCase):
    def test_cued_assembler_only_emits_contiguous_full_native_bouts(self):
        assembler = CuedSequenceAssembler(100., 4, max_duration_s=2.)
        first = np.ones((70, 4))
        last = np.full((50, 4), 2.)
        assembler.begin('trial-a', 1000)
        assembler.append(first, 1000)
        assembler.append(last, 1070)
        first[:] = 9.
        batch = assembler.finish('trial-a', 1120)
        self.assertIsInstance(batch, CompleteSequenceBatch)
        self.assertEqual(batch.emg.shape, (1, 120, 4))
        self.assertEqual(batch.durations_seconds[0], 1.2)
        np.testing.assert_array_equal(batch.emg[0, :70], 1.)
        self.assertTrue(batch.full_coverage)

        assembler.begin('trial-b', 0)
        assembler.append(np.ones((100, 4)), 0)
        with self.assertRaisesRegex(ValueError, 'gap'):
            assembler.append(np.ones((3, 4)), 102)
        with self.assertRaisesRegex(RuntimeError, 'start'):
            assembler.finish('trial-b', 103)

    def test_cued_assembler_rejects_unattested_and_short_bouts(self):
        assembler = CuedSequenceAssembler(100., 4)
        with self.assertRaisesRegex(RuntimeError, 'start'):
            assembler.append(np.ones((100, 4)), 0)
        assembler.begin('trial-a', 0)
        assembler.append(np.ones((99, 4)), 0)
        with self.assertRaisesRegex(ValueError, 'one second'):
            assembler.finish('trial-a', 99)
        assembler.begin('trial-b', 20)
        assembler.append(np.ones((100, 4)), 20)
        with self.assertRaisesRegex(ValueError, 'end marker'):
            assembler.finish('wrong-trial', 120)
        assembler.begin('trial-c', 0)
        with self.assertRaisesRegex(ValueError, 'finite'):
            assembler.append(np.full((100, 4), np.nan), 0)

    def test_dtw_euclidean_cost_warp_band_and_path_length_on_known_sequences(self):
        first = np.array([[0.0], [1.0], [2.0]])
        identical = first.copy()
        changed = np.array([[0.0], [1.0], [3.0]])
        self.assertEqual(dtw_distance(first, identical, band=0), 0.0)
        self.assertAlmostEqual(dtw_distance(first, changed, band=0), 1.0 / 3.0)
        stretched = np.array([[0.0], [1.0], [1.0], [2.0]])
        self.assertEqual(dtw_distance(first, stretched, band=1), 0.0)

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
