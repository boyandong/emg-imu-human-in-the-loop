import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.canonical_delivery import SCHEMAS, build, verify


class CanonicalDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.output=self.root/'delivery'
        (self.root/'manifests').mkdir()
        run='feature_bank_fixture_final'
        row={'run_id':run,'dataset':'fixture','subject':'1','condition':'ALL','feature_family':'F0',
             'shots_per_class':'0','macro_f1':'0.5','log_loss':'1.0'}
        for filename in SCHEMAS:
            with (self.root/filename).open('w',newline='') as h:
                writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        for name,meta in ((run,{'phase':'final','source_run':'feature_bank_fixture_validation'}),
                          ('feature_bank_fixture_validation',{'phase':'validation','target_session':2,'families':['F0']})):
            (self.root/'manifests'/f'{name}__run_manifest.json').write_text(json.dumps(meta))

    def test_training_state_reuse_never_inherits_other_phase_target_session(self):
        with contextlib.redirect_stdout(io.StringIO()):build(self.root,self.output)
        with (self.output/'feature_family_results.csv').open() as h:row=next(csv.DictReader(h))
        self.assertEqual(row['session/domain'],'N/A')
        self.assertEqual(row['calibration_budget'],'0')
        self.assertEqual(row['macro_f1'],'0.5')

    def test_changed_canonical_metrics_are_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):build(self.root,self.output)
        path=self.output/'feature_family_results.csv'
        path.write_text(path.read_text().replace('0.5','0.9'))
        with self.assertRaises(ValueError):verify(self.root,self.output)

    def test_native_trial_complementarity_uses_recorded_count_denominator(self):
        source=self.root/'error_complementarity.csv'
        row={'run_id':'feature_bank_fixture_final','dataset':'fixture','subject':'1',
             'evaluation_unit':'whole_native_trial_mean','evaluation_trials':'20',
             'disagreement_fraction':'0.35','core_correct_increment_wrong':'3',
             'core_wrong_increment_correct':'0'}
        with source.open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        with contextlib.redirect_stdout(io.StringIO()):build(self.root,self.output)
        with (self.output/'error_complementarity.csv').open() as h:
            derived=next(csv.DictReader(h))
        self.assertEqual(derived['disagreement_rate'],'0.35')
        self.assertEqual(float(derived['a_correct_b_wrong']),0.15)
        self.assertEqual(float(derived['a_wrong_b_correct']),0.0)
        notes=json.loads(derived['metadata_notes_json'])
        self.assertIn('core_correct_increment_wrong/evaluation_trials',notes['a_correct_b_wrong'])
        self.assertIn('core_wrong_increment_correct/evaluation_trials',notes['a_wrong_b_correct'])

    def test_native_trial_complementarity_rejects_invalid_counts(self):
        source=self.root/'error_complementarity.csv'
        row={'run_id':'feature_bank_fixture_final','dataset':'fixture',
             'evaluation_unit':'whole_native_trial_mean','evaluation_trials':'2',
             'core_correct_increment_wrong':'3'}
        with source.open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        with self.assertRaisesRegex(ValueError,'Invalid native trial complementarity counts'):
            build(self.root,self.output)
