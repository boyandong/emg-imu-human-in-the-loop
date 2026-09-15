import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('result_verifier',Path(__file__).resolve().parents[1]/'benchmarks/verify_feature_bank_results.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NestedSplitIntegrityTests(unittest.TestCase):
    def test_nested_subject_train_test_overlap_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifests').mkdir()
            (root/'provenance.json').write_text('[]')
            (root/'manifests/run__subject_split_trial_ids.json').write_text(json.dumps({'18':{'train':['same'],'test':['same']}}))
            with self.assertRaisesRegex(ValueError,'split leakage'):
                module.verify(root,root)

    def test_scenario_train_test_lists_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifests').mkdir()
            (root/'provenance.json').write_text('[]')
            (root/'manifests/run__scenario_split_trial_ids.json').write_text(json.dumps([{'subject':4,'train':['source'],'test':['target']}]))
            self.assertEqual(module.verify(root,root)['explicit_split_checks'],1)


if __name__=='__main__':unittest.main()
