import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from emgimu.baseline import BaselinePredictor
from emgimu.cli import cmd_evaluate, cmd_freeze_model, cmd_train_baseline, cmd_validate
from emgimu.simulate import create_smoke_dataset


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class CliSmokeTests(unittest.TestCase):
    def test_dataset_to_model_to_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_smoke_dataset(Path(directory) / "data")
            self.assertEqual(cmd_validate(argparse.Namespace(dataset=str(root), formal=False)), 0)
            model = Path(directory) / "baseline.pkl"
            validation = Path(directory) / "validation.json"
            self.assertEqual(cmd_train_baseline(argparse.Namespace(dataset=str(root), output=str(model))), 0)
            self.assertTrue(model.exists())
            self.assertEqual(cmd_evaluate(argparse.Namespace(
                dataset=str(root), model=str(model), split="validation", unlock_test=False,
                output=str(validation),
            )), 0)
            evidence = json.loads(validation.read_text(encoding="utf-8"))
            self.assertEqual(evidence["split"], "validation")
            self.assertEqual(evidence["format_version"], 2)
            self.assertEqual(evidence["metric_weighting"], "equal_session_then_trial_v1")
            self.assertEqual(evidence["confusion_matrix_weighting"], "raw_window_counts")
            self.assertEqual(evidence["model_status"], "development_unvalidated")
            self.assertEqual(len(evidence["model_sha256"]), 64)
            self.assertEqual(len(evidence["dataset_sha256"]), 64)
            self.assertIn("meets_v1_target", evidence)
            trained = BaselinePredictor.load(model)
            self.assertEqual(
                trained.metadata["window_weighting"], "equal_session_then_trial_v1",
            )
            frozen = Path(directory) / "formal.pkl"
            self.assertEqual(cmd_freeze_model(argparse.Namespace(
                dataset=str(root), model=str(model), kind="baseline",
                validation_report=str(validation), output=str(frozen),
            )), 1)
            self.assertFalse(frozen.exists())
            self.assertFalse(frozen.with_suffix(".pkl.freeze.json").exists())

    def test_test_session_requires_explicit_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_smoke_dataset(Path(directory) / "data")
            model = Path(directory) / "baseline.pkl"
            cmd_train_baseline(argparse.Namespace(dataset=str(root), output=str(model)))
            with self.assertRaises(SystemExit):
                cmd_evaluate(argparse.Namespace(
                    dataset=str(root), model=str(model), split="test", unlock_test=False,
                ))


if __name__ == "__main__":
    unittest.main()
