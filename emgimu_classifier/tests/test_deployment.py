import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from emgimu.baseline import BaselinePredictor
from emgimu.calibration import SessionCalibration
from emgimu.cli import cmd_serve_deployment
from emgimu.deployment import create_deployment_bundle, verify_deployment_bundle


class DeploymentTests(unittest.TestCase):
    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _formal_fixture(self, root: Path):
        rng = np.random.default_rng(19)
        count = 140
        direction = np.arange(count) % 7
        gesture = np.arange(count) % 4
        imu = rng.normal(0, 0.03, (count, 36))
        emg = rng.normal(0, 0.03, (count, 48))
        imu[np.arange(count), direction] += 2.5
        emg[np.arange(count), gesture] += 2.5
        predictor = BaselinePredictor().fit(
            emg, imu, direction, gesture,
            validation_emg_features=emg,
            validation_imu_features=imu,
            validation_direction=direction,
            validation_gesture=gesture,
        )
        source_hash = "a" * 64
        dataset_hash = "b" * 64
        validation = root / "validation.json"
        validation.write_text(json.dumps({
            "format_version": 2,
            "split": "validation",
            "metric_weighting": "equal_session_then_trial_v1",
            "confusion_matrix_weighting": "raw_window_counts",
            "model_sha256": source_hash,
            "dataset_sha256": dataset_hash,
            "meets_v1_target": True,
        }), encoding="utf-8")
        validation_hash = self._sha(validation)
        predictor.metadata.update({
            "model_status": "formal_frozen",
            "source_model_sha256": source_hash,
            "formal_dataset_sha256": dataset_hash,
            "validation_report_sha256": validation_hash,
            "freeze_protocol_version": 1,
        })
        model = root / "formal.pkl"
        predictor.save(model)
        receipt = root / "formal.pkl.freeze.json"
        receipt.write_text(json.dumps({
            "format_version": 1,
            "status": "formal_frozen",
            "source_model_sha256": source_hash,
            "frozen_model_sha256": self._sha(model),
            "dataset_sha256": dataset_hash,
            "validation_report_path": str(validation),
            "validation_report_sha256": validation_hash,
        }), encoding="utf-8")
        calibration = root / "calibration.json"
        calibration.write_text(
            json.dumps(SessionCalibration.identity().to_dict()), encoding="utf-8",
        )
        return model, receipt, calibration

    def test_bundle_verifies_and_detects_member_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, receipt, calibration = self._formal_fixture(root)
            manifest = create_deployment_bundle(
                root / "bundle", model=model, kind="baseline",
                freeze_receipt=receipt, calibration=calibration,
            )
            report = verify_deployment_bundle(manifest)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["runtime"]["config_version"], 1)
            self.assertEqual(report["runtime"]["input_port"], 9100)
            self.assertFalse(report["runtime"]["publish_legacy"])
            self.assertIn("validation_report", report["members"])
            bundled_calibration = Path(report["members"]["calibration"])
            bundled_calibration.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_deployment_bundle(manifest)

    def test_formal_serve_uses_only_verified_bundle_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, receipt, calibration = self._formal_fixture(root)
            manifest = create_deployment_bundle(
                root / "bundle", model=model, kind="baseline",
                freeze_receipt=receipt, calibration=calibration,
                input_host="10.0.0.4", input_port=9201,
                output_host="10.0.0.5", output_port=9202,
                publish_legacy=True,
            )
            args = Namespace(
                manifest=str(manifest), device="cpu", audit_dir=str(root / "audit"),
                run_id="formal-run", audit_fsync_every=7,
            )
            with patch("emgimu.cli.cmd_serve", return_value=0) as serve:
                self.assertEqual(cmd_serve_deployment(args), 0)
            forwarded = serve.call_args.args[0]
            self.assertEqual(forwarded.model, str((root / "bundle" / "model.pkl").resolve()))
            self.assertEqual(forwarded.input_host, "10.0.0.4")
            self.assertEqual(forwarded.input_port, 9201)
            self.assertEqual(forwarded.output_host, "10.0.0.5")
            self.assertEqual(forwarded.output_port, 9202)
            self.assertTrue(forwarded.publish_legacy)
            self.assertEqual(forwarded.deployment_manifest, str(manifest.resolve()))
            self.assertEqual(forwarded.audit_fsync_every, 7)

    def test_invalid_runtime_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, receipt, calibration = self._formal_fixture(root)
            manifest = create_deployment_bundle(
                root / "bundle", model=model, kind="baseline",
                freeze_receipt=receipt, calibration=calibration,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["runtime"]["input_port"] = 0
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ports"):
                verify_deployment_bundle(manifest)

    def test_rehashed_obsolete_validation_evidence_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, receipt, calibration = self._formal_fixture(root)
            manifest = create_deployment_bundle(
                root / "bundle", model=model, kind="baseline",
                freeze_receipt=receipt, calibration=calibration,
            )
            evidence = root / "bundle" / "validation_evidence.json"
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            payload["metric_weighting"] = "uniform_windows"
            evidence.write_text(json.dumps(payload), encoding="utf-8")
            deployment = json.loads(manifest.read_text(encoding="utf-8"))
            deployment["members"]["validation_report"]["sha256"] = self._sha(evidence)
            # Rehashing both the manifest and copied receipt is insufficient:
            # the frozen model metadata retains the original evidence hash.
            bundled_receipt = root / "bundle" / "freeze_receipt.json"
            receipt_payload = json.loads(bundled_receipt.read_text(encoding="utf-8"))
            receipt_payload["validation_report_sha256"] = self._sha(evidence)
            bundled_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
            deployment["members"]["freeze_receipt"]["sha256"] = self._sha(bundled_receipt)
            manifest.write_text(json.dumps(deployment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance|obsolete"):
                verify_deployment_bundle(manifest)

    def test_reference_or_development_model_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, receipt, calibration = self._formal_fixture(root)
            predictor = BaselinePredictor.load(model)
            predictor.metadata["model_status"] = "development_unvalidated"
            predictor.save(model)
            receipt.write_text(json.dumps({
                "format_version": 1,
                "status": "formal_frozen",
                "source_model_sha256": predictor.metadata["source_model_sha256"],
                "frozen_model_sha256": self._sha(model),
                "dataset_sha256": predictor.metadata["formal_dataset_sha256"],
                "validation_report_path": str(root / "validation.json"),
                "validation_report_sha256": predictor.metadata["validation_report_sha256"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formal_frozen"):
                create_deployment_bundle(
                    root / "bundle", model=model, kind="baseline",
                    freeze_receipt=receipt, calibration=calibration,
                )


if __name__ == "__main__":
    unittest.main()
