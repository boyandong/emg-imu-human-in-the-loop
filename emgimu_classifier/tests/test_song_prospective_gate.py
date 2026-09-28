import json
import tempfile
import unittest
from pathlib import Path

import h5py

from benchmarks.song_real8.prospective_gate import check_session, sha256


class ProspectiveGateTests(unittest.TestCase):
    def test_new_session_passes_but_old_and_modified_model_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"
            manifest = root / "manifest.json"
            model.write_text('{"frozen":true}', encoding="utf-8")
            manifest.write_text('{"model":"frozen"}', encoding="utf-8")
            candidate = root / "2026-09-29_S05"
            candidate.mkdir()
            signal = candidate / "session.h5"
            with h5py.File(signal, "w") as handle:
                meta = handle.create_group("meta")
                meta.attrs.update({
                    "schema_version": "3.0", "session_id": "S05", "date": "2026-09-29",
                    "protocol_name": "jilv_music_28_v2", "dataset_split": "test",
                    "model_frozen_confirmed": True, "num_emg_channels": 8,
                    "emg_nominal_rate_hz": 250, "imu_nominal_rate_hz": 112,
                })
                handle.create_dataset("streams/emg/raw", data=[[1, 2, 3, 4, 5, 6, 7, 8]])
                handle.create_dataset("streams/imu/accel", data=[[1, 2, 3]])
                handle.create_dataset("trials", data=[1])
            old_hash = "a" * 64
            freeze = {
                "frozen_date_local": "2026-09-28",
                "model_relative_path": "model.json",
                "model_sha256": sha256(model),
                "model_manifest_relative_path": "manifest.json",
                "model_manifest_sha256": sha256(manifest),
                "protocol_name": "jilv_music_28_v2",
                "existing_session_hdf5_sha256": {"S04": old_hash},
                "required_hand_states": ["neutral", "index_pinch", "fist", "open_hand"],
                "scope": "Eligibility only",
            }
            freeze_path = root / "freeze.json"
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            config = {"session": {
                "session_id": "S05", "protocol_name": "jilv_music_28_v2",
                "dataset_split": "test", "model_frozen_confirmed": True,
            }}
            (candidate / "session_config.json").write_text(json.dumps(config), encoding="utf-8")
            readiness = {
                "status": "passed", "protocol_required": "jilv_music_28_v2",
                "hdf5_sha256": sha256(signal),
                "checks": {
                    "hdf5_v3": True, "formal_protocol": True,
                    "sampling_rates": {"emg_nominal_hz": 250.0, "imu_nominal_hz": 112.0},
                    "valid_trial_counts": {f"still_{state}": 1 for state in freeze["required_hand_states"]},
                },
            }
            (candidate / "SESSION_COLLECTION_READINESS.json").write_text(json.dumps(readiness), encoding="utf-8")

            accepted = check_session(candidate, freeze_path=freeze_path, repo=root)
            self.assertTrue(accepted["eligible_for_prospective_scoring"])
            self.assertEqual(accepted["reasons"], [])

            model.write_text("modified", encoding="utf-8")
            self.assertIn("frozen_model_hash_mismatch",
                          check_session(candidate, freeze_path=freeze_path, repo=root)["reasons"])
            model.write_text('{"frozen":true}', encoding="utf-8")
            config["session"]["session_id"] = "S04"
            (candidate / "session_config.json").write_text(json.dumps(config), encoding="utf-8")
            rejected = check_session(candidate, freeze_path=freeze_path, repo=root)
            self.assertIn("previously_inspected_session_id", rejected["reasons"])
            self.assertIn("session_id_directory_mismatch", rejected["reasons"])


if __name__ == "__main__":
    unittest.main()
