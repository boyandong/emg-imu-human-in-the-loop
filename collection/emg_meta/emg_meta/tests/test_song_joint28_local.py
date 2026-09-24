from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from emgforce.inference.song_joint28_local import SongJoint28WindowRuntime


BUNDLE = Path(__file__).resolve().parents[1] / "model_assets/song_joint28_window"


def test_tracked_bundle_predicts_finite_28_state_probabilities():
    runtime = SongJoint28WindowRuntime(BUNDLE)
    rng = np.random.default_rng(20260925)
    emg = rng.normal(size=(50, 8)).astype(np.float32)
    imu = rng.normal(size=(22, 6)).astype(np.float32)
    hand, arm, joint = runtime.predict_filtered_window(emg, imu)
    assert (hand.shape, arm.shape, joint.shape) == ((4,), (7,), (28,))
    assert np.isfinite(joint).all()
    np.testing.assert_allclose(joint.sum(), 1, atol=1e-12)
    for i, (a, h) in enumerate(runtime.joint_indices):
        np.testing.assert_allclose(joint[i], arm[a] * hand[h], atol=1e-12)
    with pytest.raises(ValueError, match="50×8"):
        runtime.predict_filtered_window(emg[:-1], imu)
    with pytest.raises(ValueError, match="22×6"):
        runtime.predict_filtered_window(emg, imu[:-1])


def test_bundle_rejects_tampered_model_and_invalid_label_order(tmp_path):
    directory = tmp_path / "bundle"
    shutil.copytree(BUNDLE, directory)
    artifact = directory / "song_joint28_model.json"
    manifest_path = directory / "song_joint28_manifest.json"
    with artifact.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        SongJoint28WindowRuntime(directory)
    model = json.loads(artifact.read_text(encoding="utf-8"))
    model["joint_indices"][0] = model["joint_indices"][1]
    artifact.write_text(json.dumps(model), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="label mapping"):
        SongJoint28WindowRuntime(directory)
