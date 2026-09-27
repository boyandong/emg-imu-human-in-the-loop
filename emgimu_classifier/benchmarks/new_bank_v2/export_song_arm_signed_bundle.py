"""Export the frozen signed-axis candidate as a separate opt-in local bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.new_bank_v2.song_arm_signed_continuous import MODEL_FILE, ROOT


REPO = ROOT.parents[2]
ASSETS = REPO / "collection/emg_meta/emg_meta/model_assets"
BASE = ASSETS / "song_joint28_window"
OUTPUT = ASSETS / "song_joint28_signed"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path = OUTPUT) -> dict:
    base_manifest = json.loads((BASE / "song_joint28_manifest.json").read_text(encoding="utf-8"))
    base_model_path = BASE / "song_joint28_model.json"
    if _sha(base_model_path) != base_manifest["sha256"]:
        raise ValueError("tracked baseline model digest changed")
    arm = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    continuous_path = ROOT / "SONG_ARM_SIGNED_CONTINUOUS_RESULTS.json"
    continuous = json.loads(continuous_path.read_text(encoding="utf-8"))
    if (_sha(MODEL_FILE) != continuous["candidate_arm_model_sha256"]
            or _sha(base_model_path) != continuous["baseline_model_sha256"]
            or arm["source_hdf5_sha256"] != base_manifest["source_hdf5_sha256"]):
        raise ValueError("signed candidate is not bound to the frozen baseline/source")
    model = json.loads(base_model_path.read_text(encoding="utf-8"))
    if (model["model_kind"] != "song_28_source_factorized_f0v2_f2a_f3c_f6"
            or arm["classes"] != model["arm_classes"]):
        raise ValueError("baseline/candidate arm class mismatch")
    model["model_kind"] = "song_28_source_factorized_f0v2_f2a_f3c_f6_signed"
    model["arm_feature_kind"] = "f6_signed_19"
    model["arm"] = {key: arm[key] for key in
                    ("scaler_mean", "scaler_scale", "coef", "intercept")}
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "song_joint28_model.json"
    model_path.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"format_version": 1, "artifact": model_path.name, "sha256": _sha(model_path),
                "model_status": "exploratory_one_person_one_day_offline_only",
                "source_hdf5_sha256": arm["source_hdf5_sha256"],
                "reference_baseline_model_sha256": _sha(base_model_path),
                "reference_signed_arm_model_sha256": _sha(MODEL_FILE),
                "reference_continuous_results_sha256": _sha(continuous_path),
                "training_runtime": base_manifest["training_runtime"],
                "limitation": "One participant/day, cue-labelled continuous software replay; no new-device-session, measured live accuracy or end-to-end latency."}
    manifest_path = output / "song_joint28_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"model_sha256": _sha(model_path), "manifest_sha256": _sha(manifest_path),
            "output": str(output)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
