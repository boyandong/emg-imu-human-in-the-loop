from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import BaselinePredictor
from .calibration import SessionCalibration
from .consistency import ConditionalConsistencyModel
from .runtime import HumanStateEstimator
from .training import load_neural_artifact


DEPLOYMENT_FORMAT_VERSION = 1


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_predictor(path: str | Path, kind: str):
    if kind == "baseline":
        return BaselinePredictor.load(path)
    if kind == "neural":
        return load_neural_artifact(path)
    raise ValueError("deployment model kind must be baseline or neural")


def _safe_member(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("deployment manifest path escapes its bundle")
    return candidate


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_freeze_receipt(
    receipt: dict[str, Any],
    predictor: Any,
    model_hash: str,
) -> None:
    if receipt.get("format_version") != 1 or receipt.get("status") != "formal_frozen":
        raise ValueError("freeze receipt format or status is invalid")
    for key in (
        "source_model_sha256", "frozen_model_sha256", "dataset_sha256",
        "validation_report_sha256",
    ):
        if not _is_sha256(receipt.get(key)):
            raise ValueError(f"freeze receipt has invalid {key}")
    if receipt["frozen_model_sha256"] != model_hash:
        raise ValueError("freeze receipt does not match the frozen model")
    metadata = dict(getattr(predictor, "metadata", {}) or {})
    expected = {
        "source_model_sha256": receipt["source_model_sha256"],
        "formal_dataset_sha256": receipt["dataset_sha256"],
        "validation_report_sha256": receipt["validation_report_sha256"],
        "freeze_protocol_version": 1,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("freeze receipt does not match frozen model provenance")


def _validate_validation_evidence(
    evidence: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if evidence.get("format_version") != 2 or evidence.get("split") != "validation":
        raise ValueError("bundled validation evidence format or split is invalid")
    if evidence.get("metric_weighting") != "equal_session_then_trial_v1":
        raise ValueError("bundled validation evidence uses obsolete metric weighting")
    if evidence.get("confusion_matrix_weighting") != "raw_window_counts":
        raise ValueError("bundled validation evidence lacks raw confusion counts")
    if evidence.get("model_sha256") != receipt["source_model_sha256"]:
        raise ValueError("bundled validation evidence does not match the source model")
    if evidence.get("dataset_sha256") != receipt["dataset_sha256"]:
        raise ValueError("bundled validation evidence does not match the formal dataset")
    if evidence.get("meets_v1_target") is not True:
        raise ValueError("bundled validation evidence does not meet the registered target")


def create_deployment_bundle(
    output: str | Path,
    *,
    model: str | Path,
    kind: str,
    freeze_receipt: str | Path,
    calibration: str | Path,
    consistency: str | Path | None = None,
    publish_legacy: bool = False,
    input_host: str = "127.0.0.1",
    input_port: int = 9100,
    output_host: str = "127.0.0.1",
    output_port: int = 9000,
) -> Path:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite deployment bundle: {destination}")
    model_path = Path(model).resolve()
    receipt_path = Path(freeze_receipt).resolve()
    calibration_path = Path(calibration).resolve()
    if not str(input_host).strip() or not str(output_host).strip():
        raise ValueError("deployment network hosts must not be empty")
    if (
        isinstance(input_port, bool) or not isinstance(input_port, int)
        or isinstance(output_port, bool) or not isinstance(output_port, int)
        or not 1 <= input_port <= 65535 or not 1 <= output_port <= 65535
    ):
        raise ValueError("deployment network ports must lie in 1..65535")
    predictor = _load_predictor(model_path, kind)
    if predictor.metadata.get("model_status") != "formal_frozen":
        raise ValueError("deployment requires a formal_frozen model")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model_hash = _sha256(model_path)
    _validate_freeze_receipt(receipt, predictor, model_hash)
    validation_path_value = receipt.get("validation_report_path")
    if not isinstance(validation_path_value, str) or not validation_path_value:
        raise ValueError("freeze receipt does not locate its validation evidence")
    validation_path = Path(validation_path_value).resolve()
    if not validation_path.is_file() or _sha256(validation_path) != receipt["validation_report_sha256"]:
        raise ValueError("validation evidence hash does not match the freeze receipt")
    validation_evidence = json.loads(validation_path.read_text(encoding="utf-8"))
    _validate_validation_evidence(validation_evidence, receipt)
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    session_calibration = SessionCalibration.from_dict(calibration_payload)
    # Constructor enforces model/calibration rate and contract compatibility.
    consistency_model = (
        ConditionalConsistencyModel.load(consistency) if consistency is not None else None
    )
    HumanStateEstimator(
        predictor, session_calibration, consistency_model=consistency_model,
    )

    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        model_name = "model.pkl" if kind == "baseline" else "model.pt"
        members: dict[str, dict[str, str]] = {}
        sources = {
            "model": (model_path, model_name),
            "freeze_receipt": (receipt_path, "freeze_receipt.json"),
            "calibration": (calibration_path, "calibration.json"),
            "validation_report": (validation_path, "validation_evidence.json"),
        }
        if consistency is not None:
            sources["consistency"] = (Path(consistency).resolve(), "consistency.json")
        for key, (source, name) in sources.items():
            target = staging / name
            shutil.copy2(source, target)
            members[key] = {"path": name, "sha256": _sha256(target)}
        manifest: dict[str, Any] = {
            "format_version": DEPLOYMENT_FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_kind": kind,
            "model_status": "formal_frozen",
            "runtime": {
                "config_version": 1,
                "input_host": str(input_host).strip(),
                "input_port": input_port,
                "output_host": str(output_host).strip(),
                "output_port": output_port,
                "publish_legacy": bool(publish_legacy),
            },
            "members": members,
        }
        manifest_path = staging / "deployment.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        verify_deployment_bundle(manifest_path)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "deployment.json"


def verify_deployment_bundle(manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != DEPLOYMENT_FORMAT_VERSION:
        raise ValueError("unsupported deployment format")
    if payload.get("model_status") != "formal_frozen":
        raise ValueError("deployment does not declare a formal_frozen model")
    runtime_payload = payload.get("runtime")
    if not isinstance(runtime_payload, dict):
        raise ValueError("deployment runtime config must be an object")
    runtime = dict(runtime_payload)
    expected_runtime = {
        "config_version", "input_host", "input_port", "output_host", "output_port",
        "publish_legacy",
    }
    if set(runtime) != expected_runtime or runtime.get("config_version") != 1:
        raise ValueError("deployment runtime config is missing, unknown, or unsupported")
    if (
        not isinstance(runtime["input_host"], str) or not runtime["input_host"].strip()
        or not isinstance(runtime["output_host"], str) or not runtime["output_host"].strip()
    ):
        raise ValueError("deployment runtime hosts must not be empty")
    if (
        isinstance(runtime["input_port"], bool) or not isinstance(runtime["input_port"], int)
        or isinstance(runtime["output_port"], bool) or not isinstance(runtime["output_port"], int)
        or not 1 <= runtime["input_port"] <= 65535
        or not 1 <= runtime["output_port"] <= 65535
    ):
        raise ValueError("deployment runtime ports must lie in 1..65535")
    if not isinstance(runtime["publish_legacy"], bool):
        raise ValueError("deployment publish_legacy must be boolean")
    members = dict(payload.get("members", {}))
    required = {"model", "freeze_receipt", "calibration", "validation_report"}
    allowed = required | {"consistency"}
    if not required.issubset(members) or not set(members).issubset(allowed):
        raise ValueError("deployment bundle has missing or unknown members")
    resolved: dict[str, Path] = {}
    for key, record in members.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"deployment member record is invalid: {key}")
        if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
            raise ValueError(f"deployment member record has invalid types: {key}")
        path = _safe_member(root, str(record["path"]))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"deployment member hash mismatch: {key}")
        resolved[key] = path
    kind = str(payload.get("model_kind"))
    predictor = _load_predictor(resolved["model"], kind)
    if predictor.metadata.get("model_status") != "formal_frozen":
        raise ValueError("bundled model metadata is not formal_frozen")
    receipt = json.loads(resolved["freeze_receipt"].read_text(encoding="utf-8"))
    _validate_freeze_receipt(receipt, predictor, members["model"]["sha256"])
    if receipt["validation_report_sha256"] != members["validation_report"]["sha256"]:
        raise ValueError("bundled validation evidence hash does not match receipt")
    evidence = json.loads(resolved["validation_report"].read_text(encoding="utf-8"))
    _validate_validation_evidence(evidence, receipt)
    calibration = SessionCalibration.from_dict(
        json.loads(resolved["calibration"].read_text(encoding="utf-8"))
    )
    consistency_model = (
        ConditionalConsistencyModel.load(resolved["consistency"])
        if "consistency" in resolved else None
    )
    HumanStateEstimator(predictor, calibration, consistency_model=consistency_model)
    return {
        "status": "ok",
        "manifest": str(manifest_path),
        "model_kind": kind,
        "model_sha256": members["model"]["sha256"],
        "calibration_sha256": members["calibration"]["sha256"],
        "runtime": runtime,
        "members": {key: str(value) for key, value in resolved.items()},
    }
