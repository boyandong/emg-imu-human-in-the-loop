from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baseline import BaselinePredictor, fit_lda_sanity_baselines, _macro_f1
from .calibration import SessionCalibration, fit_session_calibration
from .data import (
    build_feature_windows,
    build_raw_windows,
    dataset_report,
    discover_trials,
    hierarchical_window_weights,
    trials_for_split,
)
from .metrics import evaluate_predictions
from .metrics import first_stable_latency
from .training import (
    copy_neural_artifact_with_metadata,
    load_neural_artifact,
    save_neural_artifact,
    train_dual_branch,
)
from .simulate import create_smoke_dataset
from .runtime import HumanStateEstimator
from .consistency import ConditionalConsistencyModel
from .osc import OscPublisher
from .service import (
    LiveClassifierService,
    RawOscServer,
    RuntimeHealthMonitor,
    StateCsvLogger,
    write_json_atomic,
)
from .state import Direction, Gesture
from .datasets import check_benchmark_dataset, read_benchmark_report
from .datasets.adapters import ADAPTERS, get_adapter
from .hdf5_v3 import adapt_hdf5_v3, inspect_hdf5_v3
from .deployment import create_deployment_bundle, verify_deployment_bundle


def _calibration_directory(root: str | Path) -> Path:
    return Path(root) / "calibration"


def load_calibrations(root: str | Path) -> dict[str, SessionCalibration]:
    result: dict[str, SessionCalibration] = {}
    for session_id in ("1", "2", "3", "4"):
        path = _calibration_directory(root) / f"session_{session_id}.json"
        if path.exists():
            result[session_id] = SessionCalibration.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return result


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _dataset_fingerprint(root: str | Path, trials: list | None = None) -> str:
    dataset_root = Path(root).resolve()
    selected = trials if trials is not None else discover_trials(dataset_root)
    files = [trial.path.resolve() for trial in selected]
    files.extend(sorted((_calibration_directory(dataset_root)).glob("session_*.json")))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.as_posix()):
        relative = path.relative_to(dataset_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _runtime_manifest(
    *,
    run_id: str,
    status: str,
    args: argparse.Namespace,
    calibration: SessionCalibration,
    predictor: object,
    paths: dict[str, Path],
    started_at_utc: str,
    error: Exception | None = None,
) -> dict[str, object]:
    metadata = dict(getattr(predictor, "metadata", {}) or {})
    selected_metadata = {
        key: metadata[key]
        for key in (
            "artifact_format_version", "model_kind", "sample_rate_hz", "window_ms",
            "hop_ms", "calibration_contract", "session_adapter", "model_status",
        )
        if key in metadata
    }
    payload: dict[str, object] = {
        "format_version": 1,
        "run_id": run_id,
        "launch_mode": (
            "formal_bundle" if getattr(args, "deployment_manifest", None)
            else "development_loose_files"
        ),
        "status": status,
        "started_at_utc": started_at_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "kind": args.kind,
            "path": str(Path(args.model).resolve()),
            "sha256": _sha256(args.model),
            "metadata": selected_metadata,
            "formal_status": metadata.get("model_status", "legacy_unspecified"),
            "formal_claim_allowed": metadata.get("model_status") == "formal_frozen",
        },
        "calibration": {
            "path": str(Path(args.calibration).resolve()),
            "sha256": _sha256(args.calibration),
            "version": calibration.calibration_version,
            "emg_scale_mode": calibration.emg_scale_mode,
            "sample_rate_hz": calibration.sample_rate_hz,
        },
        "consistency": (
            {
                "path": str(Path(args.consistency).resolve()),
                "sha256": _sha256(args.consistency),
            }
            if args.consistency else None
        ),
        "deployment": (
            {
                "path": str(Path(args.deployment_manifest).resolve()),
                "sha256": _sha256(args.deployment_manifest),
            }
            if getattr(args, "deployment_manifest", None) else None
        ),
        "network": {
            "input_host": args.input_host,
            "input_port": args.input_port,
            "output_host": args.output_host,
            "output_port": args.output_port,
            "publish_legacy": args.publish_legacy,
        },
        "artifacts": {key: str(value.resolve()) for key, value in paths.items()},
    }
    if status in {"stopped", "failed"}:
        payload["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def cmd_validate(args: argparse.Namespace) -> int:
    report = dataset_report(discover_trials(args.dataset))
    calibrations = load_calibrations(args.dataset)
    report["missing_calibrations"] = [
        session for session in ("1", "2", "3", "4") if session not in calibrations
    ]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    has_missing_split = any(report["missing_combinations_by_split"].values())
    formal_failed = bool(args.formal and report["formal_collection_issues"])
    return 1 if report["missing_combinations"] or has_missing_split or report["missing_calibrations"] or formal_failed else 0


def cmd_make_smoke(args: argparse.Namespace) -> int:
    output = create_smoke_dataset(args.output, repetitions=args.repetitions)
    print(str(output))
    return 0


def cmd_fit_calibration(args: argparse.Namespace) -> int:
    source = np.load(args.input, allow_pickle=False)
    directions = {}
    for name in ("forward", "backward", "left", "right", "up", "down"):
        key = f"direction_{name}"
        if key in source.files:
            from .state import Direction
            directions[Direction[name.upper()]] = np.asarray(source[key]).reshape(3)
    reference = source["reference_emg_profile"] if "reference_emg_profile" in source.files else None
    calibration = fit_session_calibration(
        source["rest_emg"], source["gesture_emg"], source["rest_accel"], source["rest_gyro"],
        source["motion_accel"], source["motion_gyro"],
        observed_direction_vectors=directions or None,
        reference_emg_profile=reference,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return 0


def cmd_train_baseline(args: argparse.Namespace) -> int:
    trials = discover_trials(args.dataset)
    calibrations = load_calibrations(args.dataset)
    train = build_feature_windows(trials_for_split(trials, "train"), calibrations)
    validation = build_feature_windows(trials_for_split(trials, "validation"), calibrations)
    train_weights = hierarchical_window_weights(train.trial_id, train.session_id)
    validation_weights = hierarchical_window_weights(validation.trial_id, validation.session_id)
    predictor = BaselinePredictor().fit(
        train.emg, train.imu, train.direction, train.gesture,
        validation_emg_features=validation.emg,
        validation_imu_features=validation.imu,
        validation_direction=validation.direction,
        validation_gesture=validation.gesture,
        train_sample_weight=train_weights,
        validation_sample_weight=validation_weights,
    )
    used_sessions = {trial.session_id for trial in [*trials_for_split(trials, "train"), *trials_for_split(trials, "validation")]}
    predictor.metadata["calibration_contract"] = {
        "versions": sorted({calibrations[session].calibration_version for session in used_sessions}),
        "emg_scale_modes": sorted({calibrations[session].emg_scale_mode for session in used_sessions}),
    }
    predictor.metadata["model_status"] = "development_unvalidated"
    predictor.metadata["window_weighting"] = "equal_session_then_trial_v1"
    lda_gesture, lda_direction = fit_lda_sanity_baselines(
        train.emg, train.imu, train.gesture, train.direction,
    )
    lda_d = lda_direction.predict(validation.imu)
    lda_h = lda_gesture.predict(validation.emg)
    predictor.metadata["lda_validation"] = {
        "direction_macro_f1": _macro_f1(validation.direction, lda_d),
        "gesture_macro_f1": _macro_f1(validation.gesture, lda_h),
        "joint_accuracy": float(np.mean(
            (lda_d == validation.direction) & (lda_h == validation.gesture)
        )),
    }
    predictor.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "train_windows": len(train.emg),
        "validation_windows": len(validation.emg),
        "direction_threshold": predictor.direction_threshold,
        "gesture_threshold": predictor.gesture_threshold,
        "lda_validation": predictor.metadata["lda_validation"],
    }, indent=2))
    return 0


def cmd_adapt_baseline(args: argparse.Namespace) -> int:
    source_model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    if output == source_model:
        raise SystemExit("adapted model must use a different output path; the base artifact is immutable")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing adapted model: {output}")
    with np.load(args.input, allow_pickle=False) as values:
        required = {"emg_features", "imu_features", "direction", "gesture"}
        missing = sorted(required - set(values.files))
        if missing:
            raise SystemExit(f"session adapter input is missing: {', '.join(missing)}")
        predictor = BaselinePredictor.load(source_model)
        predictor.fit_bounded_session_adapter(
            values["emg_features"], values["imu_features"],
            values["direction"], values["gesture"],
            max_abs_bias=args.max_abs_bias,
            min_samples_per_class=args.min_samples_per_class,
        )
    predictor.save(output)
    print(json.dumps({
        "output": str(output),
        "base_model": str(source_model),
        "base_model_sha256": predictor.metadata["session_adapter"]["base_artifact_sha256"],
        "direction_bias": predictor.direction_logit_bias.tolist(),
        "gesture_bias": predictor.gesture_logit_bias.tolist(),
    }, indent=2))
    return 0


def cmd_train_neural(args: argparse.Namespace) -> int:
    trials = discover_trials(args.dataset)
    calibrations = load_calibrations(args.dataset)
    train = build_raw_windows(trials_for_split(trials, "train"), calibrations)
    validation = build_raw_windows(trials_for_split(trials, "validation"), calibrations)
    result = train_dual_branch(
        train, validation, device=args.device, max_epochs=args.max_epochs,
        patience=args.patience, batch_size=args.batch_size,
    )
    used_sessions = {trial.session_id for trial in [*trials_for_split(trials, "train"), *trials_for_split(trials, "validation")]}
    result.predictor.metadata.update({
        "artifact_format_version": 1,
        "model_kind": "dual_branch_causal_tcn",
        "sample_rate_hz": 200,
        "window_ms": 200,
        "hop_ms": 40,
        "calibration_contract": {
            "versions": sorted({calibrations[session].calibration_version for session in used_sessions}),
            "emg_scale_modes": sorted({calibrations[session].emg_scale_mode for session in used_sessions}),
        },
        "model_status": "development_unvalidated",
        "classes": {"direction": list(range(7)), "gesture": list(range(4))},
    })
    save_neural_artifact(result, args.output)
    print(json.dumps({
        "output": str(args.output),
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
    }, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("Session 4 is locked. Re-run once with --unlock-test only after freezing the model.")
    test_record = Path(args.dataset) / "SESSION4_FINAL_RESULT.json"
    if args.split == "test" and test_record.exists():
        raise SystemExit(f"Session 4 was already evaluated; locked result: {test_record}")
    kind = getattr(args, "kind", "baseline")
    predictor = BaselinePredictor.load(args.model) if kind == "baseline" else load_neural_artifact(args.model)
    trials = discover_trials(args.dataset)
    selected_trials = trials_for_split(trials, args.split)
    calibrations = load_calibrations(args.dataset)
    windows = (
        build_feature_windows(selected_trials, calibrations)
        if kind == "baseline" else build_raw_windows(selected_trials, calibrations)
    )
    d_pred: list[int] = []
    h_pred: list[int] = []
    q_d: list[float] = []
    q_h: list[float] = []
    for emg, imu in zip(windows.emg, windows.imu):
        prediction = predictor.predict_features(emg, imu) if kind == "baseline" else predictor.predict(emg, imu)
        d_pred.append(int(prediction.direction)); h_pred.append(int(prediction.gesture))
        q_d.append(prediction.q_direction); q_h.append(prediction.q_gesture)
    latencies: list[float] = []
    for trial in selected_trials:
        if "arm_onset_ms" not in trial.events and "hand_onset_ms" not in trial.events:
            continue
        estimator = HumanStateEstimator(predictor, calibrations[trial.session_id])
        states = estimator.push_batch(
            trial.timestamp_ms, trial.emg, trial.accel, trial.gyro,
            window_quality=trial.window_quality,
            missing_mask=trial.missing_mask,
            channel_quality=trial.channel_quality,
            timestamp_valid=trial.timestamp_valid,
            imu_valid=trial.imu_valid,
            interpolated_imu=trial.interpolated_imu,
            quality_gate_pass=trial.quality_gate_pass,
        )
        timestamps = np.asarray([state.timestamp_ms for state in states])
        if "arm_onset_ms" in trial.events:
            after = np.flatnonzero(trial.timestamp_ms >= trial.events["arm_onset_ms"])
            if len(after):
                target = int(trial.direction[after[0]])
                if target != int(Direction.NONE) and target != int(Direction.UNKNOWN):
                    latency = first_stable_latency(
                        trial.events["arm_onset_ms"], timestamps,
                        np.asarray([int(state.direction) for state in states]), target,
                    )
                    if latency is not None: latencies.append(latency)
        if "hand_onset_ms" in trial.events:
            after = np.flatnonzero(trial.timestamp_ms >= trial.events["hand_onset_ms"])
            if len(after):
                target = int(trial.gesture[after[0]])
                if target != int(Gesture.NEUTRAL) and target != int(Gesture.UNKNOWN):
                    latency = first_stable_latency(
                        trial.events["hand_onset_ms"], timestamps,
                        np.asarray([int(state.gesture) for state in states]), target,
                    )
                    if latency is not None: latencies.append(latency)
    result = evaluate_predictions(
        windows.direction, windows.gesture, np.asarray(d_pred), np.asarray(h_pred),
        np.asarray(q_d), np.asarray(q_h),
        latency_ms=np.asarray(latencies) if latencies else None,
        sample_weight=hierarchical_window_weights(windows.trial_id, windows.session_id),
    )
    result_payload = asdict(result)
    result_payload["format_version"] = 2
    result_payload["metric_weighting"] = "equal_session_then_trial_v1"
    result_payload["confusion_matrix_weighting"] = "raw_window_counts"
    result_payload["model_status"] = getattr(predictor, "metadata", {}).get(
        "model_status", "legacy_unspecified",
    )
    model_path = Path(args.model)
    result_payload.update({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": _dataset_fingerprint(args.dataset, trials),
        "split": args.split,
        "meets_v1_target": result.meets_v1_target,
    })
    if args.split == "test":
        write_json_atomic(test_record, result_payload)
    output = getattr(args, "output", None)
    if output:
        destination = Path(output)
        if destination.exists():
            raise SystemExit(f"refusing to overwrite evaluation report: {destination}")
        write_json_atomic(destination, result_payload)
    print(json.dumps(result_payload, indent=2))
    return 0


def _deployment_readiness(args: argparse.Namespace) -> dict[str, object]:
    trials = discover_trials(args.dataset)
    report = dataset_report(trials)
    calibrations = load_calibrations(args.dataset)
    issues = list(report["formal_collection_issues"])
    if report["missing_combinations"]:
        issues.append(f"dataset is missing {len(report['missing_combinations'])} D/H combinations")
    missing_calibrations = [value for value in ("1", "2", "3", "4") if value not in calibrations]
    if missing_calibrations:
        issues.append(f"missing calibrations: {missing_calibrations}")
    kind = getattr(args, "kind", "baseline")
    predictor = BaselinePredictor.load(args.model) if kind == "baseline" else load_neural_artifact(args.model)
    metadata = dict(getattr(predictor, "metadata", {}) or {})
    model_hash = _sha256(args.model)
    dataset_hash = _dataset_fingerprint(args.dataset, trials)
    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    if validation.get("format_version") != 2:
        issues.append("validation report format is obsolete or unsupported")
    if validation.get("split") != "validation":
        issues.append("evidence report is not a validation split")
    if validation.get("metric_weighting") != "equal_session_then_trial_v1":
        issues.append("validation report does not use session/trial-balanced metrics")
    if validation.get("confusion_matrix_weighting") != "raw_window_counts":
        issues.append("validation report does not preserve raw confusion-matrix counts")
    if validation.get("model_sha256") != model_hash:
        issues.append("validation report model hash does not match")
    if validation.get("dataset_sha256") != dataset_hash:
        issues.append("validation report dataset hash does not match")
    if not validation.get("meets_v1_target", False):
        issues.append("validation report does not meet the registered v1 target")
    if metadata.get("model_status") not in {"development_unvalidated", "formal_frozen"}:
        issues.append("model has legacy/unspecified provenance")
    if metadata.get("window_weighting") != "equal_session_then_trial_v1":
        issues.append("model was not trained with session/trial-balanced windows")
    classes = metadata.get("classes", {})
    if (
        set(classes.get("direction", [])) != set(range(7))
        or set(classes.get("gesture", [])) != set(range(4))
    ):
        issues.append("model does not contain all 7 direction and 4 gesture classes")
    payload = {
        "format_version": 2,
        "status": "ready_to_freeze" if not issues else "blocked",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "model_status": metadata.get("model_status", "legacy_unspecified"),
        "formal_trial_count": report["trial_count"],
        "validation_report": str(Path(args.validation_report).resolve()),
        "issues": issues,
    }
    return payload


def cmd_deployment_check(args: argparse.Namespace) -> int:
    payload = _deployment_readiness(args)
    if getattr(args, "output", None):
        output = Path(args.output)
        if output.exists():
            raise SystemExit(f"refusing to overwrite deployment readiness report: {output}")
        write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "ready_to_freeze" else 1


def cmd_freeze_model(args: argparse.Namespace) -> int:
    source = Path(args.model).resolve()
    destination = Path(args.output).resolve()
    if source == destination:
        raise SystemExit("frozen model must use a different output path")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite frozen model: {destination}")
    receipt_path = destination.with_suffix(destination.suffix + ".freeze.json")
    if receipt_path.exists():
        raise SystemExit(f"refusing to overwrite freeze receipt: {receipt_path}")
    readiness = _deployment_readiness(args)
    if readiness["status"] != "ready_to_freeze":
        print(json.dumps(readiness, indent=2))
        return 1
    validation_path = Path(args.validation_report).resolve()
    frozen_at = datetime.now(timezone.utc).isoformat()
    updates = {
        "model_status": "formal_frozen",
        "frozen_at_utc": frozen_at,
        "source_model_sha256": readiness["model_sha256"],
        "formal_dataset_sha256": readiness["dataset_sha256"],
        "validation_report_sha256": _sha256(validation_path),
        "freeze_protocol_version": 1,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_model = destination.with_name(f".{destination.name}.{os.getpid()}.freeze.tmp")
    staged_receipt = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.freeze.tmp")
    try:
        if args.kind == "baseline":
            predictor = BaselinePredictor.load(source)
            predictor.metadata.pop("loaded_artifact_sha256", None)
            predictor.metadata.update(updates)
            predictor.save(staged_model)
            verified = BaselinePredictor.load(staged_model)
        else:
            copy_neural_artifact_with_metadata(source, staged_model, updates)
            verified = load_neural_artifact(staged_model)
        if verified.metadata.get("model_status") != "formal_frozen":
            raise RuntimeError("frozen artifact failed metadata readback")
        receipt = {
            "format_version": 1,
            "status": "formal_frozen",
            "frozen_at_utc": frozen_at,
            "source_model_sha256": readiness["model_sha256"],
            "frozen_model_path": str(destination),
            "frozen_model_sha256": _sha256(staged_model),
            "dataset_sha256": readiness["dataset_sha256"],
            "validation_report_path": str(validation_path),
            "validation_report_sha256": _sha256(validation_path),
        }
        write_json_atomic(staged_receipt, receipt)
        os.replace(staged_model, destination)
        try:
            os.replace(staged_receipt, receipt_path)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    finally:
        staged_model.unlink(missing_ok=True)
        staged_receipt.unlink(missing_ok=True)
    print(json.dumps(receipt, indent=2))
    return 0


def cmd_package_deployment(args: argparse.Namespace) -> int:
    manifest = create_deployment_bundle(
        args.output, model=args.model, kind=args.kind,
        freeze_receipt=args.freeze_receipt, calibration=args.calibration,
        consistency=args.consistency, publish_legacy=args.publish_legacy,
        input_host=args.input_host, input_port=args.input_port,
        output_host=args.output_host, output_port=args.output_port,
    )
    print(json.dumps(verify_deployment_bundle(manifest), indent=2))
    return 0


def cmd_verify_deployment(args: argparse.Namespace) -> int:
    print(json.dumps(verify_deployment_bundle(args.manifest), indent=2))
    return 0


def cmd_serve_deployment(args: argparse.Namespace) -> int:
    verified = verify_deployment_bundle(args.manifest)
    runtime = verified["runtime"]
    members = verified["members"]
    serve_args = argparse.Namespace(
        model=members["model"], kind=verified["model_kind"],
        calibration=members["calibration"], consistency=members.get("consistency"),
        device=args.device,
        input_host=runtime["input_host"], input_port=runtime["input_port"],
        output_host=runtime["output_host"], output_port=runtime["output_port"],
        publish_legacy=runtime["publish_legacy"],
        audit_dir=args.audit_dir, run_id=args.run_id,
        health_json=None, run_manifest=None, log_csv=None,
        audit_fsync_every=args.audit_fsync_every,
        deployment_manifest=str(Path(args.manifest).resolve()),
    )
    return cmd_serve(serve_args)


def cmd_serve(args: argparse.Namespace) -> int:
    calibration = SessionCalibration.from_dict(json.loads(Path(args.calibration).read_text(encoding="utf-8")))
    predictor = (
        BaselinePredictor.load(args.model)
        if args.kind == "baseline" else load_neural_artifact(args.model, device=args.device)
    )
    consistency = ConditionalConsistencyModel.load(args.consistency) if args.consistency else None
    estimator = HumanStateEstimator(predictor, calibration, consistency_model=consistency)
    run_id = getattr(args, "run_id", None) or _runtime_run_id()
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise SystemExit("run-id must be a single safe path component")
    if getattr(args, "audit_fsync_every", 25) < 1:
        raise SystemExit("audit-fsync-every must be positive")
    run_directory = Path(getattr(args, "audit_dir", "artifacts/runtime")) / run_id
    uses_run_directory = not all(
        getattr(args, name, None) for name in ("log_csv", "health_json", "run_manifest")
    )
    if uses_run_directory and run_directory.exists():
        raise SystemExit(f"refusing to reuse existing runtime audit directory: {run_directory}")
    paths = {
        "state_csv": Path(args.log_csv) if args.log_csv else run_directory / "states.csv",
        "health_json": (
            Path(args.health_json) if getattr(args, "health_json", None)
            else run_directory / "health.json"
        ),
        "manifest_json": (
            Path(args.run_manifest) if getattr(args, "run_manifest", None)
            else run_directory / "manifest.json"
        ),
    }
    publisher = OscPublisher(args.output_host, args.output_port, publish_legacy=args.publish_legacy)
    logger = StateCsvLogger(
        paths["state_csv"], run_id=run_id,
        fsync_every=getattr(args, "audit_fsync_every", 25),
    )
    health = RuntimeHealthMonitor(paths["health_json"], run_id=run_id)
    service = LiveClassifierService(estimator, publisher, logger, health)
    server = RawOscServer(
        service.accept, args.input_host, args.input_port,
        error_callback=lambda error: health.observe_error(error, parse=True),
    )
    started_at_utc = datetime.now(timezone.utc).isoformat()
    write_json_atomic(paths["manifest_json"], _runtime_manifest(
        run_id=run_id, status="running", args=args, calibration=calibration,
        predictor=predictor, paths=paths, started_at_utc=started_at_utc,
    ))
    print(f"raw OSC {args.input_host}:{args.input_port} -> state OSC {args.output_host}:{args.output_port}")
    print(f"runtime audit: {run_directory.resolve()}")
    terminal_status = "stopped"
    terminal_error: Exception | None = None
    try:
        server.run_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        terminal_status = "failed"
        terminal_error = exc
        raise
    finally:
        publisher.close()
        logger.close()
        health.close(status=terminal_status)
        write_json_atomic(paths["manifest_json"], _runtime_manifest(
            run_id=run_id, status=terminal_status, args=args, calibration=calibration,
            predictor=predictor, paths=paths, started_at_utc=started_at_utc,
            error=terminal_error,
        ))
    return 0


def cmd_fit_consistency(args: argparse.Namespace) -> int:
    records = []
    with Path(args.input).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("onset_lag_ms", "").strip():
                continue
            key = tuple(int(row[name]) for name in ("direction", "gesture", "arm_phase", "hand_phase"))
            records.append((key, float(row["activation"]), float(row["motion"]), float(row["onset_lag_ms"])))
    model = ConditionalConsistencyModel(min_samples=args.min_samples).fit(records)
    model.save(args.output)
    print(json.dumps({"conditions": len(model.stats), "output": args.output}, indent=2))
    return 0


def cmd_benchmark_adapt(args: argparse.Namespace) -> int:
    result = get_adapter(args.adapter).adapt(args.source, args.output)
    print(json.dumps({
        "dataset_id": result.dataset_id,
        "output": str(result.output_root),
        "manifest": str(result.manifest_path),
        "trial_count": result.trial_count,
        "warnings": list(result.warnings),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark_check(args: argparse.Namespace) -> int:
    report = check_benchmark_dataset(args.dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


def cmd_benchmark_report(args: argparse.Namespace) -> int:
    print(json.dumps(read_benchmark_report(args.dataset), indent=2, ensure_ascii=False))
    return 0


def cmd_hdf5_v3_check(args: argparse.Namespace) -> int:
    report = inspect_hdf5_v3(args.source)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


def cmd_hdf5_v3_adapt(args: argparse.Namespace) -> int:
    result = adapt_hdf5_v3(args.source, args.output)
    print(json.dumps({
        "output": str(result.output_root),
        "session_count": result.session_count,
        "trial_count": result.trial_count,
        "warnings": list(result.warnings),
    }, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMG-IMU human-state tools")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("make-smoke-dataset")
    smoke.add_argument("output")
    smoke.add_argument("--repetitions", type=int, default=1)
    smoke.set_defaults(func=cmd_make_smoke)
    validate = commands.add_parser("validate-dataset")
    validate.add_argument("dataset")
    validate.add_argument("--formal", action="store_true", help="require at least 3 trials per state in every session")
    validate.set_defaults(func=cmd_validate)
    calibration = commands.add_parser("fit-calibration")
    calibration.add_argument("input")
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=cmd_fit_calibration)
    baseline = commands.add_parser("train-baseline")
    baseline.add_argument("dataset")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(func=cmd_train_baseline)
    adapt_baseline = commands.add_parser(
        "adapt-baseline", help="fit a bounded labeled-session output adapter without changing the SVMs",
    )
    adapt_baseline.add_argument("--model", required=True)
    adapt_baseline.add_argument("--input", required=True, help="NPZ with EMG/IMU features and D/H labels")
    adapt_baseline.add_argument("--output", required=True)
    adapt_baseline.add_argument("--max-abs-bias", type=float, default=0.35)
    adapt_baseline.add_argument("--min-samples-per-class", type=int, default=2)
    adapt_baseline.set_defaults(func=cmd_adapt_baseline)
    neural = commands.add_parser("train-neural")
    neural.add_argument("dataset")
    neural.add_argument("--output", required=True)
    neural.add_argument("--device", default="cpu")
    neural.add_argument("--max-epochs", type=int, default=100)
    neural.add_argument("--patience", type=int, default=15)
    neural.add_argument("--batch-size", type=int, default=128)
    neural.set_defaults(func=cmd_train_neural)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--unlock-test", action="store_true")
    evaluate.add_argument("--output", help="write a hash-bound evaluation report without overwriting")
    evaluate.set_defaults(func=cmd_evaluate)
    deployment = commands.add_parser(
        "deployment-check", help="verify formal data and hash-bound validation evidence before model freeze",
    )
    deployment.add_argument("dataset")
    deployment.add_argument("--model", required=True)
    deployment.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    deployment.add_argument("--validation-report", required=True)
    deployment.add_argument("--output", help="write the machine-readable readiness report")
    deployment.set_defaults(func=cmd_deployment_check)
    freeze = commands.add_parser(
        "freeze-model", help="re-run deployment gates and write a separate formal-frozen artifact",
    )
    freeze.add_argument("dataset")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    freeze.add_argument("--validation-report", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=cmd_freeze_model)
    package = commands.add_parser(
        "package-deployment", help="create a hash-verified bundle from a formal-frozen model",
    )
    package.add_argument("--model", required=True)
    package.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    package.add_argument("--freeze-receipt", required=True)
    package.add_argument("--calibration", required=True)
    package.add_argument("--consistency")
    package.add_argument("--publish-legacy", action="store_true")
    package.add_argument("--input-host", default="127.0.0.1")
    package.add_argument("--input-port", type=int, default=9100)
    package.add_argument("--output-host", default="127.0.0.1")
    package.add_argument("--output-port", type=int, default=9000)
    package.add_argument("--output", required=True)
    package.set_defaults(func=cmd_package_deployment)
    verify_package = commands.add_parser(
        "verify-deployment", help="verify every member and contract in a deployment bundle",
    )
    verify_package.add_argument("manifest")
    verify_package.set_defaults(func=cmd_verify_deployment)
    serve_package = commands.add_parser(
        "serve-deployment", help="verify a formal bundle and serve its fixed runtime config",
    )
    serve_package.add_argument("manifest")
    serve_package.add_argument("--device", default="cpu")
    serve_package.add_argument("--audit-dir", default="artifacts/runtime")
    serve_package.add_argument("--run-id")
    serve_package.add_argument("--audit-fsync-every", type=int, default=25)
    serve_package.set_defaults(func=cmd_serve_deployment)
    serve = commands.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--kind", choices=("baseline", "neural"), default="baseline")
    serve.add_argument("--calibration", required=True)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--input-host", default="127.0.0.1")
    serve.add_argument("--input-port", type=int, default=9100)
    serve.add_argument("--output-host", default="127.0.0.1")
    serve.add_argument("--output-port", type=int, default=9000)
    serve.add_argument("--publish-legacy", action="store_true")
    serve.add_argument("--consistency", help="optional fitted shadow consistency JSON")
    serve.add_argument("--log-csv", help="append runtime state and onset-lag observations")
    serve.add_argument("--audit-dir", default="artifacts/runtime", help="root for per-run audit bundles")
    serve.add_argument("--run-id", help="explicit unique audit run identifier")
    serve.add_argument("--health-json", help="override the per-run atomic health snapshot path")
    serve.add_argument("--run-manifest", help="override the immutable-input run manifest path")
    serve.add_argument("--audit-fsync-every", type=int, default=25, help="durably sync CSV every N states")
    serve.set_defaults(func=cmd_serve)
    consistency = commands.add_parser("fit-consistency")
    consistency.add_argument("input", help="CSV of labeled A/M/onset-lag observations")
    consistency.add_argument("--output", required=True)
    consistency.add_argument("--min-samples", type=int, default=12)
    consistency.set_defaults(func=cmd_fit_consistency)
    benchmark_adapt = commands.add_parser(
        "benchmark-adapt", help="convert a public dataset to the canonical benchmark format",
    )
    benchmark_adapt.add_argument("adapter", choices=sorted(ADAPTERS))
    benchmark_adapt.add_argument("source")
    benchmark_adapt.add_argument("--output", required=True)
    benchmark_adapt.set_defaults(func=cmd_benchmark_adapt)
    benchmark_check = commands.add_parser(
        "benchmark-check", help="validate a converted public benchmark",
    )
    benchmark_check.add_argument("dataset")
    benchmark_check.set_defaults(func=cmd_benchmark_check)
    benchmark_report = commands.add_parser(
        "benchmark-report", help="print a converted benchmark's manifest and reports",
    )
    benchmark_report.add_argument("dataset")
    benchmark_report.set_defaults(func=cmd_benchmark_report)
    hdf5_check = commands.add_parser(
        "hdf5-v3-check", help="validate one self-collected HDF5 3.0 session",
    )
    hdf5_check.add_argument("source")
    hdf5_check.set_defaults(func=cmd_hdf5_v3_check)
    hdf5_adapt = commands.add_parser(
        "hdf5-v3-adapt", help="convert HDF5 3.0 sessions to the formal classifier dataset",
    )
    hdf5_adapt.add_argument("source")
    hdf5_adapt.add_argument("--output", required=True)
    hdf5_adapt.set_defaults(func=cmd_hdf5_v3_adapt)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
