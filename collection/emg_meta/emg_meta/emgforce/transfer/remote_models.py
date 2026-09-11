from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from emgforce.algorithms import META_CONV_LSTM, PERSONAL_MPF_TDS, get_algorithm
from emgforce.inference.export_bundle import create_bundle_from_checkpoint

from .dataset_upload import CommandResult, run_command, validate_server_settings


@dataclass(frozen=True, slots=True)
class RemoteModelInfo:
    model_id: str
    created_at: str
    checkpoint_path: str
    config_path: str
    run_dir: str
    epoch: int
    step: int
    size_bytes: int
    val_accuracy: float | None = None
    test_cler: float | None = None
    algorithm_id: str = META_CONV_LSTM
    val_event_macro_f1: float | None = None
    test_event_macro_f1: float | None = None
    val_mean_fnr: float | None = None
    test_mean_fnr: float | None = None


CommandRunner = Callable[[list[str], float], CommandResult]


def list_remote_models(
    host: str,
    repository: str = "/home/qxy/qxy/generic-neuromotor-interface",
    python: str = "/home/qxy/miniconda3/envs/neuromotor/bin/python3.12",
    *,
    runner: CommandRunner | None = None,
    algorithm_id: str = META_CONV_LSTM,
) -> list[RemoteModelInfo]:
    get_algorithm(algorithm_id)
    host, repository = _validate_remote(host, repository)
    python = _safe_absolute(python, "Python")
    run = runner or run_command
    if algorithm_id == PERSONAL_MPF_TDS:
        script = r'''
import json
from pathlib import Path
repo = Path(__import__("sys").argv[1])
for manifest_path in repo.glob("logs/*/*/bundle/manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest_path.parent / manifest["artifact"]["filename"]
    training = manifest.get("training", {})
    run_dir = manifest_path.parents[1]
    payload = {"model_id": manifest["model_id"], "created_at": f"{run_dir.parent.name} {run_dir.name.replace('-', ':')}",
      "checkpoint_path": str(artifact), "config_path": str(manifest_path), "run_dir": str(run_dir),
      "epoch": int(training.get("best_epoch", 0)), "step": 0, "size_bytes": artifact.stat().st_size, "val_accuracy": None,
      "test_cler": None, "algorithm_id": "personal_mpf_tds_v1",
      "val_event_macro_f1": training.get("val_event_macro_f1"),
      "test_event_macro_f1": training.get("test_event_macro_f1"),
      "val_mean_fnr": training.get("val_mean_fnr"), "test_mean_fnr": training.get("test_mean_fnr")}
    print(json.dumps(payload, ensure_ascii=False))
'''
    else:
        script = r'''
import json, re
from pathlib import Path

repo = Path(__import__("sys").argv[1])
for checkpoint in repo.glob("logs/*/*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"):
    run_dir = checkpoint.parents[3]
    log_path = run_dir / "discrete-gestures.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", checkpoint.name)
    if match is None:
        continue
    val_matches = re.findall(r"(?:best_checkpoint_score'|val_accuracy'):\s*([0-9.eE+-]+)", text)
    test_matches = re.findall(r"test_cler':\s*([0-9.eE+-]+)", text)
    created = f"{run_dir.parent.name} {run_dir.name.replace('-', ':')}"
    stamp = run_dir.parent.name.replace('-', '') + "_" + run_dir.name.replace('-', '')
    payload = {
        "model_id": f"emg_{stamp}_e{match.group(1)}",
        "created_at": created,
        "checkpoint_path": str(checkpoint),
        "config_path": str(run_dir / "hydra_configs" / "config.yaml"),
        "run_dir": str(run_dir),
        "epoch": int(match.group(1)),
        "step": int(match.group(2)),
        "size_bytes": checkpoint.stat().st_size,
        "val_accuracy": float(val_matches[-1]) if val_matches else None,
        "test_cler": float(test_matches[-1]) if test_matches else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
'''
    command = f"{shlex.quote(python)} -c {shlex.quote(script)} {shlex.quote(repository)}"
    result = run(_ssh_prefix() + [host, command], 30.0)
    _ensure_success(result, "无法读取服务器模型列表")
    models: list[RemoteModelInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            models.append(RemoteModelInfo(**payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"服务器模型列表格式错误：{line}") from exc
    return sorted(models, key=lambda item: item.created_at, reverse=True)


def download_remote_model(
    model: RemoteModelInfo,
    host: str,
    models_root: Path,
    repository: str = "/home/qxy/qxy/generic-neuromotor-interface",
    conda_init: str = "/home/qxy/miniconda3/etc/profile.d/conda.sh",
    conda_env: str = "neuromotor",
    python: str = "/home/qxy/miniconda3/envs/neuromotor/bin/python3.12",
    *,
    runner: CommandRunner | None = None,
) -> Path:
    host, repository = _validate_remote(host, repository)
    # Kept in the public signature for compatibility with saved UI settings.
    # Native checkpoints do not need a server-side Python export step.
    _safe_absolute(conda_init, "Conda 初始化脚本")
    _safe_absolute(python, "Python")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", conda_env):
        raise ValueError("Conda 环境名称格式不正确")
    _validate_model(model, repository)
    run = runner or run_command
    root = Path(models_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / model.model_id
    verify_command = (
        f"test -f {shlex.quote(model.checkpoint_path)} && "
        f"test -f {shlex.quote(model.config_path)} && "
        f"sha256sum {shlex.quote(model.checkpoint_path)}"
    )
    verify_result = run(_ssh_prefix() + [host, verify_command], 30.0)
    _ensure_success(verify_result, "服务器 CKPT 校验失败")
    remote_hash = verify_result.stdout.strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", remote_hash):
        raise RuntimeError("服务器没有返回有效的 CKPT SHA-256")

    with tempfile.TemporaryDirectory(prefix="emgforce-model-", dir=root) as temporary:
        temp_root = Path(temporary)
        if model.algorithm_id == PERSONAL_MPF_TDS:
            bundle = temp_root / model.model_id
            bundle.mkdir()
            remote_bundle = str(PurePosixPath(model.config_path).parent)
            for filename in ("manifest.json", "labels.json", "preprocessing.json",
                             PurePosixPath(model.checkpoint_path).name):
                result = run(_scp_prefix() + [f"{host}:{remote_bundle}/{filename}", str(bundle / filename)], 900.0)
                _ensure_success(result, f"MPF+TDS 模型文件下载失败：{filename}")
            from emgforce.inference.model_bundle import load_model_bundle
            load_model_bundle(bundle, verify_hash=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(bundle), str(destination))
            return destination
        artifact = temp_root / f"{model.model_id}.ckpt"
        config = temp_root / "training_config.yaml"
        for remote, local in ((model.checkpoint_path, artifact), (model.config_path, config)):
            result = run(_scp_prefix() + [f"{host}:{remote}", str(local)], 900.0)
            _ensure_success(result, f"模型文件下载失败：{PurePosixPath(remote).name}")
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != remote_hash:
            raise RuntimeError("下载后的 CKPT SHA-256 与服务器不一致")
        bundle = create_bundle_from_checkpoint(
            artifact, config_path=config, output_dir=temp_root,
            model_id=model.model_id, channels=8, sample_rate=2000,
        )
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["training"] = {
            "val_accuracy": model.val_accuracy,
            "test_cler": model.test_cler,
            "trained_at": model.created_at,
            "remote_run_dir": model.run_dir,
        }
        manifest["source_checkpoint"] = {
            "path": model.checkpoint_path,
            "epoch": model.epoch,
            "step": model.step,
            "sha256": remote_hash,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # The same remote run may previously have been downloaded as TorchScript.
        # Only replace it after the complete CKPT bundle has passed verification.
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(bundle), str(destination))
    return destination


def _validate_remote(host: str, repository: str) -> tuple[str, str]:
    host, _ = validate_server_settings(host, "/tmp/emgforce")
    return host, _safe_absolute(repository, "训练仓库")


def _validate_model(model: RemoteModelInfo, repository: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", model.model_id):
        raise ValueError("模型 ID 格式不安全")
    get_algorithm(model.algorithm_id)
    logs_root = PurePosixPath(repository) / "logs"
    for value, title in (
        (model.checkpoint_path, "checkpoint"),
        (model.config_path, "训练配置"),
        (model.run_dir, "训练目录"),
    ):
        path = PurePosixPath(_safe_absolute(value, title))
        try:
            path.relative_to(logs_root)
        except ValueError as exc:
            raise ValueError(f"{title}不在服务器训练日志目录内") from exc


def _safe_absolute(value: str, title: str) -> str:
    value = value.strip().rstrip("/")
    path = PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts or value in {"", "/"}:
        raise ValueError(f"{title}必须是安全的绝对路径")
    return value


def _ssh_prefix() -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def _scp_prefix() -> list[str]:
    return ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def _ensure_success(result: CommandResult, title: str) -> None:
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}"
        raise RuntimeError(f"{title}：{detail}")
