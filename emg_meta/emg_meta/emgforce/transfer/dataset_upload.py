from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
import h5py
import numpy as np

from emgforce.collection_protocol import (
    FORMAL_PROTOCOL_NAME, SESSION_MANIFEST_FILENAME,
)

from emgforce.processing.meta_corpus import (
    CORPUS_FILENAME, MANIFEST_FILENAME, validate_training_export,
    write_training_manifest,
)


HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")
REMOTE_PATTERN = re.compile(r"^/[A-Za-z0-9_./-]+$")
REQUIRED_CORPUS_COLUMNS = {"dataset", "start", "end", "split"}
SSH_RETRY_ATTEMPTS = 3
SSH_COMMAND_TIMEOUT_SEC = 60.0
SSH_HASH_TIMEOUT_SEC = 600.0
SCP_TIMEOUT_SEC = 1800.0
FORMAL_DATASET_MANIFEST_FILENAME = "formal_collection_manifest.json"


@dataclass(frozen=True, slots=True)
class UploadItem:
    local_path: Path
    relative_path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class UploadPlan:
    data_root: Path
    items: tuple[UploadItem, ...]
    sessions: int
    prompts: int
    train_sessions: int
    val_sessions: int
    test_sessions: int
    total_bytes: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], float], CommandResult]
ProgressCallback = Callable[[int, int, str], None]


def validate_server_settings(host: str, remote_root: str) -> tuple[str, str]:
    host = host.strip()
    remote_root = remote_root.strip().rstrip("/")
    if not HOST_PATTERN.fullmatch(host):
        raise ValueError("SSH 主机只能包含字母、数字、点、连字符、下划线和 @")
    if not REMOTE_PATTERN.fullmatch(remote_root) or ".." in PurePosixPath(remote_root).parts:
        raise ValueError("远端目录必须是安全的绝对路径，且不能包含 ..")
    if remote_root in {"/", "~", "~/"}:
        raise ValueError("不能把服务器根目录或用户主目录作为数据集目录")
    return host, remote_root


def build_upload_plan(data_root: Path, *, refresh_manifest: bool = True) -> UploadPlan:
    root = Path(data_root).resolve()
    formal_manifests = sorted(root.rglob(SESSION_MANIFEST_FILENAME))
    if formal_manifests:
        return _build_formal_upload_plan(root, formal_manifests)
    corpus_path = root / CORPUS_FILENAME
    if not corpus_path.exists():
        raise ValueError(f"缺少 {CORPUS_FILENAME}，请先完成 Meta 对齐导出")
    frame = pd.read_csv(corpus_path)
    missing = REQUIRED_CORPUS_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Corpus 缺少列：" + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Corpus 中没有可上传的 Session")
    if frame["dataset"].isna().any() or frame["split"].isna().any():
        raise ValueError("Corpus 的 dataset 和 split 不能为空")
    invalid_splits = sorted(set(frame["split"].astype(str)) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError("Corpus 包含无效 split：" + ", ".join(invalid_splits))
    if frame["dataset"].astype(str).duplicated().any():
        raise ValueError("Corpus 中存在重复 dataset 行")

    dataset_items: list[UploadItem] = []
    prompt_total = 0
    for value in frame["dataset"].astype(str):
        relative = _safe_relative_dataset(value)
        local = root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            local.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Dataset 路径越出 data 根目录：{value}") from exc
        if not local.exists():
            raise ValueError(f"Corpus 登记的文件不存在：{relative}")
        readiness = validate_training_export(local)
        if not readiness.ready:
            raise ValueError(
                f"{relative} 未通过训练就绪检查：" + "；".join(readiness.problems))
        prompt_total += readiness.prompt_count
        dataset_items.append(UploadItem(local, relative, local.stat().st_size))

    if refresh_manifest:
        write_training_manifest(root, frame)
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(f"缺少 {MANIFEST_FILENAME}")
    # Upload large datasets first and publish the corpus last.  The remote
    # corpus therefore acts as the commit point and never references a file
    # that has not yet passed its SHA-256 check.
    metadata_items = [
        UploadItem(manifest_path, MANIFEST_FILENAME, manifest_path.stat().st_size),
        UploadItem(corpus_path, CORPUS_FILENAME, corpus_path.stat().st_size),
    ]
    items = tuple(dataset_items + metadata_items)
    split_counts = frame["split"].value_counts()
    train = int(split_counts.get("train", 0))
    val = int(split_counts.get("val", 0))
    test = int(split_counts.get("test", 0))
    if train == 0:
        raise ValueError("至少需要一个 train Session")
    if val == 0:
        raise ValueError("至少需要一个 val Session，Meta 训练需要 val_accuracy")
    warnings = () if test else ("当前没有 test Session，仅适合训练管线验证",)
    return UploadPlan(
        root, items, len(dataset_items), prompt_total, train, val, test,
        sum(item.size_bytes for item in items), warnings,
    )


def _build_formal_upload_plan(root: Path, manifest_paths: list[Path]) -> UploadPlan:
    """Upload only sessions that already passed the collection-side gate."""
    dataset_items: list[UploadItem] = []
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    split_counts = Counter()
    formal_trials = 0
    sessions_by_participant: dict[str, set[str]] = {}

    for readiness_path in manifest_paths:
        payload = json.loads(readiness_path.read_text(encoding="utf-8"))
        relative_manifest = readiness_path.relative_to(root).as_posix()
        if payload.get("status") != "passed":
            warnings.append(f"跳过未通过采集门禁的 Session：{relative_manifest}")
            continue
        hdf5_path = readiness_path.with_name(str(payload.get("hdf5_file", "session.h5")))
        if not hdf5_path.is_file():
            raise ValueError(f"门禁通过但 HDF5 不存在：{hdf5_path}")
        actual_hash = sha256_file(hdf5_path)
        if actual_hash != str(payload.get("hdf5_sha256", "")):
            raise ValueError(f"门禁后 HDF5 已变化，必须重新验收：{hdf5_path}")
        with h5py.File(hdf5_path, "r") as handle:
            meta = handle["meta"].attrs
            protocol = str(meta.get("protocol_name", ""))
            if protocol != FORMAL_PROTOCOL_NAME:
                raise ValueError(f"正式上传只接受 {FORMAL_PROTOCOL_NAME}：{hdf5_path}")
            participant = str(meta.get("participant_id", ""))
            session = str(meta.get("session_id", "")).upper()
            split = str(meta.get("dataset_split", ""))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"正式 Session split 无效：{hdf5_path}")
            trials = handle["trials"][:]
            count = sum(bool(row["valid"]) and _decode_text(row["trial_kind"]) == "formal"
                        for row in trials)
        sessions_by_participant.setdefault(participant, set()).add(session)
        split_counts[split] += 1
        formal_trials += count
        relative_hdf5 = hdf5_path.relative_to(root).as_posix()
        dataset_items.extend((
            UploadItem(hdf5_path, relative_hdf5, hdf5_path.stat().st_size),
            UploadItem(readiness_path, relative_manifest, readiness_path.stat().st_size),
        ))
        records.append({
            "participant_id": participant, "session_id": session, "split": split,
            "dataset": relative_hdf5, "readiness": relative_manifest,
            "hdf5_sha256": actual_hash, "valid_formal_trials": count,
        })

    if not records:
        raise ValueError("没有 status=passed 的正式 Session 可上传")
    for participant, sessions in sessions_by_participant.items():
        if "S03" in sessions and "S02" not in sessions:
            raise ValueError(f"{participant} 缺少 S02，不能上传 S03")
        if "S04" in sessions and not {"S01", "S02", "S03"}.issubset(sessions):
            raise ValueError(f"{participant} 的 S04 缺少前置 S01–S03")
    if split_counts["train"] < 2 or split_counts["val"] < 1:
        warnings.append("尚未形成完整 S01–S03 训练/验证集合")
    if split_counts["test"] == 0:
        warnings.append("S04 尚未采集；冻结算法后再创建 final test")

    root_manifest = root / FORMAL_DATASET_MANIFEST_FILENAME
    manifest_payload = {
        "format": "formal_emg_hdf5_v3_collection_v1",
        "protocol": FORMAL_PROTOCOL_NAME,
        "sessions": sorted(records, key=lambda row: (
            str(row["participant_id"]), str(row["session_id"]))),
        "sessions_by_split": dict(split_counts),
        "valid_formal_trials": formal_trials,
    }
    temporary = root_manifest.with_name(f".{root_manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(root_manifest)
    items = tuple(dataset_items + [UploadItem(
        root_manifest, root_manifest.name, root_manifest.stat().st_size)])
    return UploadPlan(
        root, items, len(records), formal_trials,
        int(split_counts["train"]), int(split_counts["val"]),
        int(split_counts["test"]), sum(item.size_bytes for item in items),
        tuple(warnings),
    )


def _decode_text(value: object) -> str:
    return bytes(value).decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def test_ssh_connection(host: str, remote_root: str,
                        runner: CommandRunner | None = None) -> str:
    host, remote_root = validate_server_settings(host, remote_root)
    _require_programs()
    run = runner or run_command
    command = (
        "printf 'EMGFORCE_SSH_OK\\n'; "
        f"if [ -d {shlex.quote(remote_root)} ]; then printf 'REMOTE_DIR_EXISTS\\n'; "
        "else printf 'REMOTE_DIR_NEW\\n'; fi"
    )
    result = _run_with_retry(
        run, _ssh_prefix() + [host, command], SSH_COMMAND_TIMEOUT_SEC,
        context="SSH 连接测试",
    )
    _ensure_success(result, "SSH 连接测试失败")
    if "EMGFORCE_SSH_OK" not in result.stdout:
        raise RuntimeError("服务器没有返回预期的连接确认")
    return result.stdout.strip()


def upload_plan(plan: UploadPlan, host: str, remote_root: str, *,
                runner: CommandRunner | None = None,
                progress: ProgressCallback | None = None,
                cancelled: Callable[[], bool] | None = None) -> tuple[int, int]:
    host, remote_root = validate_server_settings(host, remote_root)
    _require_programs()
    run = runner or run_command
    is_cancelled = cancelled or (lambda: False)
    report = progress or (lambda _current, _total, _message: None)
    uploaded = skipped = 0
    total = len(plan.items)
    root_result = _run_with_retry(
        run,
        _ssh_prefix() + [host, f"mkdir -p -- {shlex.quote(remote_root)}"],
        SSH_COMMAND_TIMEOUT_SEC, context="创建远端数据集目录",
        cancelled=is_cancelled,
    )
    _ensure_success(root_result, "无法创建远端数据集目录")

    for index, item in enumerate(plan.items, start=1):
        if is_cancelled():
            raise InterruptedError("上传已取消")
        remote = _remote_join(remote_root, item.relative_path)
        remote_parent = str(PurePosixPath(remote).parent)
        local_hash = sha256_file(item.local_path)
        report(index - 1, total, f"检查 {item.relative_path}")
        mkdir = _run_with_retry(
            run,
            _ssh_prefix() + [host, f"mkdir -p -- {shlex.quote(remote_parent)}"],
            SSH_COMMAND_TIMEOUT_SEC, context=f"创建远端目录 {remote_parent}",
            cancelled=is_cancelled,
        )
        _ensure_success(mkdir, f"无法创建远端目录：{remote_parent}")
        current_hash = _remote_sha256(host, remote, run, cancelled=is_cancelled)
        if current_hash == local_hash:
            skipped += 1
            report(index, total, f"已存在且一致，跳过：{item.relative_path}")
            continue

        temporary = f"{remote}.uploading-{os.getpid()}"
        scp = _run_with_retry(
            run, _scp_prefix() + [str(item.local_path), f"{host}:{temporary}"],
            SCP_TIMEOUT_SEC, context=f"上传 {item.relative_path}",
            cancelled=is_cancelled,
        )
        _ensure_success(scp, f"上传失败：{item.relative_path}")
        uploaded_hash = _remote_sha256(
            host, temporary, run, cancelled=is_cancelled)
        if uploaded_hash != local_hash:
            _run_with_retry(
                run,
                _ssh_prefix() + [host, f"rm -f -- {shlex.quote(temporary)}"],
                SSH_COMMAND_TIMEOUT_SEC, context=f"清理临时文件 {temporary}",
                cancelled=is_cancelled,
            )
            raise RuntimeError(f"服务器 SHA-256 校验失败：{item.relative_path}")
        finalize = (
            "stamp=$(date +%Y%m%d-%H%M%S); "
            f"if [ -f {shlex.quote(temporary)} ]; then "
            f"if [ -f {shlex.quote(remote)} ]; then "
            f"mv -- {shlex.quote(remote)} {shlex.quote(remote)}.backup-$stamp; fi; "
            f"mv -- {shlex.quote(temporary)} {shlex.quote(remote)}; "
            f"elif [ ! -f {shlex.quote(remote)} ]; then exit 1; fi"
        )
        # This command is deliberately idempotent.  If the first SSH session
        # completes the move but its response is lost, a retry sees the final
        # file and succeeds; the hash check below still proves its contents.
        result = _run_with_retry(
            run, _ssh_prefix() + [host, finalize], SSH_COMMAND_TIMEOUT_SEC,
            context=f"发布 {item.relative_path}", cancelled=is_cancelled,
        )
        _ensure_success(result, f"无法完成远端原子替换：{item.relative_path}")
        final_hash = _remote_sha256(host, remote, run, cancelled=is_cancelled)
        if final_hash != local_hash:
            raise RuntimeError(f"远端发布后 SHA-256 校验失败：{item.relative_path}")
        uploaded += 1
        report(index, total, f"已上传并校验：{item.relative_path}")
    return uploaded, skipped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(args: list[str], timeout: float) -> CommandResult:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace", creationflags=flags,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _remote_sha256(host: str, remote: str, run: CommandRunner, *,
                   cancelled: Callable[[], bool] | None = None) -> str:
    command = (
        f"if [ -f {shlex.quote(remote)} ]; then "
        f"sha256sum -- {shlex.quote(remote)} | cut -d ' ' -f 1; fi"
    )
    result = _run_with_retry(
        run, _ssh_prefix() + [host, command], SSH_HASH_TIMEOUT_SEC,
        context=f"计算远端校验值 {remote}", cancelled=cancelled,
    )
    _ensure_success(result, f"无法计算远端校验值：{remote}")
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _safe_relative_dataset(value: str) -> str:
    value = value.replace("\\", "/").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Dataset 必须使用 data 根目录下的安全相对路径：{value}")
    if path.name != "session_meta_aligned.hdf5":
        raise ValueError(f"只允许上传 session_meta_aligned.hdf5：{value}")
    return path.as_posix()


def _remote_join(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{PurePosixPath(relative).as_posix()}"


def _ssh_prefix() -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ConnectionAttempts=2", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]


def _scp_prefix() -> list[str]:
    return [
        "scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ConnectionAttempts=2", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]


def _run_with_retry(run: CommandRunner, args: list[str], timeout: float, *,
                    context: str,
                    cancelled: Callable[[], bool] | None = None) -> CommandResult:
    """Retry transient SSH/SCP failures without retrying local validation work."""
    is_cancelled = cancelled or (lambda: False)
    last_error: BaseException | None = None
    last_result: CommandResult | None = None
    for attempt in range(1, SSH_RETRY_ATTEMPTS + 1):
        if is_cancelled():
            raise InterruptedError("上传已取消")
        try:
            result = run(args, timeout)
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            if not _is_timeout_error(exc):
                raise
            last_error = exc
        else:
            last_result = result
            if not _is_transient_connection_failure(result):
                return result
        if attempt < SSH_RETRY_ATTEMPTS:
            time.sleep(float(attempt))

    detail = str(last_error) if last_error is not None else _command_detail(last_result)
    raise RuntimeError(
        f"{context}失败：SSH/SCP 已自动重试 {SSH_RETRY_ATTEMPTS} 次；{detail}"
    ) from last_error


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, subprocess.TimeoutExpired) or "超时" in str(exc) or "timed out" in str(exc).lower()


def _is_transient_connection_failure(result: CommandResult) -> bool:
    if result.returncode != 255:
        return False
    detail = f"{result.stderr}\n{result.stdout}".lower()
    markers = (
        "timed out", "connection reset", "connection closed",
        "connection refused", "network is unreachable", "broken pipe",
        "connection aborted", "connection was aborted",
    )
    return any(marker in detail for marker in markers)


def _command_detail(result: CommandResult | None) -> str:
    if result is None:
        return "未返回结果"
    return result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}"


def _require_programs() -> None:
    missing = [program for program in ("ssh", "scp") if shutil.which(program) is None]
    if missing:
        raise RuntimeError("系统找不到命令：" + ", ".join(missing))


def _ensure_success(result: CommandResult, context: str) -> None:
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}"
        raise RuntimeError(f"{context}：{detail}")
