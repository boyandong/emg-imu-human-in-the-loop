from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from emgforce.processing.meta_corpus import CORPUS_FILENAME, rebuild_corpus
from emgforce.transfer.dataset_upload import (
    CommandResult, UploadItem, UploadPlan, build_upload_plan, sha256_file,
    upload_plan, validate_server_settings,
)


def _write_ready_export(path: Path, participant: str, session: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = 1_700_000_000.0 + np.arange(4000) / 2000.0
    dtype = np.dtype([("emg", "f4", (8,)), ("time", "f8")])
    data = np.zeros(len(times), dtype=dtype)
    data["time"] = times
    with h5py.File(path, "x") as handle:
        dataset = handle.create_dataset("data", data=data)
        dataset.attrs["task"] = "discrete_gestures"
        dataset.attrs["sample_rate"] = 2000.0
        dataset.attrs["preprocessing_version"] = "meta_8ch_v1"
        meta = handle.create_group("meta")
        meta.attrs["participant_id"] = participant
        meta.attrs["session_id"] = session
    pd.DataFrame({
        "name": ["thumb_click"], "time": [times[100]],
    }).to_hdf(path, key="prompts", mode="a", format="fixed")
    pd.DataFrame({
        "start": [times[0]], "end": [times[-1]], "name": ["default"],
    }).to_hdf(path, key="stages", mode="a", format="fixed")


def test_build_upload_plan_validates_corpus_and_refreshes_manifest(tmp_path) -> None:
    root = tmp_path / "data"
    train = root / "P001" / "S01" / "session_meta_aligned.hdf5"
    val = root / "P001" / "S02" / "session_meta_aligned.hdf5"
    _write_ready_export(train, "P001", "S01")
    _write_ready_export(val, "P001", "S02")
    pd.DataFrame([
        {"dataset": "P001/S01/session_meta_aligned.hdf5", "start": 1, "end": 2,
         "split": "train", "prompt_count": 1},
        {"dataset": "P001/S02/session_meta_aligned.hdf5", "start": 3, "end": 4,
         "split": "val", "prompt_count": 1},
    ]).to_csv(root / CORPUS_FILENAME, index=False)
    (root / "training_manifest.json").write_text('{"stale": true}', encoding="utf-8")

    plan = build_upload_plan(root)

    assert plan.sessions == 2
    assert (plan.train_sessions, plan.val_sessions, plan.test_sessions) == (1, 1, 0)
    assert plan.prompts == 2
    assert len(plan.items) == 4
    assert plan.items[-1].relative_path == CORPUS_FILENAME
    assert plan.warnings
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sessions_by_split"] == {"test": 0, "train": 1, "val": 1}


def test_rebuild_corpus_scans_exports_and_removes_stale_rows(tmp_path) -> None:
    root = tmp_path / "data"
    train = root / "P001" / "2026-08-21_S01" / "session_meta_aligned.hdf5"
    val = root / "P001" / "2026-08-21_S011" / "session_meta_aligned.hdf5"
    _write_ready_export(train, "P001", "S01")
    _write_ready_export(val, "P001", "S011")
    pd.DataFrame([{
        "dataset": "P999/missing/session_meta_aligned.hdf5",
        "start": 1, "end": 2, "split": "train",
    }]).to_csv(root / CORPUS_FILENAME, index=False)

    result = rebuild_corpus(root)

    frame = pd.read_csv(result.corpus_path)
    assert result.sessions == 2
    assert not list(root.glob("*.backup-*.csv"))
    assert set(frame["dataset"]) == {
        "P001/2026-08-21_S01/session_meta_aligned.hdf5",
        "P001/2026-08-21_S011/session_meta_aligned.hdf5",
    }
    assert dict(zip(frame["session"], frame["split"])) == {
        "S01": "train", "S011": "val",
    }
    manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sessions"] == 2


def test_rebuild_corpus_uses_fixed_session_splits(tmp_path) -> None:
    root = tmp_path / "data"
    for number in range(1, 16):
        _write_ready_export(
            root / "P001" / f"2026-08-{number:02d}_S{number:02d}"
            / "session_meta_aligned.hdf5",
            "P001", f"S{number:02d}",
        )

    result = rebuild_corpus(root)

    frame = pd.read_csv(result.corpus_path).set_index("session")
    assert (result.train_sessions, result.val_sessions, result.test_sessions) == (10, 2, 3)
    assert set(frame.loc[[f"S{number:02d}" for number in range(1, 11)], "split"]) == {"train"}
    assert set(frame.loc[["S11", "S12"], "split"]) == {"val"}
    assert set(frame.loc[["S13", "S14", "S15"], "split"]) == {"test"}


def test_upload_plan_rejects_unsafe_dataset_path(tmp_path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame([{
        "dataset": "../session_meta_aligned.hdf5",
        "start": 1, "end": 2, "split": "train",
    }]).to_csv(root / CORPUS_FILENAME, index=False)

    with pytest.raises(ValueError, match="安全相对路径"):
        build_upload_plan(root)


def test_server_settings_reject_shell_syntax_and_broad_roots() -> None:
    assert validate_server_settings(
        "my-gpu-server", "/home/qxy/emg_data/custom") == (
            "my-gpu-server", "/home/qxy/emg_data/custom")
    with pytest.raises(ValueError):
        validate_server_settings("server;shutdown", "/home/qxy/data")
    with pytest.raises(ValueError):
        validate_server_settings("server", "/")
    with pytest.raises(ValueError):
        validate_server_settings("server", "/home/qxy/../root")


def test_uploader_skips_remote_files_with_identical_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("emgforce.transfer.dataset_upload.shutil.which", lambda _name: "found")
    local = tmp_path / "file.bin"
    local.write_bytes(b"already uploaded")
    item = UploadItem(local, "P001/file.bin", local.stat().st_size)
    plan = UploadPlan(tmp_path, (item,), 1, 0, 1, 1, 0, item.size_bytes)
    expected = sha256_file(local)
    commands: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        commands.append(args)
        if args[0] == "ssh" and "sha256sum" in args[-1]:
            return CommandResult(0, expected + "\n", "")
        return CommandResult(0, "", "")

    uploaded, skipped = upload_plan(
        plan, "server", "/home/qxy/data/custom", runner=runner)

    assert (uploaded, skipped) == (0, 1)
    assert not any(command[0] == "scp" for command in commands)


def test_uploader_retries_transient_ssh_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("emgforce.transfer.dataset_upload.shutil.which", lambda _name: "found")
    monkeypatch.setattr("emgforce.transfer.dataset_upload.time.sleep", lambda _delay: None)
    local = tmp_path / "file.bin"
    local.write_bytes(b"already uploaded")
    item = UploadItem(local, "P001/file.bin", local.stat().st_size)
    plan = UploadPlan(tmp_path, (item,), 1, 0, 1, 1, 0, item.size_bytes)
    expected = sha256_file(local)
    attempts = 0

    def runner(args: list[str], _timeout: float) -> CommandResult:
        nonlocal attempts
        if args[0] == "ssh" and "mkdir -p" in args[-1] and attempts == 0:
            attempts += 1
            raise RuntimeError("命令执行超时：ssh")
        if args[0] == "ssh" and "sha256sum" in args[-1]:
            return CommandResult(0, expected + "\n", "")
        return CommandResult(0, "", "")

    assert upload_plan(plan, "server", "/home/qxy/data/custom", runner=runner) == (0, 1)
    assert attempts == 1


def test_uploader_uses_ssh_keepalive_options(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("emgforce.transfer.dataset_upload.shutil.which", lambda _name: "found")
    local = tmp_path / "file.bin"
    local.write_bytes(b"already uploaded")
    item = UploadItem(local, "P001/file.bin", local.stat().st_size)
    plan = UploadPlan(tmp_path, (item,), 1, 0, 1, 1, 0, item.size_bytes)
    expected = sha256_file(local)
    commands: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        commands.append(args)
        if args[0] == "ssh" and "sha256sum" in args[-1]:
            return CommandResult(0, expected + "\n", "")
        return CommandResult(0, "", "")

    upload_plan(plan, "server", "/home/qxy/data/custom", runner=runner)

    assert commands
    assert all("ServerAliveInterval=15" in command for command in commands)
