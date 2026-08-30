from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from emgforce.transfer.dataset_upload import CommandResult
from emgforce.transfer.remote_models import (
    RemoteModelInfo, download_remote_model, list_remote_models,
)


def test_list_remote_models_parses_and_sorts_server_results() -> None:
    older = {
        "model_id": "emg_20260819_162106_e27",
        "created_at": "2026-08-19 16:21:06",
        "checkpoint_path": "/repo/logs/2026-08-19/16-21-06/lightning_logs/version_0/checkpoints/epoch=27-step=476.ckpt",
        "config_path": "/repo/logs/2026-08-19/16-21-06/hydra_configs/config.yaml",
        "run_dir": "/repo/logs/2026-08-19/16-21-06",
        "epoch": 27, "step": 476, "size_bytes": 100,
        "val_accuracy": 0.21, "test_cler": None,
    }
    newer = {
        "model_id": "emg_20260821_113915_e45",
        "created_at": "2026-08-21 11:39:15",
        "checkpoint_path": "/repo/logs/2026-08-21/11-39-15/lightning_logs/version_0/checkpoints/epoch=45-step=1426.ckpt",
        "config_path": "/repo/logs/2026-08-21/11-39-15/hydra_configs/config.yaml",
        "run_dir": "/repo/logs/2026-08-21/11-39-15",
        "epoch": 45, "step": 1426, "size_bytes": 200,
        "val_accuracy": 0.55, "test_cler": 0.09,
    }
    stdout = "\n".join((json.dumps(older), json.dumps(newer)))
    models = list_remote_models(
        "server", "/repo", "/env/python",
        runner=lambda _args, _timeout: CommandResult(0, stdout, ""),
    )
    assert [model.model_id for model in models] == [
        "emg_20260821_113915_e45", "emg_20260819_162106_e27"]
    assert models[0].val_accuracy == 0.55
    assert models[0].test_cler == 0.09


def test_remote_model_rejects_unsafe_server_paths() -> None:
    model = RemoteModelInfo(
        "safe", "2026-08-21 00:00:00", "/tmp/model.ckpt",
        "/tmp/config.yaml", "/tmp", 1, 1, 1,
    )
    with pytest.raises(ValueError, match="日志目录"):
        download_remote_model(model, "server", __import__("pathlib").Path("models"),
                              repository="/repo", runner=lambda *_: None)


def test_download_remote_model_keeps_native_checkpoint(tmp_path: Path) -> None:
    checkpoint = b"native-lightning-checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()
    model = RemoteModelInfo(
        "emg_run_e4", "2026-08-21 12:00:00",
        "/repo/logs/2026-08-21/12-00-00/lightning_logs/version_0/checkpoints/epoch=4-step=9.ckpt",
        "/repo/logs/2026-08-21/12-00-00/hydra_configs/config.yaml",
        "/repo/logs/2026-08-21/12-00-00", 4, 9, len(checkpoint), 0.6, 0.1,
    )

    def runner(args: list[str], _timeout: float) -> CommandResult:
        if args[0] == "ssh":
            assert "sha256sum" in args[-1]
            assert ".ts" not in args[-1]
            return CommandResult(0, f"{digest}  checkpoint.ckpt\n", "")
        destination = Path(args[-1])
        if str(args[-2]).endswith("config.yaml"):
            destination.write_text("lightning_module: {}\n", encoding="utf-8")
        else:
            destination.write_bytes(checkpoint)
        return CommandResult(0, "", "")

    bundle = download_remote_model(
        model, "server", tmp_path, repository="/repo", runner=runner)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact"]["format"] == "pytorch_lightning"
    assert manifest["artifact"]["filename"].endswith(".ckpt")
    assert manifest["source_checkpoint"]["sha256"] == digest
    assert not list(bundle.glob("*.ts"))
