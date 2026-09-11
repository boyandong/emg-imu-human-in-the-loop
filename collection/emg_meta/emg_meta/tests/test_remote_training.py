from __future__ import annotations

import pytest

from emgforce.transfer.dataset_upload import CommandResult
from emgforce.transfer.remote_training import (
    RemoteTrainingSettings, fetch_remote_training, start_remote_training,
    validate_training_settings,
)


def test_remote_training_settings_reject_shell_injection() -> None:
    with pytest.raises(ValueError, match="tmux"):
        validate_training_settings(RemoteTrainingSettings(tmux_session="train; reboot"))
    with pytest.raises(ValueError, match="GPU"):
        validate_training_settings(RemoteTrainingSettings(gpu_ids="0;1"))
    with pytest.raises(ValueError, match="绝对路径"):
        validate_training_settings(RemoteTrainingSettings(repository="/home/qxy/../root"))


def test_start_remote_training_creates_tmux_and_sends_expected_command() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        commands.append(args)
        remote = args[-1]
        if remote.startswith("tmux display-message"):
            return CommandResult(1, "", "")
        if "EMGFORCE_TRAINING_READY" in remote:
            return CommandResult(
                0, "PyTorch 2.4 CUDA True\nEMGFORCE_TRAINING_READY\n", "")
        return CommandResult(0, "", "")

    message = start_remote_training(RemoteTrainingSettings(), runner=runner)

    assert "emgforce_train" in message
    launch = commands[-1][-1]
    assert "tmux has-session -t emgforce_train" in launch
    assert "tmux new-session -d" in launch
    assert "tmux send-keys" in launch
    assert "generic_neuromotor_interface.train" in launch
    assert "conda activate neuromotor" in launch
    assert "trainer.max_epochs=30" in launch
    assert "trainer.strategy=ddp" in launch
    assert "trainer.strategy=auto" not in launch
    assert "data_module.batch_size=4" in launch
    assert "CUDA_VISIBLE_DEVICES=0" in launch


def test_fetch_remote_training_parses_live_tmux_output() -> None:
    def runner(_args: list[str], _timeout: float) -> CommandResult:
        return CommandResult(
            0,
            "EMGFORCE_PANE=python3.12\nEMGFORCE_LOG_BEGIN\nEpoch 4: 72%\nval_accuracy=0.81\n",
            "",
        )

    status = fetch_remote_training(RemoteTrainingSettings(), runner=runner)

    assert status.exists and status.running
    assert status.pane_command == "python3.12"
    assert "Epoch 4: 72%" in status.output
    assert "val_accuracy=0.81" in status.output


def test_fetch_remote_training_reports_missing_tmux_session() -> None:
    status = fetch_remote_training(
        RemoteTrainingSettings(),
        runner=lambda _args, _timeout: CommandResult(
            0, "EMGFORCE_SESSION_MISSING\n", ""),
    )
    assert not status.exists
    assert not status.running


def test_mpf_tds_training_uses_independent_module_and_repository() -> None:
    commands: list[list[str]] = []

    def runner(args: list[str], _timeout: float) -> CommandResult:
        commands.append(args)
        remote = args[-1]
        if remote.startswith("tmux display-message"):
            return CommandResult(1, "", "")
        if "EMGFORCE_TRAINING_READY" in remote:
            assert "MPFTDSNetwork" in remote
            return CommandResult(0, "EMGFORCE_TRAINING_READY\n", "")
        return CommandResult(0, "", "")

    settings = RemoteTrainingSettings(
        repository="/home/qxy/qxy/emgforce-mpf-tds",
        tmux_session="emgforce_train_mpf_tds",
        algorithm_id="personal_mpf_tds_v1",
    )
    start_remote_training(settings, runner=runner)
    launch = commands[-1][-1]
    assert "mpf_tds.train" in launch
    assert "generic_neuromotor_interface.train" not in launch
    assert "--data-root" in launch
