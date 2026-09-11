from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from emgforce.algorithms import ALGORITHMS, META_CONV_LSTM, PERSONAL_MPF_TDS, get_algorithm

from .dataset_upload import CommandResult, run_command, validate_server_settings


SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
GPU_PATTERN = re.compile(r"^\d+(?:,\d+)*$")
CONDA_ENV_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True, slots=True)
class RemoteTrainingSettings:
    host: str = "my-gpu-server"
    repository: str = "/home/qxy/qxy/generic-neuromotor-interface"
    data_location: str = "/home/qxy/qxy/emg_data/emgforce_dataset"
    conda_init: str = "/home/qxy/miniconda3/etc/profile.d/conda.sh"
    conda_env: str = "neuromotor"
    python: str = "/home/qxy/miniconda3/envs/neuromotor/bin/python3.12"
    tmux_session: str = "emgforce_train"
    gpu_ids: str = "0"
    max_epochs: int = 30
    batch_size: int = 4
    algorithm_id: str = META_CONV_LSTM


@dataclass(frozen=True, slots=True)
class RemoteTrainingStatus:
    exists: bool
    running: bool
    pane_command: str = ""
    output: str = ""


CommandRunner = Callable[[list[str], float], CommandResult]


def validate_training_settings(settings: RemoteTrainingSettings) -> RemoteTrainingSettings:
    get_algorithm(settings.algorithm_id)
    host, data_location = validate_server_settings(
        settings.host, settings.data_location)
    repository = _safe_absolute_path(settings.repository, "训练仓库")
    conda_init = _safe_absolute_path(settings.conda_init, "Conda 初始化脚本")
    conda_env = settings.conda_env.strip()
    python = _safe_absolute_path(settings.python, "Python")
    session = settings.tmux_session.strip()
    gpu_ids = settings.gpu_ids.strip()
    if not SESSION_PATTERN.fullmatch(session):
        raise ValueError("tmux 名称只能包含字母、数字、下划线和连字符（最多64字符）")
    if not GPU_PATTERN.fullmatch(gpu_ids):
        raise ValueError("GPU 编号格式应为 0 或 0,1")
    if not CONDA_ENV_PATTERN.fullmatch(conda_env):
        raise ValueError("Conda 环境名称格式不正确")
    if not 1 <= int(settings.max_epochs) <= 10000:
        raise ValueError("训练轮数必须在 1–10000 之间")
    if not 1 <= int(settings.batch_size) <= 4096:
        raise ValueError("Batch size 必须在 1–4096 之间")
    return RemoteTrainingSettings(
        host, repository, data_location, conda_init, conda_env, python, session, gpu_ids,
        int(settings.max_epochs), int(settings.batch_size), settings.algorithm_id,
    )


def test_training_connection(settings: RemoteTrainingSettings, *,
                             runner: CommandRunner | None = None) -> str:
    settings = validate_training_settings(settings)
    run = runner or run_command
    if settings.algorithm_id == META_CONV_LSTM:
        python_check = (
            "import torch; import pytorch_lightning as pl; "
            "from generic_neuromotor_interface.lightning import DiscreteGesturesModule; "
            "pl.Trainer(accelerator=\"gpu\", devices=1, strategy=\"ddp\", "
            "logger=False, enable_checkpointing=False); ")
    else:
        python_check = "import torch; from mpf_tds.model import MPFTDSNetwork; MPFTDSNetwork(); "
    checks = (
        "command -v tmux >/dev/null || { echo '服务器未安装 tmux'; exit 11; }; "
        f"test -d {shlex.quote(settings.repository)} || "
        "{ echo '训练仓库不存在'; exit 12; }; "
        f"test -f {shlex.quote(settings.conda_init)} || "
        "{ echo 'Conda 初始化脚本不存在'; exit 13; }; "
        f"test -x {shlex.quote(settings.python)} || "
        "{ echo '训练 Python 不可执行'; exit 14; }; "
        f"test -f {shlex.quote(settings.data_location + '/discrete_gestures_corpus.csv')} || "
        "{ echo '远端训练 CSV 不存在，请先上传数据'; exit 15; }; "
        f". {shlex.quote(settings.conda_init)} && "
        f"conda activate {shlex.quote(settings.conda_env)} && "
        f"cd {shlex.quote(settings.repository)} && "
        f"{shlex.quote(settings.python)} -c '{python_check}"
        "print(\"Conda\", __import__(\"os\").environ.get(\"CONDA_DEFAULT_ENV\"), "
        "\"PyTorch\", torch.__version__, \"CUDA\", torch.cuda.is_available())'; "
        "printf 'EMGFORCE_TRAINING_READY\\n'"
    )
    result = run(_ssh_prefix() + [settings.host, checks], 30.0)
    _ensure_success(result, "远端训练环境检查失败")
    if "EMGFORCE_TRAINING_READY" not in result.stdout:
        raise RuntimeError("服务器未返回训练环境确认")
    return result.stdout.replace("EMGFORCE_TRAINING_READY", "").strip()


def start_remote_training(settings: RemoteTrainingSettings, *,
                          runner: CommandRunner | None = None) -> str:
    settings = validate_training_settings(settings)
    run = runner or run_command
    check = run(
        _ssh_prefix() + [settings.host,
                         f"tmux display-message -p -t {shlex.quote(settings.tmux_session)} "
                         "'#{pane_current_command}' 2>/dev/null"],
        15.0,
    )
    reuse_session = check.returncode == 0
    pane_command = check.stdout.strip()
    if reuse_session and pane_command not in {"", "bash", "sh", "zsh", "fish"}:
        raise RuntimeError(
            f"tmux 会话 {settings.tmux_session} 正在运行 {pane_command}，不能重复启动")

    test_training_connection(settings, runner=run)
    command = _training_command(settings)
    ensure_session = (
        "for other in emgforce_train emgforce_train_conv_lstm emgforce_train_mpf_tds; do "
        f"[ \"$other\" = {shlex.quote(settings.tmux_session)} ] && continue; "
        "if tmux display-message -p -t \"$other\" '#{pane_current_command}' 2>/dev/null | "
        "grep -Eq '^(python|python3|python3\\.[0-9]+)$'; then "
        "echo \"另一个训练任务正在运行：$other\" >&2; exit 17; fi; done; "
        f"(tmux has-session -t {shlex.quote(settings.tmux_session)} 2>/dev/null || "
        f"tmux new-session -d -s {shlex.quote(settings.tmux_session)} "
        f"-c {shlex.quote(settings.repository)}) && "
    )
    launch = (
        ensure_session +
        f"tmux set-option -t {shlex.quote(settings.tmux_session)} history-limit 20000 && "
        f"tmux send-keys -l -t {shlex.quote(settings.tmux_session)} "
        f"{shlex.quote(command)} && "
        f"tmux send-keys -t {shlex.quote(settings.tmux_session)} C-m"
    )
    result = run(_ssh_prefix() + [settings.host, launch], 30.0)
    _ensure_success(result, "无法创建远端 tmux 训练任务")
    return f"训练已在 tmux 会话 {settings.tmux_session} 中启动"


def fetch_remote_training(settings: RemoteTrainingSettings, *,
                          runner: CommandRunner | None = None) -> RemoteTrainingStatus:
    settings = validate_training_settings(settings)
    run = runner or run_command
    command = (
        f"if tmux has-session -t {shlex.quote(settings.tmux_session)} 2>/dev/null; then "
        f"printf 'EMGFORCE_PANE='; tmux display-message -p -t "
        f"{shlex.quote(settings.tmux_session)} '#{{pane_current_command}}'; "
        "printf 'EMGFORCE_LOG_BEGIN\\n'; "
        f"tmux capture-pane -p -J -S -1000 -t {shlex.quote(settings.tmux_session)}; "
        "else printf 'EMGFORCE_SESSION_MISSING\\n'; fi"
    )
    result = run(_ssh_prefix() + [settings.host, command], 20.0)
    _ensure_success(result, "无法读取远端训练日志")
    if "EMGFORCE_SESSION_MISSING" in result.stdout:
        return RemoteTrainingStatus(False, False)
    lines = result.stdout.splitlines()
    pane = lines[0].removeprefix("EMGFORCE_PANE=").strip() if lines else ""
    try:
        marker = lines.index("EMGFORCE_LOG_BEGIN")
    except ValueError as exc:
        raise RuntimeError("远端 tmux 日志格式无法识别") from exc
    output = "\n".join(lines[marker + 1:]).rstrip()
    running = pane not in {"", "bash", "sh", "zsh", "fish"}
    return RemoteTrainingStatus(True, running, pane, output)


def stop_remote_training(settings: RemoteTrainingSettings, *,
                         runner: CommandRunner | None = None) -> str:
    settings = validate_training_settings(settings)
    run = runner or run_command
    command = (
        f"if tmux has-session -t {shlex.quote(settings.tmux_session)} 2>/dev/null; then "
        f"tmux kill-session -t {shlex.quote(settings.tmux_session)}; "
        "printf 'EMGFORCE_TRAINING_STOPPED\\n'; "
        "else printf 'EMGFORCE_SESSION_MISSING\\n'; fi"
    )
    result = run(_ssh_prefix() + [settings.host, command], 20.0)
    _ensure_success(result, "无法终止远端训练")
    return (f"已终止 tmux 会话 {settings.tmux_session}"
            if "EMGFORCE_TRAINING_STOPPED" in result.stdout
            else f"tmux 会话 {settings.tmux_session} 不存在")


def _training_command(settings: RemoteTrainingSettings) -> str:
    if settings.algorithm_id == META_CONV_LSTM:
        args = [
            settings.python, "-u", "-m", "generic_neuromotor_interface.train",
            "--config-name=discrete_gestures", f"data_location={settings.data_location}",
            f"trainer.max_epochs={settings.max_epochs}", "trainer.accelerator=gpu",
            "trainer.strategy=ddp", "+trainer.devices=1",
            f"data_module.batch_size={settings.batch_size}", "eval=true",
        ]
    elif settings.algorithm_id == PERSONAL_MPF_TDS:
        args = [settings.python, "-u", "-m", "mpf_tds.train",
                "--data-root", settings.data_location, "--output-root", "logs",
                "--epochs", str(settings.max_epochs), "--batch-size", str(settings.batch_size),
                "--device", "cuda"]
    else:
        raise ValueError(f"不支持的训练算法：{settings.algorithm_id}")
    training = " ".join(shlex.quote(value) for value in args)
    return (
        f". {shlex.quote(settings.conda_init)} && "
        f"conda activate {shlex.quote(settings.conda_env)} && "
        "printf '\\n[EMGForce] conda=%s, training started at %s\\n' "
        "\"$CONDA_DEFAULT_ENV\" \"$(date '+%F %T')\"; "
        f"CUDA_VISIBLE_DEVICES={settings.gpu_ids} {training}; "
        "code=$?; printf '\\n[EMGForce] training finished with exit code %s at %s\\n' "
        "\"$code\" \"$(date '+%F %T')\""
    )


def _safe_absolute_path(value: str, title: str) -> str:
    value = value.strip().rstrip("/")
    path = PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts or value in {"", "/"}:
        raise ValueError(f"{title}必须是安全的绝对路径")
    return value


def _ssh_prefix() -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def _ensure_success(result: CommandResult, title: str) -> None:
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}"
        raise RuntimeError(f"{title}：{detail}")
