from __future__ import annotations

from dataclasses import dataclass


META_CONV_LSTM = "meta_conv_lstm_v1"
PERSONAL_MPF_TDS = "personal_mpf_tds_v1"


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    algorithm_id: str
    display_name: str
    short_name: str
    runtime_backend: str
    remote_repository: str
    tmux_session: str
    conda_env: str
    python: str
    sample_rate_hz: int = 2000
    input_channels: int = 8


ALGORITHMS: dict[str, AlgorithmSpec] = {
    META_CONV_LSTM: AlgorithmSpec(
        META_CONV_LSTM,
        "Meta Conv1D + LSTM（通用模型）",
        "Conv-LSTM",
        "lightning_conv_lstm",
        "/home/qxy/qxy/generic-neuromotor-interface",
        "emgforce_train_conv_lstm",
        "neuromotor",
        "/home/qxy/miniconda3/envs/neuromotor/bin/python3.12",
    ),
    PERSONAL_MPF_TDS: AlgorithmSpec(
        PERSONAL_MPF_TDS,
        "MPF + TDS（单参与者论文复现）",
        "MPF+TDS",
        "mpf_tds",
        "/home/qxy/qxy/emgforce-mpf-tds",
        "emgforce_train_mpf_tds",
        "neuromotor",
        "/home/qxy/miniconda3/envs/neuromotor/bin/python3.12",
    ),
}


def get_algorithm(algorithm_id: str | None) -> AlgorithmSpec:
    resolved = algorithm_id or META_CONV_LSTM
    try:
        return ALGORITHMS[resolved]
    except KeyError as exc:
        raise ValueError(f"不支持的算法：{resolved}") from exc


def algorithm_display_name(algorithm_id: str | None, *, short: bool = False) -> str:
    spec = get_algorithm(algorithm_id)
    return spec.short_name if short else spec.display_name
