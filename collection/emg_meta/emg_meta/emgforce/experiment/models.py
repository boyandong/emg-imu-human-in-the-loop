from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .events import EventType


@dataclass(slots=True)
class ParticipantInfo:
    participant_id: str
    dominant_hand: str = ""

    def validate(self) -> None:
        if not self.participant_id.strip():
            raise ValueError("参与者编号必填")


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    experiment_name: str
    protocol_name: str
    serial_port: str = ""
    stage_id: int = 1
    stage_name: str = "default"
    dataset_split: str = "train"
    quality_report_json: str = ""
    donning_notes: str = ""
    tested_arm: str = ""
    channel1_orientation: str = ""
    anatomical_marker: str = ""
    strap_setting: str = ""
    stabilization_sec: int = 60
    physical_condition: str = ""
    recorded_arm: str = ""
    donning_code: str = "D01"
    anatomical_distance_mm: float = 0.0
    strap_scale: str = ""
    strap_tightness: int = -1
    skin_condition: str = ""
    fatigue_before: int = -1
    fatigue_after: int = -1
    reference_photo_name: str = ""
    reference_photo_sha256: str = ""
    operator_id: str = ""
    quality_override_reason: str = ""
    model_frozen_confirmed: bool = False

    def validate(self) -> None:
        if not self.session_id.strip():
            raise ValueError("实验场次编号必填")
        if not self.experiment_name.strip():
            raise ValueError("实验名称必填")
        if self.dataset_split not in {"auto", "train", "val", "test"}:
            raise ValueError("数据集划分只能是自动、训练、验证或测试")
        if self.strap_tightness not in range(-1, 6):
            raise ValueError("绑带松紧必须为 0–5，未知时为 -1")
        for name, value in (("采集前疲劳", self.fatigue_before),
                            ("采集后疲劳", self.fatigue_after)):
            if value not in range(-1, 11):
                raise ValueError(f"{name}必须为 0–10，未记录时为 -1")


@dataclass(slots=True)
class ProtocolConfig:
    name: str
    labels: list[str]
    trials_per_class: int
    countdown_sec: float = 3.0
    prompt_duration_sec: float = 1.0
    rest_min_sec: float = 1.0
    rest_max_sec: float = 2.0
    randomize: bool = True
    trials_per_label: dict[str, int] = field(default_factory=dict)
    hold_min_sec: float = 2.5
    hold_max_sec: float = 2.5
    release_display_sec: float = 1.2
    timed_null_actions: list[str] = field(default_factory=list)
    timed_null_repetitions: int = 0
    timed_null_prompt_duration_sec: float = 1.0
    null_instructions: dict[str, str] = field(default_factory=dict)
    continuous_null_blocks: list[dict[str, Any]] = field(default_factory=list)
    posture_name: str = ""
    posture_instruction: str = ""
    onset_offsets_ms: list[int] = field(default_factory=list)
    formal_collection: bool = False
    protocol_version: str = ""
    block_size: int = 0
    block_break_sec: float = 0.0
    calibration_blocks: list[dict[str, Any]] = field(default_factory=list)
    transition_guard_ms: int = 300
    quality_threshold_version: str = ""
    source_filename: str = ""
    source_sha256: str = ""

    def validate(self) -> None:
        if not self.name.strip() or not self.labels:
            raise ValueError("实验协议必须包含名称和非空动作标签")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("实验协议中的动作标签不能重复")
        if self.trials_per_class < 1:
            raise ValueError("每类试次数必须大于 0")
        unknown_trial_labels = set(self.trials_per_label) - set(self.labels)
        if unknown_trial_labels:
            raise ValueError(
                "分类试次数包含未声明的动作：" + ", ".join(sorted(unknown_trial_labels)))
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 1
               for count in self.trials_per_label.values()):
            raise ValueError("每个动作的独立试次数必须为正整数")
        if min(self.countdown_sec, self.prompt_duration_sec,
               self.rest_min_sec, self.rest_max_sec, self.hold_min_sec,
               self.hold_max_sec, self.release_display_sec) < 0:
            raise ValueError("实验协议的时间参数不能为负")
        if self.rest_min_sec > self.rest_max_sec:
            raise ValueError("最短休息时间不能大于最长休息时间")
        if self.hold_min_sec > self.hold_max_sec:
            raise ValueError("最短保持时间不能大于最长保持时间")

        if self.timed_null_repetitions < 0:
            raise ValueError("定时 null 动作重复次数不能为负数")
        if self.timed_null_prompt_duration_sec < 0:
            raise ValueError("定时 null 动作提示时长不能为负数")
        null_names = list(self.timed_null_actions)
        for block in self.continuous_null_blocks:
            if not isinstance(block, dict):
                raise ValueError("连续 null 阶段必须使用对象配置")
            name = str(block.get("name", "")).strip()
            if not name or not name.startswith("null_"):
                raise ValueError("连续 null 阶段名称必须以 null_ 开头")
            if float(block.get("duration_sec", 0)) <= 0:
                raise ValueError("连续 null 阶段时长必须大于 0")
            null_names.append(name)
        if any(not name.startswith("null_") for name in self.timed_null_actions):
            raise ValueError("定时 null 动作名称必须以 null_ 开头")
        if len(null_names) != len(set(null_names)):
            raise ValueError("null 动作或阶段名称不能重复")
        if set(null_names) & set(self.labels):
            raise ValueError("null 名称不能与目标动作标签重复")
        unknown_instructions = set(self.null_instructions) - set(null_names)
        if unknown_instructions:
            raise ValueError(
                "null 操作说明包含未声明的行为："
                + ", ".join(sorted(unknown_instructions)))
        if any(not str(text).strip() for text in self.null_instructions.values()):
            raise ValueError("null 操作说明不能为空")
        if self.posture_instruction.strip() and not self.posture_name.strip():
            raise ValueError("配置姿态说明时必须同时配置 posture_name")
        if any(offset not in {-200, 0, 200} for offset in self.onset_offsets_ms):
            raise ValueError("动作相对起始偏移只允许 -200、0 或 200 ms")
        if self.block_size < 0 or self.block_break_sec < 0:
            raise ValueError("block 大小和强制休息时长不能为负")
        if self.formal_collection and (self.block_size < 1 or self.block_break_sec <= 0):
            raise ValueError("正式协议必须配置 block_size 和 block_break_sec")
        if self.transition_guard_ms < 0:
            raise ValueError("过渡保护时间不能为负")
        calibration_names: list[str] = []
        for block in self.calibration_blocks:
            if not isinstance(block, dict):
                raise ValueError("校准块必须使用对象配置")
            name = str(block.get("name", "")).strip()
            if not name.startswith("calibration_"):
                raise ValueError("校准块名称必须以 calibration_ 开头")
            if float(block.get("duration_sec", 0)) <= 0:
                raise ValueError("校准块采集时长必须大于 0")
            if not str(block.get("instruction", "")).strip():
                raise ValueError("校准块必须提供动作说明")
            calibration_names.append(name)
        if len(calibration_names) != len(set(calibration_names)):
            raise ValueError("校准块名称不能重复")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def null_kind(self, label: str) -> str | None:
        if label in self.timed_null_actions:
            return "timed"
        if any(str(block.get("name")) == label for block in self.continuous_null_blocks):
            return "continuous"
        return None

    def prompt_duration(self, label: str) -> float:
        if label in self.timed_null_actions:
            return self.timed_null_prompt_duration_sec
        for block in self.continuous_null_blocks:
            if str(block.get("name")) == label:
                return float(block["duration_sec"])
        for block in self.calibration_blocks:
            if str(block.get("name")) == label:
                return float(block["duration_sec"])
        return self.prompt_duration_sec

    def calibration_block(self, label: str) -> dict[str, Any] | None:
        for block in self.calibration_blocks:
            if str(block.get("name")) == label:
                return block
        return None

    def trial_count(self, label: str) -> int:
        return int(self.trials_per_label.get(label, self.trials_per_class))

    def null_instruction(self, label: str) -> str:
        if label in self.null_instructions:
            return self.null_instructions[label]
        for block in self.continuous_null_blocks:
            if str(block.get("name")) == label:
                return str(block.get("instruction", label))
        return label


@dataclass(slots=True)
class ExperimentEvent:
    event_id: int
    sample_index: int
    time_sec: float
    pc_monotonic_ns: int
    event_type: EventType
    label: str = ""
    trial_id: int = -1
    stage_id: int = -1
    donning_id: int = -1
    note: str = ""


@dataclass(slots=True)
class CueEvent:
    """UI action cue aligned to the authoritative EMG sample index."""

    name: str
    sample_index: int
    time_sec: float
    trial_id: int
    stage_id: int
    scheduled_monotonic_ns: int = -1
    emitted_monotonic_ns: int = -1
    event_uid: str = ""


@dataclass(slots=True)
class TrialInfo:
    trial_id: int
    label: str
    stage_id: int
    donning_id: int
    trial_start_sample: int = -1
    rest_start_sample: int = -1
    prompt_start_sample: int = -1
    prompt_end_sample: int = -1
    trial_end_sample: int = -1
    valid: bool = True
    reject_reason: str = ""
    note: str = ""
    relative_onset_offset_ms: int = 0
    trial_kind: str = "formal"
    block_index: int = 0
    attempt: int = 1
    rerecord_of_trial_id: int = -1
    event_uid: str = ""
    stable_start_sample: int = -1
    stable_end_sample: int = -1
    release_prompt_sample: int = -1
    completion_status: str = "pending"
    discard_reason: str = ""
