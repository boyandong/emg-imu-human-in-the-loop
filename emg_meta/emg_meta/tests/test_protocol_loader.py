from __future__ import annotations

import json
from pathlib import Path

import pytest

from emgforce.experiment.protocol_loader import ProtocolLoader
from emgforce.experiment.prompt_engine import PromptEngine


def test_protocol_loader_needs_no_python_change_for_new_labels(tmp_path) -> None:
    payload = {"name": "pinch_v1", "labels": ["index_pinch", "middle_pinch"],
               "trials_per_class": 3, "countdown_sec": 0, "prompt_duration_sec": .1,
               "rest_min_sec": .2, "rest_max_sec": .3, "randomize": False}
    (tmp_path / "pinch.json").write_text(json.dumps(payload), encoding="utf-8")
    loader = ProtocolLoader(tmp_path)
    assert loader.discover()["pinch_v1"].name == "pinch.json"
    config = loader.load("pinch")
    assert config.labels == ["index_pinch", "middle_pinch"]


def test_protocol_loader_rejects_unknown_fields(tmp_path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({"name": "x", "labels": ["x"],
        "trials_per_class": 1, "python_callback": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="未知"):
        ProtocolLoader(tmp_path).load("bad")


def test_meta_protocol_declares_both_published_null_collection_modes() -> None:
    protocol_dir = Path(__file__).parents[1] / "protocols"
    config = ProtocolLoader(protocol_dir).load("meta_discrete_7")
    assert config.timed_null_actions == ["null_finger_snap", "null_finger_flick"]
    assert config.name == "meta_discrete_7_short_v2"
    assert config.trials_per_class == 12
    assert config.timed_null_repetitions == 3
    assert "中指与拇指" in config.null_instruction("null_finger_snap")
    assert "食指向前弹出" in config.null_instruction("null_finger_flick")
    assert config.null_kind("null_typing") == "continuous"
    assert config.prompt_duration("null_typing") == 60.0
    sequence = PromptEngine(seed=1).prepare(config)
    assert len(sequence) == 91
    assert sum(config.null_kind(label) is None for label in sequence) == 84
    assert sum(config.null_kind(label) == "timed" for label in sequence) == 6
    assert sum(config.null_kind(label) == "continuous" for label in sequence) == 1


def test_posture_protocol_is_additive_and_keeps_null_schedule_unchanged() -> None:
    protocol_dir = Path(__file__).parents[1] / "protocols"
    base = ProtocolLoader(protocol_dir).load("meta_discrete_7")
    posture = ProtocolLoader(protocol_dir).load("meta_discrete_7_posture_palm_up_v1")

    assert posture.posture_name == "掌心向上"
    assert "掌心朝上" in posture.posture_instruction
    assert posture.timed_null_actions == base.timed_null_actions
    assert posture.timed_null_repetitions == base.timed_null_repetitions
    assert posture.continuous_null_blocks == base.continuous_null_blocks
    assert PromptEngine(seed=7).prepare(posture) == PromptEngine(seed=7).prepare(base)
