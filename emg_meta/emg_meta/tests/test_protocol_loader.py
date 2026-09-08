from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

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


def test_jilv_music_protocol_balances_stationary_and_moving_combinations() -> None:
    protocol_dir = Path(__file__).parents[1] / "protocols"
    config = ProtocolLoader(protocol_dir).load("jilv_music_21")
    assert config.name == "jilv_music_21_v1"
    assert config.labels == [
        f"{arm}_{hand}"
        for arm in ("still", "up", "down", "left", "right", "forward", "backward")
        for hand in ("index_pinch", "fist", "open_hand")
    ]
    sequence = PromptEngine(seed=9).prepare(config)
    assert len(sequence) == 108
    assert set(sequence) == set(config.labels)
    counts = Counter(sequence)
    assert all(counts[label] == 18 for label in config.labels[:3])
    assert all(counts[label] == 3 for label in config.labels[3:])
    assert sum(counts[label] for label in config.labels[:3]) == 54
    assert sum(counts[label] for label in config.labels[3:]) == 54
    assert config.randomize is True
    assert sequence != [
        label for label in config.labels for _ in range(config.trial_count(label))
    ]


def test_existing_session_id_advances_without_overwrite(tmp_path: Path) -> None:
    from emgforce.ui.experiment_page import ExperimentPage

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    participant_dir = tmp_path / "Dong"
    (participant_dir / "2026-09-01_S01").mkdir(parents=True)
    (participant_dir / "2026-09-02_S03").mkdir()
    page = ExperimentPage(Path(__file__).parents[1] / "protocols", tmp_path)
    assert page._available_session_id("Dong") == "S02"
    page.participant_id.setText("Dong")
    page.refresh_session_id()
    assert page.session_id.text() == "S02"
    assert "2 / 11" in page.session_plan_status.text()
    app.processEvents()


def test_participant_is_limited_to_eleven_sessions(tmp_path: Path) -> None:
    from emgforce.ui.experiment_page import ExperimentPage

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    for number in range(1, 12):
        (tmp_path / "P001" / f"2026-09-01_S{number:02d}").mkdir(parents=True)
    page = ExperimentPage(Path(__file__).parents[1] / "protocols", tmp_path)
    with pytest.raises(ValueError, match="11 / 11"):
        page._available_session_id("P001")
    page.participant_id.setText("P001")
    page.refresh_session_id()
    assert page.session_id.text() == "已完成"
    assert not page.start.isEnabled()
    app.processEvents()
