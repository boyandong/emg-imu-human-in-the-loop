from __future__ import annotations

import os
from collections import Counter

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from emgforce.experiment.models import ProtocolConfig
from emgforce.experiment.prompt_engine import PromptEngine, PromptState
from emgforce.ui.prompt_window import (
    NAVIGATION_DELTAS, NavigationGridCanvas, ParticipantPromptWindow, TaskRoute,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


DISCRETE_LABELS = [
    "thumb_tap", "thumb_swipe_left", "thumb_swipe_right",
    "thumb_swipe_up", "thumb_swipe_down", "index_hold", "middle_hold",
]


def test_discrete_protocol_is_grouped_into_balanced_routes() -> None:
    protocol = ProtocolConfig("grid", DISCRETE_LABELS, 30, randomize=True)
    sequence = PromptEngine(seed=7).prepare(protocol)
    assert len(sequence) == 210
    assert Counter(sequence) == Counter({label: 30 for label in DISCRETE_LABELS})
    for start in range(0, len(sequence), 7):
        route = sequence[start:start + 7]
        assert sum(action in NAVIGATION_DELTAS for action in route) == 4
        assert route[0] in NAVIGATION_DELTAS
        assert route[1] not in NAVIGATION_DELTAS
        assert route[2] in NAVIGATION_DELTAS
        assert route[3] not in NAVIGATION_DELTAS
        assert route[4] in NAVIGATION_DELTAS
        assert route[5] not in NAVIGATION_DELTAS
        assert route[6] in NAVIGATION_DELTAS


def test_meta_null_collection_is_appended_as_dedicated_stages() -> None:
    protocol = ProtocolConfig(
        "grid_with_null", DISCRETE_LABELS, 2, randomize=False,
        timed_null_actions=["null_finger_snap", "null_finger_flick"],
        timed_null_repetitions=2,
        continuous_null_blocks=[{
            "name": "null_typing", "instruction": "请自然打字", "duration_sec": 12,
        }],
    )
    sequence = PromptEngine(seed=7).prepare(protocol)
    assert len(sequence) == 19
    assert sequence[:14].count("thumb_tap") == 2
    assert Counter(sequence[14:18]) == Counter({
        "null_finger_snap": 2, "null_finger_flick": 2,
    })
    assert sequence[-1] == "null_typing"
    assert protocol.null_kind("thumb_tap") is None
    assert protocol.null_kind("null_finger_snap") == "timed"
    assert protocol.null_kind("null_typing") == "continuous"
    assert protocol.prompt_duration("null_typing") == 12


def test_null_prompts_do_not_emit_target_gesture_cues() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "null", ["thumb_tap"], 1, countdown_sec=0, prompt_duration_sec=0,
        rest_min_sec=0, rest_max_sec=0, randomize=False,
        timed_null_actions=["null_finger_snap"], timed_null_repetitions=1,
        timed_null_prompt_duration_sec=0,
        continuous_null_blocks=[{
            "name": "null_typing", "instruction": "请自然打字", "duration_sec": 1,
        }],
    )
    engine = PromptEngine(seed=1)
    cues: list[str] = []
    engine.gesture_cued.connect(lambda _trial, name: cues.append(name))
    engine.prepare(protocol); engine.start(); engine._timer.stop()
    engine._advance()  # countdown -> target rest
    engine._advance()  # target cue
    engine._advance()  # target end -> timed null rest
    engine._advance()  # timed null prompt, deliberately no target cue
    engine._advance()  # timed null end -> continuous null rest
    engine._advance()  # continuous null prompt, deliberately no target cue
    app.processEvents()
    assert cues == ["thumb_tap"]


def test_prompt_window_reports_route_count_instead_of_action_count() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig("grid", DISCRETE_LABELS, 30, randomize=True)
    sequence = PromptEngine(seed=11).prepare(protocol)
    prompt = ParticipantPromptWindow()
    prompt.configure_task(sequence, protocol.labels)
    prompt.update_prompt(PromptState.PROMPT, sequence[14], 15, len(sequence))
    app.processEvents()
    assert prompt.progress_label.text() == "Trial 3 / 30"
    assert prompt.instruction_text.text().startswith("正在采集：")
    assert prompt.canvas.mode in {"navigation", "activation"}


def test_prompt_window_distinguishes_timed_and_continuous_null() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "null_ui", ["thumb_tap"], 1,
        timed_null_actions=["null_finger_snap"], timed_null_repetitions=1,
        continuous_null_blocks=[{
            "name": "null_typing", "instruction": "请在键盘上自然、连续地打字",
            "duration_sec": 10,
        }],
    )
    sequence = PromptEngine(seed=1).prepare(protocol)
    prompt = ParticipantPromptWindow()
    prompt.configure_task(sequence, protocol.labels, protocol)
    prompt.update_prompt(PromptState.PROMPT, "null_finger_snap", 2, len(sequence))
    assert prompt.canvas.mode == "timed_null"
    assert "非目标动作" in prompt.state_label.text()
    prompt.update_prompt(PromptState.PROMPT, "null_typing", 3, len(sequence))
    app.processEvents()
    assert prompt.canvas.mode == "continuous_null"
    assert "自然、连续地打字" in prompt.instruction_text.text()


def test_continuous_null_countdown_resumes_after_pause() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "null_ui", ["thumb_tap"], 1,
        continuous_null_blocks=[{
            "name": "null_typing", "instruction": "请在键盘上自然、连续地打字",
            "duration_sec": 10,
        }],
    )
    sequence = PromptEngine(seed=1).prepare(protocol)
    prompt = ParticipantPromptWindow()
    prompt.configure_task(sequence, protocol.labels, protocol)
    trial = sequence.index("null_typing") + 1

    prompt.update_prompt(PromptState.PROMPT, "null_typing", trial, len(sequence))
    prompt.update_phase_duration(PromptState.PROMPT, 10)
    prompt.update_prompt(PromptState.PAUSED, "null_typing", trial, len(sequence))
    paused_remaining = prompt.canvas.countdown_remaining_sec
    assert not prompt.canvas.countdown_timer.isActive()

    prompt.update_prompt(PromptState.PROMPT, "null_typing", trial, len(sequence))
    assert prompt.canvas.countdown_timer.isActive()
    assert prompt.canvas.countdown_remaining_sec == paused_remaining
    prompt.canvas._countdown_deadline -= 1
    prompt.canvas._advance_countdown()
    assert prompt.canvas.countdown_remaining_sec < paused_remaining


def test_timed_null_prompt_uses_protocol_instruction() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "null_ui", ["thumb_tap"], 1,
        timed_null_actions=["null_finger_flick"], timed_null_repetitions=1,
        null_instructions={
            "null_finger_flick": "拇指压住食指后释放；手腕保持不动",
        },
    )
    sequence = PromptEngine(seed=1).prepare(protocol)
    prompt = ParticipantPromptWindow()
    prompt.configure_task(sequence, protocol.labels, protocol)
    prompt.update_prompt(PromptState.PROMPT, "null_finger_flick", 2, len(sequence))
    app.processEvents()
    assert prompt.canvas.mode == "timed_null"
    assert prompt.instruction_text.text() == "现在执行：拇指压住食指后释放；手腕保持不动"


def test_rest_phase_scrolls_the_next_action_to_the_indicator() -> None:
    app = QApplication.instance() or QApplication([])
    prompt = ParticipantPromptWindow()
    prompt.configure_task(["thumb_swipe_left"], ["thumb_swipe_left"])
    prompt.update_prompt(PromptState.REST, "thumb_swipe_left", 1, 1)
    app.processEvents()
    assert prompt.canvas.mode == "scroll_preview"
    assert prompt.canvas.preview_action == "thumb_swipe_left"
    assert prompt.instruction_text.text() == "下一动作：拇指向左滑 · 请准备"
    assert prompt.instruction_text.property("fingerTone") == "neutral"
    prompt.update_phase_duration(PromptState.REST, 2.5)
    prompt.canvas.scroll_animation.setCurrentTime(1250)
    assert prompt.canvas.scroll_animation.duration() == 2500
    assert 0.45 <= prompt.canvas.scroll_progress <= 0.55
    prompt.update_prompt(PromptState.PROMPT, "thumb_swipe_left", 1, 1)
    assert prompt.canvas.mode == "navigation"
    assert prompt.canvas.scroll_animation.state().name == "Stopped"


def test_each_navigation_label_places_arrow_tip_in_its_own_direction() -> None:
    center = QPointF(100, 100)
    expected = {
        "thumb_swipe_left": lambda tip: tip.x() < center.x(),
        "thumb_swipe_right": lambda tip: tip.x() > center.x(),
        "thumb_swipe_up": lambda tip: tip.y() < center.y(),
        "thumb_swipe_down": lambda tip: tip.y() > center.y(),
    }
    for label, check in expected.items():
        polygon = NavigationGridCanvas._arrow_polygon(center, NAVIGATION_DELTAS[label])
        assert check(polygon[3]), label


def test_click_prompt_uses_its_own_non_directional_marker_mode() -> None:
    app = QApplication.instance() or QApplication([])
    canvas = NavigationGridCanvas()
    canvas.load_route(prompt_route := TaskRoute.from_actions(["click"]))
    canvas.show_navigation("click")
    app.processEvents()
    assert prompt_route.actions == ["click"]
    assert canvas.mode == "navigation"
    assert canvas.navigation_action == "click"


def test_prompt_text_uses_a_distinct_tone_for_each_finger() -> None:
    app = QApplication.instance() or QApplication([])
    prompt = ParticipantPromptWindow()
    labels = ["thumb_tap", "index_hold", "middle_hold"]
    prompt.configure_task(labels, labels)
    for trial, (label, tone) in enumerate(zip(labels, ("thumb", "index", "middle")), 1):
        prompt.update_prompt(PromptState.PROMPT, label, trial, len(labels))
        app.processEvents()
        assert prompt.instruction_text.property("fingerTone") == tone


def test_optional_posture_instruction_is_visible_without_changing_actions() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "posture", ["thumb_tap"], 1,
        posture_name="掌心向上", posture_instruction="保持前臂稳定并使掌心朝上",
    )
    prompt = ParticipantPromptWindow()
    prompt.configure_task(["thumb_tap"], ["thumb_tap"], protocol)
    app.processEvents()

    assert prompt.posture_text.isHidden() is False
    assert "掌心向上" in prompt.posture_text.text()


def test_hold_trial_emits_meta_press_then_release_without_becoming_two_trials() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "hold", ["index_hold"], 1, countdown_sec=0, rest_min_sec=0,
        rest_max_sec=0, randomize=False, hold_min_sec=0,
        hold_max_sec=0, release_display_sec=0,
    )
    engine = PromptEngine(seed=1)
    cues: list[tuple[int, str]] = []
    ended: list[tuple[int, str]] = []
    engine.gesture_cued.connect(lambda trial, name: cues.append((trial, name)))
    engine.trial_ended.connect(lambda trial, label: ended.append((trial, label)))
    engine.prepare(protocol)
    engine.start(); engine._timer.stop()
    engine._advance()  # countdown -> trial rest
    engine._advance()  # rest -> press/hold
    assert engine.state == PromptState.HOLD
    engine._advance()  # hold -> release
    assert engine.state == PromptState.RELEASE
    engine._advance()  # release -> trial end
    app.processEvents()
    assert cues == [(1, "index_press"), (1, "index_release")]
    assert ended == [(1, "index_hold")]


def test_hold_subphases_have_distinct_press_and_release_ui() -> None:
    app = QApplication.instance() or QApplication([])
    prompt = ParticipantPromptWindow()
    prompt.configure_task(["middle_hold"], ["middle_hold"])
    prompt.update_prompt(PromptState.HOLD, "middle_hold", 1, 1)
    assert prompt.canvas.activation_action == "middle_press"
    assert "按下并保持" in prompt.instruction_text.text()
    prompt.update_prompt(PromptState.RELEASE, "middle_hold", 1, 1)
    assert prompt.canvas.activation_action == "middle_release"
    assert "松开" in prompt.instruction_text.text()


def test_skipping_a_paused_hold_ends_the_native_trial() -> None:
    app = QApplication.instance() or QApplication([])
    protocol = ProtocolConfig(
        "hold", ["index_hold"], 1, countdown_sec=0, rest_min_sec=0,
        rest_max_sec=0, randomize=False, hold_min_sec=1, hold_max_sec=1,
    )
    engine = PromptEngine(seed=1)
    skipped: list[int] = []
    ended: list[int] = []
    engine.trial_skipped.connect(lambda trial, _label: skipped.append(trial))
    engine.trial_ended.connect(lambda trial, _label: ended.append(trial))
    engine.prepare(protocol); engine.start(); engine._timer.stop()
    engine._advance(); engine._timer.stop()
    engine._advance(); engine._timer.stop()
    engine.pause(); engine.skip(); app.processEvents()
    assert skipped == [1]
    assert ended == [1]
