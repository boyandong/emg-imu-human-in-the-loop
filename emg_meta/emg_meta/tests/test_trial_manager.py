from __future__ import annotations

from emgforce.experiment.trial_manager import TrialManager


def test_bad_trial_is_retained_with_reason_and_indices() -> None:
    manager = TrialManager()
    manager.start(1, "left_swipe", 2, 3, 10)
    manager.set_index("rest_start_sample", 10)
    manager.set_index("prompt_start_sample", 20)
    manager.mark_bad("wrong_gesture", "performed right swipe")
    manager.set_index("prompt_end_sample", 30)
    trial = manager.end(40)
    assert not trial.valid
    assert trial.reject_reason == "wrong_gesture"
    assert trial.note == "performed right swipe"
    assert (trial.trial_start_sample, trial.prompt_start_sample, trial.trial_end_sample) == (10, 20, 40)
    assert manager.trials == [trial]

