"""F4d MANUS session references stay disjoint from held native trials."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.new_bank_v2.roam_posture_run import sha256

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_f4d_manus_multiuser_session_context_split_and_dimensions() -> None:
    protocol = json.loads((ROOT / "F4D_MANUS_SESSION_PROTOCOL.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "F4D_MANUS_SESSION_RESULTS.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == sha256(ROOT / "F4D_MANUS_SESSION_PROTOCOL.json")
    assert result["parent_protocol_sha256"] == sha256(ROOT / protocol["parent_protocol"])
    assert result["archive_sha256"].lower() == protocol["archive_sha256"].lower()
    assert (result["total_source_trials"], result["total_current_calibration_trials"],
            result["total_held_out_trials"]) == (108, 72, 144)
    assert set(result["users"]) == {str(user) for user in protocol["users"]}
    for user in result["users"].values():
        source = set(user["long_trial_ids"])
        assert len(source) == 18
        assert len(user["bands_hz"]) == 4
        for phase in ("validation", "final"):
            session = user["sessions"][phase]
            calibration = set(session["calibration_trial_ids"])
            held = {speed: set(part["native_trials"])
                    for speed, part in session["held_out"].items()}
            assert len(calibration) == 6
            assert {speed: len(trials) for speed, trials in held.items()} == {"slow": 6, "fast": 6}
            assert not (source & calibration or source & held["slow"] or source & held["fast"]
                        or calibration & held["slow"] or calibration & held["fast"]
                        or held["slow"] & held["fast"])
            assert np.asarray(session["session_minus_long_vector"]).shape == (32,)
            assert np.isfinite(session["session_minus_long_vector"]).all()
            for part in session["held_out"].values():
                assert np.asarray(part["held_mean_minus_long_vector"]).shape == (32,)
                assert np.isfinite(part["held_mean_minus_session_l2"])
