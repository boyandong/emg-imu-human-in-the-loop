"""Frozen full-action desktop scope, actual runtime tests and install identity."""
import json
from pathlib import Path
from benchmarks.song_real8.verify_temporal_live_gui_v1 import verify

ROOT=Path(__file__).resolve().parents[1]


def test_full_action_gui_regression_acceptance_reconstructs_current_delivery():
    saved=json.loads((ROOT/'feature_bank/TEMPORAL_LIVE_GUI_V1_ACCEPTANCE.json').read_text())
    assert verify()==saved
    assert saved['unique_tests']==197 and saved['desktop_shortcut_installed']
    assert saved['window_off_probability_identity_verified'] and saved['quality_Unknown_preserved']
    assert saved['estimated_auto_chunk_and_direct_interval_parity_verified']
    assert saved['actual_Qt_subprocess_acquisition_lifecycle_verified']
    assert saved['default_model']=='six' and saved['default_temporal_mode']=='off'
    assert not any(saved[k] for k in ('native_four_channel_temporal_primary_guard','default_promoted','physical_validation_proven','completion_proven'))
