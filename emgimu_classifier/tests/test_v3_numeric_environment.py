"""A saved V3 run must reject a different numerical dependency environment."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmarks.new_bank_v3.numeric_environment import fingerprint, verify_saved


def test_saved_run_environment_version_guard() -> None:
    snapshot = fingerprint()
    names = ("SPEC_F2C_GRAB_RESULTS.json", "SPEC_F2C_WEARING_RESULTS.json",
             "SPEC_F2C_TRANSFER_RESULTS.json")
    workspace = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="v3-env-", dir=workspace) as folder:
        trial_path = Path(folder)
        for name in names:
            (trial_path / name).write_text(json.dumps({"numeric_environment": snapshot}),
                                           encoding="utf-8")
        assert verify_saved(trial_path) == []
        changed = json.loads((trial_path / names[1]).read_text(encoding="utf-8"))
        changed["numeric_environment"]["packages"]["numpy"] = "incompatible-version"
        (trial_path / names[1]).write_text(json.dumps(changed), encoding="utf-8")
        assert verify_saved(trial_path) == [names[1]]
