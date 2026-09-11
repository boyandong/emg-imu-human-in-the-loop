from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from emgforce.processing.global_alignment_template import (
    build_global_template, load_global_template,
)
from emgforce.processing.template_alignment import FEATURE_NAME, METHOD_NAME


def _write_export(path: Path, participant: str, value: float) -> None:
    with h5py.File(path, "x") as handle:
        meta = handle.create_group("meta")
        meta.attrs["participant_id"] = participant
        events = handle.create_dataset("alignment_events", data=np.zeros(1))
        events.attrs["algorithm"] = METHOD_NAME
        events.attrs["feature_extractor"] = FEATURE_NAME
        templates = handle.create_group("alignment_templates")
        templates.create_dataset(
            "thumb_up", data=np.full((21, 192), value, dtype=np.float32))


def test_global_template_is_participant_balanced_and_hash_checked(tmp_path) -> None:
    # P1 has two sessions, but must have the same weight as P2.
    paths = [tmp_path / name for name in ("p1a.h5", "p1b.h5", "p2.h5")]
    _write_export(paths[0], "P1", 1.0)
    _write_export(paths[1], "P1", 3.0)
    _write_export(paths[2], "P2", 6.0)
    target = tmp_path / "global.h5"

    build_global_template(paths, target)
    bundle = load_global_template(target)

    assert bundle.participant_count == 2
    assert bundle.session_count == 3
    np.testing.assert_allclose(bundle.templates["thumb_up"], 4.0)
