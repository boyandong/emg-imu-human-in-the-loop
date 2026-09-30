"""Read back the retrospective F2 decision from frozen matched source scores."""
import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "new_bank_v2"


def test_f2_review_preserves_matched_axes_and_validation_only_gate():
    protocol_path = ROOT / "F2_CROSS_AXIS_REVIEW_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "F2_CROSS_AXIS_REVIEW_AUDIT.json").read_text(encoding="utf-8"))
    rows_path = ROOT / "F2_CROSS_AXIS_REVIEW_CELLS.csv"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["protocol_sha256"] == digest(protocol_path)
    assert audit["cells_sha256"] == digest(rows_path)
    for name, expected in audit["source_results_sha256"].items():
        assert expected == protocol["source_results_sha256"][name] == digest(ROOT / name)
    with rows_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == audit["cells"] == 24
    assert {(row["phase"], row["axis"], row["arm"]) for row in rows} == {
        (phase, axis, arm) for phase in ("validation", "final")
        for axis in protocol["axes"] for arm in protocol["arms"]}
    lookup = {(row["phase"], row["axis"], row["arm"]): row for row in rows}
    expected_counts = {"wearing_shift": 120, "manus_session": 108, "grab_unseen_user": 56}
    for row in rows:
        assert int(row["trials"]) == expected_counts[row["axis"]]
        base = lookup[(row["phase"], row["axis"], "F0v2")]
        assert float(row["delta_macro_f1"]) == pytest.approx(
            float(row["macro_f1"])-float(base["macro_f1"]), abs=1e-12)
        assert float(row["delta_log_loss"]) == pytest.approx(
            float(row["log_loss"])-float(base["log_loss"]), abs=1e-12)
    wearing = json.loads((ROOT / "F2_AC_WEARING_RESULTS.json").read_text(encoding="utf-8"))
    manus = json.loads((ROOT / "MANUS_REST_TRANSFER_RESULTS.json").read_text(encoding="utf-8"))
    grab = json.loads((ROOT / "GRAB_USER_RESULTS.json").read_text(encoding="utf-8"))
    assert float(lookup[("validation", "wearing_shift", "F0v2+F2a")]["macro_f1"]) == pytest.approx(
        wearing["scores"]["validation"]["F0v2+F2a_trace"]["pooled"]["macro_f1"])
    assert float(lookup[("validation", "manus_session", "F0v2+F2a")]["log_loss"]) == pytest.approx(
        manus["scores"]["validation"]["F0v2+F2a"]["pooled"]["log_loss"])
    assert float(lookup[("validation", "grab_unseen_user", "F0v2+F2a")]["macro_f1"]) == pytest.approx(
        grab["scores"]["F0v2+F2a"]["validation"]["macro_f1"])
    for arm in protocol["arms"][1:]:
        validation = [lookup[("validation", axis, arm)] for axis in protocol["axes"]]
        violations = [f"{row['axis']}:{name}" for row in validation
                      for name, fails in (("macro_f1", float(row["delta_macro_f1"]) < -1e-12),
                                          ("log_loss", float(row["delta_log_loss"]) > 1e-12)) if fails]
        assert violations == audit["validation_violations"][arm]
    assert audit["eligible_additions"] == []
    assert audit["three_axis_default"] == "F0v2"
