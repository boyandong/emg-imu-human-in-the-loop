"""Read back the native-input sensitivity audit without rerunning the archive loader."""
import csv
import json
from pathlib import Path

from benchmarks.new_bank_v2.roam_posture_run import sha256
from emgimu.feature_bank.calibration import EPS


ROOT = Path(__file__).resolve().parents[1] / "benchmarks/new_bank_v2"


def test_document_normalizer_native_input_audit_is_bounded() -> None:
    protocol_path = ROOT / "DOCUMENT_NORMALIZER_EPN_PROTOCOL.json"
    result = json.loads((ROOT / "DOCUMENT_NORMALIZER_EPN_AUDIT.json").read_text(encoding="utf-8"))
    cells_path = ROOT / "DOCUMENT_NORMALIZER_EPN_CELLS.csv"
    with cells_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["protocol_sha256"] == sha256(protocol_path)
    assert result["cells_sha256"] == sha256(cells_path)
    assert len(rows) == result["cells"] == 34
    assert sum(row["phase"].startswith("source") for row in rows) == result["source_cells"] == 16
    assert sum(row["phase"] in ("validation", "descriptive_final") for row in rows) == result["target_cells"] == 18
    assert len({(row["phase"], row["user"], row["shots_per_class"]) for row in rows}) == 34
    assert all(int(row["calibration_trials_count"]) > 0 for row in rows)
    assert all(len(row["calibration_trials_sha256"]) == 64 for row in rows)
    assert result["minimum_q95"] == min(float(row["minimum_q95"]) for row in rows)
    assert result["minimum_q95"] > EPS
    assert result["channels_q95_at_or_below_epsilon"] == sum(
        int(row["channels_q95_at_or_below_epsilon"]) for row in rows) == 0
    assert result["maximum_relative_denominator_difference"] == max(
        float(row["maximum_relative_denominator_difference"]) for row in rows)
    assert result["maximum_relative_denominator_difference"] < 1e-8
    assert result["maximum_calibration_signal_difference"] == max(
        float(row["maximum_calibration_signal_difference"]) for row in rows)
    assert result["maximum_calibration_signal_difference"] < 1e-8
