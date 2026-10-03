"""Portable readback of the hash-bound Song F9 V3 aggregate delivery."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "song_real8"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_song_document_f9_native_observation_delivery():
    protocol_path = HERE / "F9_DOCUMENT_V3_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads((HERE / "F9_DOCUMENT_V3_RESULTS.json").read_text(encoding="utf-8"))
    assert result["status"] == "native_observation_only"
    assert result["protocol_sha256"] == _sha(protocol_path)
    assert result["parent_quality_result_sha256"] == _sha(HERE / "QUALITY_OBSERVABILITY.json")
    assert result["source_sessions"] == ["S01", "S02"]
    assert result["source_formal_windows"] == 849
    assert result["feature_dimension"] == 8 * 8 + 5
    assert set(result["evaluation_sessions"]) == {"S03", "S04"}
    for session, cell in result["evaluation_sessions"].items():
        assert cell["formal_windows"] == protocol["expected_formal_windows"][session]
        assert cell["feature_dimension"] == 69
        assert cell["availability"] == {"adc": 1.0, "line": 0.0,
                                        "low_frequency": 1.0, "ring": 0.0}
        assert 0 <= cell["median_zero_fraction"] <= 1
        assert 0 <= cell["median_longest_flatline_ratio"] <= 1
        assert cell["median_absolute_robust_amplitude_z"] >= 0
        assert cell["median_covariance_distance"] >= 0
        assert cell["synthetic_constant_ch1_median_flatline"] > .9
        assert cell["synthetic_constant_ch1_detected_fraction_at_0_9"] == 1
        assert cell["natural_ch1_above_0_9_fraction"] == 0
