"""Read-only, hash-bound public default guard for newer formula candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "PUBLIC_DEFAULT_EXTENSION_PROTOCOL.json"
OUTPUT = HERE / "PUBLIC_DEFAULT_EXTENSION_AUDIT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bound(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"frozen evidence hash changed: {path}")


def paired(name: str, base: dict, candidate: dict, axis: str) -> dict:
    f1_change = candidate["macro_f1"] - base["macro_f1"]
    loss_change = candidate["log_loss"] - base["log_loss"]
    passes = f1_change >= -1e-12 and loss_change <= 1e-12 and (
        f1_change > 1e-12 or loss_change < -1e-12
    )
    return {"candidate": name, "axis": axis, "validation_delta_macro_f1": f1_change,
            "validation_delta_log_loss": loss_change, "validation_guard_pass": passes}


def build() -> dict:
    protocol = read(PROTOCOL)
    sources = {}
    for relative, expected in protocol["source_sha256"].items():
        path = ROOT / relative
        bound(path, expected)
        sources[relative] = read(path)
    spatial = sources["benchmarks/new_bank_v3/SPEC_SPATIAL_GRAB_RESULTS.json"]
    f2c = sources["benchmarks/new_bank_v3/SPEC_F2C_CROSS_AXIS_AUDIT.json"]
    f5c = sources["benchmarks/new_bank_v2/F5C_UNIBO_BOUT_RESULTS.json"]
    f7 = sources["benchmarks/new_bank_v3/F7_DOCUMENT_ANCHOR_RESULTS.json"]
    for record, folder, stem in ((spatial, HERE, "SPEC_SPATIAL_GRAB"),
                                 (f5c, ROOT / "benchmarks" / "new_bank_v2", "F5C_UNIBO_BOUT"),
                                 (f7, HERE, "F7_DOCUMENT_ANCHOR")):
        bound(folder / f"{stem}_PROTOCOL.json", record["protocol_sha256"])
        bound(folder / f"{stem}_PREDICTIONS.csv", record["prediction_sha256"])
    bound(HERE / "SPEC_F2C_CROSS_AXIS_PROTOCOL.json", f2c["protocol_sha256"])
    bound(HERE / "SPEC_F2C_CROSS_AXIS_CELLS.csv", f2c["cells_sha256"])
    if f2c["cells"] != 6 or f2c["candidate_eligible_for_public_default"]:
        raise ValueError("centered F2c three-axis decision changed")
    if not f2c["validation_violations"]:
        raise ValueError("centered F2c negative control changed")

    checks = []
    scores = spatial["scores"]
    for arm in ("F0+F2a_spec", "F0+F3c_spec"):
        checks.append(paired(arm, scores["F0"]["validation"], scores[arm]["validation"],
                             "grab_cross_day"))
    checks.append({"candidate": "F0+F2c_spec", "axis": "wearing|manus_session|grab_unseen_user",
                   "validation_guard_pass": False, "validation_violations": f2c["validation_violations"]})
    checks.append(paired("G5+F5c", f5c["scores"]["validation"]["G5"]["pooled"],
                         f5c["scores"]["validation"]["G5+F5c"]["pooled"], "unibo_full_bout"))
    for budget in (1, 2, 5):
        cell = f7["metrics"]["validation"][str(budget)]
        checks.append(paired(f"F0v2+F7_document_{budget}shot", cell["F0v2"],
                             cell["F0v2+F7_document"], "grab_cross_day_calibrated"))
    if any(check["validation_guard_pass"] for check in checks):
        raise ValueError("new candidate passed local guard; review before default decision")
    result = {"status": "reviewed_public_negative", "protocol_sha256": sha(PROTOCOL),
              "source_sha256": protocol["source_sha256"], "candidate_checks": checks,
              "local_guard_passes": 0, "eligible_new_universal_public_defaults": [],
              "existing_public_default_retained": "F0v2",
              "boundary": protocol["boundary"]}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({"candidate_checks": len(result["candidate_checks"]),
                      "local_guard_passes": result["local_guard_passes"],
                      "default": result["existing_public_default_retained"]}))
