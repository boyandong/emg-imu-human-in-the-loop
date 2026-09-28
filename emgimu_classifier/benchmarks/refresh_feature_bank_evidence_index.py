"""Refresh current evidence hashes without changing requirement acceptance states."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "feature_bank"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_table(path: Path, update_boundary: bool = False) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    for row in rows:
        evidence = ROOT / row["evidence"]
        if not evidence.is_file():
            raise FileNotFoundError(evidence)
        if "evidence_exists" in row:
            row["evidence_exists"] = "True"
        row["evidence_sha256"] = sha(evidence)
        if update_boundary:
            requirement = row["requirement_id"]
            if requirement == "GOAL-ART5":
                row["boundary"] = ("Canonical 0/1/2/5-shot curve now includes 240 pooled/per-subject "
                                   "public DS2 v9 force-safe calibration rows; other recorded rows retain "
                                   "their provenance and unsupported original budgets remain N/A. "
                                   "Own-device cross-person/day validation remains unavailable.")
            elif requirement == "GOAL-CAP":
                row["boundary"] = ("Native channel/rate/label capabilities and unsupported factors recorded. "
                                   "Public DS2 v9 has verified force-coded trials and 0/1/2/5-shot "
                                   "personal curves under distinct unseen-high and product modes; it is "
                                   "three-channel public evidence, not eight-channel device validation.")
            elif requirement == "GOAL-CAL":
                row["boundary"] += (" Separate public DS2 v9 personal 0/1/2/5-shot curves "
                                    "now quantify four-gesture signal time 0/40/80/200 seconds; "
                                    "real product setup/operating time remains unmeasured.") if "Separate public DS2 v9 personal" not in row["boundary"] else ""
            elif requirement == "QUESTION-D":
                row["boundary"] += (" Public DS2 v9 F0/F1/F0+F1 personal curves now exist "
                                    "for both force modes, with substantial user heterogeneity; "
                                    "these are new families, not exact old-family comparisons.") if "Public DS2 v9 F0/F1" not in row["boundary"] else ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(output.getvalue().encode("utf-8"))
    return len(rows)


def main() -> None:
    requirements = refresh_table(BANK / "REQUIREMENT_AUDIT.csv", True)
    sections = refresh_table(BANK / "DOCUMENT_SECTION_AUDIT.csv")
    summary_path = BANK / "DOCUMENT_SECTION_AUDIT.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sections"] == sections
    summary["csv_sha256"] = sha(BANK / "DOCUMENT_SECTION_AUDIT.csv")
    summary_path.write_bytes((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"evidence rows: requirements={requirements}, sections={sections}")


if __name__ == "__main__":
    main()
