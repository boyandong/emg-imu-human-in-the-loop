"""Bind every F0–F9 appendix subsection to an exact reviewed source symbol."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "feature_bank"
CONTEXT_ONLY = {
    "F3. Ring Geometry / Wearing",
    "F4. Spectral State",
    "F6. Body Context / IMU",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    with (BANK / "DOCUMENT_SECTION_AUDIT.csv").open(encoding="utf-8", newline="") as stream:
        sections = list(csv.DictReader(stream))
    with (BANK / "FORMULA_IMPLEMENTATION_AUDIT.csv").open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    appendix = [row for row in sections if row["document"] == "docx_goal.txt"
                and 1110 <= int(row["section_start"]) <= 2300]
    if len(appendix) < 35:
        raise AssertionError("F0–F9 appendix section inventory unexpectedly shrank")
    rows = []
    for section in appendix:
        first, last = int(section["section_start"]), int(section["section_end"])
        bound = [review for review in reviews if first <= int(review["document_line"]) <= last]
        context = section["title"] in CONTEXT_ONLY
        if not bound and not context:
            raise AssertionError(f"unmapped formula subsection: {first} {section['title']}")
        rows.append({
            "section_start": first, "section_end": last, "title": section["title"],
            "mapping": "context_heading" if context else "direct_source_review",
            "review_item_ids": "|".join(row["item_id"] for row in bound),
            "review_statuses": "|".join(row["reviewed_status"] for row in bound),
            "source_paths": "|".join(dict.fromkeys(row["source_path"] for row in bound)),
            "scientific_completion": "not_proven",
        })
    path = BANK / "FORMULA_SUBSECTION_COVERAGE_AUDIT.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "completion_proven": False,
        "appendix_sections": len(rows),
        "direct_source_review_sections": sum(row["mapping"] == "direct_source_review" for row in rows),
        "context_headings": sum(row["mapping"] == "context_heading" for row in rows),
        "unmapped_formula_sections": [],
        "reviewed_source_rows": len(reviews),
        "formula_review_sha256": sha(BANK / "FORMULA_IMPLEMENTATION_AUDIT.csv"),
        "section_inventory_sha256": sha(BANK / "DOCUMENT_SECTION_AUDIT.csv"),
        "csv_sha256": sha(path),
        "boundary": "A direct title-to-source review map, not a per-equation numerical proof or native-device acceptance. Reviewed statuses and evidence boundaries remain in FORMULA_IMPLEMENTATION_AUDIT.csv; every subsection retains scientific_completion=not_proven until its own formula and eligible native protocol pass.",
    }
    (BANK / "FORMULA_SUBSECTION_COVERAGE_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
