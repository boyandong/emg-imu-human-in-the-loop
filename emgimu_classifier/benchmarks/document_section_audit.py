"""Group every specification line into exact reviewed sections with bounded evidence.

This complements the lossless line queue.  It does not turn explanatory text,
examples, formulas, or headings into independent requirements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PRE_SECTIONS = (1, 16, 55, 100, 168, 650, 778, 784, 807, 835, 848, 859,
                905, 943, 976, 1012, 1050, 1072, 1108)
GOAL_SECTIONS = (1, 6, 48, 90, 133, 176, 447, 534, 582, 741, 767, 800,
                 837, 864, 894, 933, 1013, 1052, 1074, 1110, 1274, 1314,
                 1365, 1401, 1405, 1444, 1480, 1535, 1551, 1591, 1626, 1663,
                 1697, 1705, 1777, 1826, 1861, 1867, 1923, 1941, 1968, 1975,
                 1985, 1993, 2005, 2047, 2140, 2146, 2169, 2185, 2205, 2224,
                 2244, 2271, 2301, 2476, 2514, 2563, 2606)


def item(status: str, evidence: str, boundary: str) -> tuple[str, str, str]:
    return status, evidence, boundary


PRE_META = {
    1: item("context", "benchmarks/discovery/BENCHMARK_SELECTION_REPORT.md", "Phase preamble and objective; assessed by the following sections."),
    16: item("partial", "benchmarks/discovery/BENCHMARK_SELECTION_REPORT.md", "Failure-first principles are followed by current selection, but retrospective evidence cannot prove original execution order."),
    55: item("partial", "feature_bank/results/historical_source_recovery_audit.json", "All four available GitHub refs and their histories were searched; exact historical algorithms remain unavailable outside current reference implementations."),
    100: item("partial", "benchmarks/discovery/SCORE_REVIEW.md", "Fourteen retrospective A-H scorecards are inspectable; their timing before acquisition/training is not proven."),
    168: item("partial", "benchmarks/discovery/DATASET_CANDIDATES.csv", "Mandatory candidates were researched and bounded; historical DS2 archive identity and some secondary provenance remain incomplete."),
    650: item("verified_narrow", "benchmarks/discovery/FAILURE_BENCHMARK_MATRIX.md", "Capability mapping is explicit and missing axes remain N/A; this does not validate every proposed secondary dataset."),
    778: item("partial", "benchmarks/discovery/DATASET_MANIFEST.json", "Tiered acquisition is represented in the manifest, but original tier-by-tier timing is not reconstructable."),
    784: item("verified_narrow", "benchmarks/discovery/DATASET_CANDIDATES.csv", "Metadata-only candidate evidence is retained."),
    807: item("partial", "benchmarks/discovery/DATASET_MANIFEST.json", "Six core archives are present and size checked; exact historical DS2 is absent."),
    835: item("partial", "benchmarks/discovery/DATASET_CANDIDATES.csv", "Secondary candidates are scored; not every candidate was downloaded because capability gaps and cost boundaries are recorded."),
    848: item("verified_narrow", "benchmarks/discovery/DISK_USAGE.json", "Large/future sources remain deferred with cost evidence."),
    859: item("partial", "benchmarks/discovery/DATASET_MANIFEST.json", "Staged paths, sizes and extraction boundaries are recorded; old acquisition logs cannot prove every transient download safety step."),
    905: item("verified_narrow", "benchmarks/discovery/DATASET_MANIFEST.json", "Raw/processed/report paths are explicit and raw archives are not committed."),
    943: item("verified_narrow", "benchmarks/discovery/SANITY_AUDIT.json", "Six native sanity reports and 36 plot hashes were reverified on the fixed sample policy."),
    976: item("verified_narrow", "benchmarks/discovery/GESTURE_ONTOLOGY.md", "Native labels and strict common-label mappings are separated."),
    1012: item("verified_narrow", "benchmarks/discovery/SENSOR_LAYOUTS.md", "Known channel topology and unknown geometry are recorded without fabricating rings."),
    1050: item("partial", "benchmarks/discovery/SCORE_REVIEW.md", "A-H comparisons are evidence bounded; exact historical DS2 still prevents a complete final ranking."),
    1072: item("partial", "benchmarks/discovery/BENCHMARK_SELECTION_REPORT.md", "The selected suite covers complementary failures, with explicit confounds and unsupported axes."),
    1108: item("partial", "benchmarks/discovery/DISCOVERY_DELIVERY_AUDIT.json", "Required discovery artifacts are present and hashed; historical DS2 identity and original phase ordering remain unproven."),
}


GOAL_META = {
    1: item("context", "feature_bank/REPORT.md", "Experiment objective preamble; assessed by the following sections."),
    6: item("partial", "feature_bank/REPORT.md", "The report answers the research questions with negative as well as positive results; exact historical-family evidence is incomplete."),
    48: item("verified_narrow", "feature_bank/delivery/conditional_incremental.csv", "Held-out delta loss/F1/Brier proxies are delivered; they are not direct conditional mutual-information estimates."),
    90: item("verified_narrow", "feature_bank/EXPERIMENT_CAPABILITIES.md", "Native capabilities and N/A boundaries are explicit; a later single-participant own-device exploratory study does not meet formal acceptance."),
    133: item("partial", "feature_bank/REPRODUCIBILITY_METADATA_AUDIT.json", "Current runs preserve splits/config/state; exact old DS2/X1-H/RLCS/CES/Frequency artifacts cannot be frozen because they were not recovered."),
    176: item("partial", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.json", "F0-F9 APIs and candidate formulas are audited; several historical identities and native calibrated-body evidence remain unavailable."),
    447: item("partial", "feature_bank/results/calibration_burden_audit.json", "Personal calibration and supported budgets are evaluated across native tasks; unsupported budgets and own-device validation remain explicit."),
    534: item("verified_narrow", "feature_bank/results/calibration_burden_audit.json", "Independent calibrated providers, source OOF temperatures, reliability shrinkage and late fusion have native evidence; no universal gain is claimed."),
    582: item("partial", "feature_bank/STAGE3_4_CLAUSE_AUDIT.json", "Finite staged screening, complementarity and selected interactions are delivered; exact historical named pairs remain unavailable."),
    741: item("partial", "feature_bank/REPORT.md", "Family roles are interpreted per failure/calibration; exact historical families prevent a final universal taxonomy."),
    767: item("verified_narrow", "feature_bank/delivery/feature_family_results.csv", "Requested classification and calibration metrics are present where supported; confidence intervals are not available for every legacy run."),
    800: item("partial", "feature_bank/results/full_system_robustness_vector.csv", "Seven available axes report mean and minimum boundaries; own-device real-noise and unavailable native axes remain N/A."),
    837: item("verified_narrow", "feature_bank/results/calibration_burden.csv", "0/1/2/5-shot costs and unsupported budgets are recorded without pooling incompatible protocols."),
    864: item("verified_narrow", "feature_bank/results/validation.json", "Leakage-sensitive tests and split/hash audits pass; tests cannot prove historical execution order or hardware accuracy."),
    894: item("partial", "feature_bank/REPORT.md", "All phases have current evidence, but retrospective recovery cannot prove the original chronological order."),
    933: item("partial", "feature_bank/delivery/SCHEMA_AUDIT.json", "All named files exist and canonical rows preserve sources; schema is complete with explicit evidence N/A fields."),
    1013: item("partial", "feature_bank/REPORT.md", "A-H questions have evidence-bounded answers; real-device and historical-family gaps constrain conclusions."),
    1052: item("partial", "feature_bank/REPORT.md", "Prior claims are retained as historical statements and separated from current reference evidence."),
    1074: item("partial", "feature_bank/ENGINEERING_CLAUSE_AUDIT.csv", "Engineering clauses have focused evidence; full scientific acceptance is assessed separately."),
}

# Precise-definition appendix sections share the formula audit except for the
# calibration/fusion/diagnostic boundaries, which have dedicated evidence.
for line in GOAL_SECTIONS:
    if line >= 1110 and line not in GOAL_META:
        GOAL_META[line] = item(
            "partial", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.json",
            "Formula/API boundary and dimensions are reviewed; candidate/reference status is distinct from recovered historical equivalence and native validation.")
GOAL_META.update({
    2301: item("partial", "feature_bank/results/calibration_burden_audit.json", "Personal normalization/prototypes/reliability/shrinkage are implemented with held-out native evidence; not every dataset supports every budget."),
    2476: item("partial", "feature_bank/results/family_specific_session_shift_audit.json", "Session updates and immutable long profiles have wearing/MANUS evidence; wearing domains are not calendar sessions and the later own-device study does not validate this full update method."),
    2514: item("verified_narrow", "feature_bank/results/quality_unknown_replay.json", "Late fusion contracts, quality weighting and explicit Unknown are tested and replayed; current Unknown rule has no demonstrated accuracy gain."),
    2563: item("partial", "feature_bank/delivery/error_complementarity.csv", "Nuisance, separation, conditional, complementarity and calibration diagnostics exist broadly; not every metric is natively identifiable for every family/dataset."),
    2606: item("verified_narrow", "feature_bank/FORMULA_IMPLEMENTATION_AUDIT.json", "Channel/topology/fit/dimension boundaries have tests and audits; later real 8-channel preliminary scores do not validate every family or live deployment."),
})


def rows_for(document: Path, starts: tuple[int, ...], metadata: dict[int, tuple[str, str, str]], evidence_root: Path) -> list[dict]:
    raw = document.read_bytes(); lines = raw.decode("utf-8-sig").splitlines()
    if starts[0] != 1 or sorted(set(starts)) != list(starts):
        raise ValueError("section starts must be a sorted partition beginning at line one")
    rows = []
    for index, start in enumerate(starts):
        end = (starts[index + 1] - 1) if index + 1 < len(starts) else len(lines)
        status, evidence, boundary = metadata[start]
        path = evidence_root / evidence
        if not path.is_file():
            raise FileNotFoundError(path)
        nonblank = [number for number in range(start, end + 1) if lines[number - 1].strip()]
        title_number = next(iter(nonblank), start)
        rows.append({"document": document.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                     "section_start": start, "section_end": end,
                     "title": lines[title_number - 1].strip(), "nonblank_lines": len(nonblank),
                     "status": status, "evidence": evidence,
                     "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "boundary": boundary})
    covered = {number for row in rows for number in range(row["section_start"], row["section_end"] + 1)
               if lines[number - 1].strip()}
    expected = {number for number, line in enumerate(lines, 1) if line.strip()}
    if covered != expected:
        raise AssertionError(f"section coverage mismatch: {document}")
    return rows


def build(pre: Path, goal: Path, output: Path, evidence_root: Path) -> None:
    rows = rows_for(pre, PRE_SECTIONS, PRE_META, evidence_root)
    rows += rows_for(goal, GOAL_SECTIONS, GOAL_META, evidence_root)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "DOCUMENT_SECTION_AUDIT.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in sorted({row["status"] for row in rows})}
    artifact = {"completion_proven": False, "sections": len(rows),
                "indexed_nonblank_lines": sum(row["nonblank_lines"] for row in rows),
                "status_counts": counts, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "documents": {document.name: hashlib.sha256(document.read_bytes()).hexdigest()
                              for document in (pre, goal)},
                "boundary": "Every nonblank source line belongs to exactly one reviewed section. Section status applies to the contextual clause group, not independently to each explanatory/formula line. Partial remains incomplete."}
    (output / "DOCUMENT_SECTION_AUDIT.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sections": len(rows), "nonblank_lines": artifact["indexed_nonblank_lines"],
                      "status_counts": counts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("pre", type=Path)
    parser.add_argument("goal", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(); build(args.pre, args.goal, args.output, args.evidence_root)
