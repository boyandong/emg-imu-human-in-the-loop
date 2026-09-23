"""Rebind frozen prediction recovery after a verified append-only result run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


def _record_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def extend(repo: Path, base_ref: str, relative_source: Path, recovery: Path) -> dict:
    repo = repo.resolve()
    source = repo / relative_source
    old_bytes = subprocess.check_output([
        "git", "-c", f"safe.directory={repo.as_posix()}", "show",
        f"{base_ref}:{relative_source.as_posix()}"], cwd=repo)
    new_bytes = source.read_bytes()
    artifact = json.loads(recovery.read_text(encoding="utf-8"))
    # Windows working trees may have CRLF while the committed blob uses LF.
    old_windows_bytes = old_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if artifact["source_csv_sha256"] not in {
            hashlib.sha256(old_bytes).hexdigest(), hashlib.sha256(old_windows_bytes).hexdigest()}:
        raise ValueError("Frozen recovery is not bound to the requested pre-append source")
    old_rows = list(csv.DictReader(io.StringIO(old_bytes.decode("utf-8-sig"))))
    new_rows = list(csv.DictReader(io.StringIO(new_bytes.decode("utf-8-sig"))))
    if (not old_rows or len(new_rows) <= len(old_rows) or
            list(old_rows[0]) != list(new_rows[0]) or new_rows[:len(old_rows)] != old_rows):
        raise ValueError("Result table is not an exact row-preserving append")
    for item in artifact["rows"]:
        index = int(item["source_row_1based"])
        if (index < 1 or index > len(old_rows) or
                item["run_id"] != new_rows[index - 1]["run_id"] or
                item["source_record_sha256"] != _record_hash(new_rows[index - 1])):
            raise ValueError("Frozen recovered row identity changed")
    appended = new_rows[len(old_rows):]
    if any(not row.get("disagreement_rate") for row in appended):
        raise ValueError("New rows require directly recorded prediction disagreement")
    artifact["source_csv_sha256"] = hashlib.sha256(new_bytes).hexdigest()
    artifact["append_only_extension"] = {
        "base_ref": base_ref,
        "unchanged_prior_rows": len(old_rows),
        "appended_rows": len(appended),
        "appended_run_ids": sorted(set(row["run_id"] for row in appended)),
        "rule": "All original parsed records and recovered row identities unchanged; appended rows have their own recorded disagreement rate; no prediction replay or model fit needed",
    }
    recovery.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(artifact["append_only_extension"]), flush=True)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--relative-source", required=True, type=Path)
    parser.add_argument("--recovery", required=True, type=Path)
    args = parser.parse_args()
    extend(args.repo, args.base_ref, args.relative_source, args.recovery)
