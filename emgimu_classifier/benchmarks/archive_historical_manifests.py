"""Preserve superseded small manifests outside the current-results namespace."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_REVISION = "ffb3999"
PREFIX = "emgimu_classifier/feature_bank/results/manifests/"


def git(root, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root)


def archive(repo, current, historical):
    repo = repo.resolve()
    current = current.resolve()
    historical = historical.resolve()
    if not current.is_relative_to(repo) or not historical.is_relative_to(repo):
        raise ValueError("Manifest output must be inside the repository")
    changed = git(repo, "diff", "--name-status", SOURCE_REVISION, "HEAD", "--", PREFIX).decode().splitlines()
    deleted = sorted(line[2:] for line in changed if line.startswith("D\t"))
    if not deleted or any(not path.startswith(PREFIX) or not path.endswith(".json") for path in deleted):
        raise ValueError("Historical manifest deletion set is unexpected")
    historical.mkdir(parents=True, exist_ok=True)
    records = []
    for old_path in deleted:
        basename = old_path.removeprefix(PREFIX)
        if (current / basename).exists():
            raise ValueError(f"Still an active manifest: {basename}")
        original = git(repo, "show", f"{SOURCE_REVISION}:{old_path}")
        json.loads(original)
        destination = historical / basename
        if destination.exists() and destination.read_bytes() != original:
            raise ValueError(f"Archived evidence changed: {basename}")
        destination.write_bytes(original)
        records.append({"historical_path": old_path, "archive_path": destination.relative_to(repo).as_posix(),
                        "bytes": len(original), "sha256": hashlib.sha256(original).hexdigest(),
                        "source_revision": SOURCE_REVISION})
    audit = {"status": "historical_evidence_preserved_separately", "completion_proven": False,
             "source_revision": SOURCE_REVISION, "files": records, "count": len(records),
             "boundary": "These superseded manifests are not part of the active consolidated result registry. "
                         "Their preservation does not validate superseded metrics or restore missing historical DS2 artifacts."}
    (historical / "ARCHIVE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "files": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("historical", type=Path)
    args = parser.parse_args()
    archive(args.repo, args.current, args.historical)
