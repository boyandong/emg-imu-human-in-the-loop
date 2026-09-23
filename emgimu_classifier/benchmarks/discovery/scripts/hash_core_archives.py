"""Fresh streaming digest check of six immutable core benchmark archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def run(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = [entry for entry in manifest["datasets"] if entry["status"] == "downloaded_verified"]
    if len(datasets) != 6:
        raise ValueError("expected six core archives")
    results = []
    for entry in datasets:
        path = Path(entry["path"])
        before = path.stat()
        if before.st_size != entry["size"]:
            raise ValueError(f"size changed before hashing: {entry['id']}")
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        processed = 0
        next_percent = 10
        print(f"hash {entry['id']}: 0%", flush=True)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
                md5.update(chunk)
                processed += len(chunk)
                percent = processed * 100 // before.st_size
                if percent >= next_percent:
                    print(f"hash {entry['id']}: {percent}%", flush=True)
                    next_percent = (percent // 10 + 1) * 10
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError(f"file changed while hashing: {entry['id']}")
        if sha.hexdigest().lower() != entry["sha256"].lower():
            raise ValueError(f"SHA-256 differs from manifest: {entry['id']}")
        if entry.get("md5") and md5.hexdigest().lower() != entry["md5"].lower():
            raise ValueError(f"MD5 differs from manifest: {entry['id']}")
        results.append({"id": entry["id"], "path": str(path), "size": processed,
                        "mtime_ns_after": after.st_mtime_ns,
                        "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
                        "manifest_sha256_match": True,
                        "manifest_md5_match": bool(entry.get("md5"))})
    audit = {"status": "all_six_core_archives_freshly_hashed_and_matched",
             "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "archive_count": len(results), "bytes_hashed": sum(row["size"] for row in results),
             "archives": results,
             "boundary": "Hashes confirm current archive bytes against recorded manifests. They do not prove original experiment chronology, dataset semantics, full extracted-file quality, or DS2 historical identity."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"all six archives matched ({audit['bytes_hashed']} bytes)", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)
