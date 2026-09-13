from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from emgimu.datasets.adapters.grabmyo import GrabMyoAdapter, audit_grabmyo_source
from emgimu.datasets.hla_schema import check_hla_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and atomically adapt a complete official GRABMyo download",
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    audit = audit_grabmyo_source(args.source_root, require_complete=True)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.output_root.parent)
    print(json.dumps({
        "stage": "source_audit", **audit,
        "source_bytes": sum(
            path.stat().st_size for path in args.source_root.rglob("*") if path.is_file()
        ),
        "output_disk_free_bytes": disk.free,
    }, indent=2), flush=True)
    result = GrabMyoAdapter().adapt(args.source_root, args.output_root)
    integrity = check_hla_dataset(args.output_root)
    summary = {
        "stage": "adaptation_complete",
        "dataset_id": result.dataset_id,
        "physical_trials": result.trial_count,
        "manifest": str(result.manifest_path),
        "warnings": result.warnings,
        "integrity": integrity,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
