from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from emgforce.processing.global_alignment_template import build_global_template
from emgforce.processing.template_alignment import METHOD_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a participant-balanced v3 global alignment template.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-participants", type=int, default=2)
    args = parser.parse_args()
    paths = []
    for path in sorted(args.data_root.rglob("session_meta_aligned.hdf5")):
        with h5py.File(path, "r") as handle:
            algorithm = handle.get("alignment_events", {}).attrs.get("algorithm", "") \
                if "alignment_events" in handle else ""
            if isinstance(algorithm, bytes):
                algorithm = algorithm.decode("utf-8")
            if algorithm == METHOD_NAME:
                paths.append(path)
    print(build_global_template(
        paths, args.output, minimum_participants=args.minimum_participants))


if __name__ == "__main__":
    main()
