"""Rerun four recent Song diagnostics in isolation and compare exact artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


STUDIES = (
    ("song_temperature_study.py", "TEMPERATURE_STUDY.json", True, False),
    ("song_gain_sensitivity.py", "GAIN_SENSITIVITY.json", True, True),
    ("song_quality_observability.py", "QUALITY_OBSERVABILITY.json", False, False),
    ("song_signal_calibration.py", "SIGNAL_CALIBRATION.json", True, True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(source: Path, collection_root: Path, saved: Path, output: Path) -> dict:
    classifier = Path(__file__).resolve().parents[1]
    source, collection_root, saved = (path.resolve() for path in
                                      (source, collection_root, saved))
    if not all((source / f"2026-09-18_{session}" / "session.h5").is_file()
               for session in ("S01", "S02", "S03", "S04")):
        raise FileNotFoundError("Four Song source sessions are required")
    reference = saved / "PROBABILITY_CALIBRATION_AUDIT.json"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(classifier / "src"), str(classifier)))
    recorded = []
    with tempfile.TemporaryDirectory(prefix="codex-song-recheck-") as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError("Temporary reproduction directory escaped OS temp root")
        for script, artifact, needs_collection, needs_reference in STUDIES:
            script_path = classifier / "benchmarks" / script
            rerun = temporary / artifact
            command = [sys.executable, str(script_path), "--source", str(source)]
            if needs_collection:
                command += ["--collection-root", str(collection_root)]
            if script == "song_temperature_study.py":
                command += ["--bundle", str(collection_root / "models" / "song_real8_f0_spd")]
            if needs_reference:
                command += ["--reference", str(reference)]
            command += ["--output", str(rerun)]
            process = subprocess.run(command, cwd=classifier, env=env,
                                     text=True, capture_output=True, check=False)
            if process.returncode:
                raise RuntimeError(f"{script} failed ({process.returncode}):\n"
                                   f"{process.stdout[-2000:]}\n{process.stderr[-2000:]}")
            saved_path = saved / artifact
            rerun_sha = sha256(rerun)
            saved_sha = sha256(saved_path)
            if rerun_sha != saved_sha:
                raise ValueError(f"Reproduced artifact differs: {artifact}")
            recorded.append({
                "script": f"benchmarks/{script}",
                "script_sha256": sha256(script_path),
                "saved_artifact": f"benchmarks/song_real8/{artifact}",
                "saved_and_reproduced_sha256": saved_sha,
                "exact_byte_match": True,
                "command_argv": command[:-1] + ["<temporary-output-path>"],
                "command_template": " ".join(command[:-1] + ["<temporary-output-path>"]),
            })
            print(f"[{len(recorded)}/{len(STUDIES)}] exact: {artifact}", flush=True)
    source_hashes = {
        session: sha256(source / f"2026-09-18_{session}" / "session.h5")
        for session in ("S01", "S02", "S03", "S04")
    }
    bundle_hashes = {
        name: {
            "model_sha256": sha256(collection_root / "models" / name / "song_f0_model.json"),
            "manifest_sha256": sha256(collection_root / "models" / name / "song_manifest.json"),
        }
        for name in ("song_real8_f0", "song_real8_f0_spd")
    }
    result = {
        "status": "four_recent_song_diagnostics_exactly_reproduced",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pythonpath": env["PYTHONPATH"],
        "source_hdf5_sha256": source_hashes,
        "local_bundle_sha256": bundle_hashes,
        "reference_sha256": sha256(reference),
        "studies": recorded,
        "boundary": "Exact-byte rerun of four recent diagnostics in the current Windows/Python environment. It does not rerun older Song experiments, retrain model bundles, validate physical USB or establish cross-person/day accuracy. Raw Song HDF5 and ignored local model bundles must be supplied separately to reproduce elsewhere.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--saved", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.collection_root, args.saved, args.output)
