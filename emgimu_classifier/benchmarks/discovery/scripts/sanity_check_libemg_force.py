from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


FILE_RE = re.compile(
    r"S(?P<subject>\d+)_(?P<condition>(?:\d+P|MVC|Light|Medium|Hard|Ramp))_C(?P<class_id>\d+)_R(?P<rep>\d+)\.csv$",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",")
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns, got {values.shape}")
    return values


def choose_samples(root: Path) -> list[Path]:
    candidates = []
    for subject in (1, 5, 10):
        folder = root / f"S{subject}"
        for condition in ("20P", "80P"):
            match = folder / f"S{subject}_{condition}_C1_R1.csv"
            if not match.is_file():
                raise FileNotFoundError(match)
            candidates.append(match)
    return candidates


def _polyline(values: np.ndarray, left: float, top: float, width: float, height: float) -> str:
    finite = np.nan_to_num(np.asarray(values, dtype=float))
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, np.finfo(float).eps)
    x = np.linspace(left, left + width, len(finite))
    y = top + height - (finite - low) * height / span
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    return f'<polyline fill="none" stroke="#2166ac" stroke-width="1" points="{points}"/>'


def write_svg(path: Path, raw: np.ndarray, envelope: np.ndarray, frequency: np.ndarray, psd: np.ndarray) -> None:
    panels = []
    for index, (title, values) in enumerate(
        (("Raw EMG channel 1", raw), ("50 ms RMS envelope", envelope), ("Periodogram 0-500 Hz", np.log10(psd + np.finfo(float).tiny)))
    ):
        top = 35 + index * 230
        panels.append(f'<text x="70" y="{top}" font-size="17" font-family="Arial">{title}</text>')
        panels.append(f'<rect x="70" y="{top + 15}" width="900" height="175" fill="white" stroke="#999"/>')
        panels.append(_polyline(values, 70, top + 15, 900, 175))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="730" viewBox="0 0 1040 730">' + ''.join(panels) + '</svg>'
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    files = sorted(args.root.glob("S*/*.csv"))
    parsed = [FILE_RE.match(path.name) for path in files]
    if len(files) != 3220 or any(match is None for match in parsed):
        raise ValueError("unexpected LibEMG Contraction Intensity file inventory")

    sampled = []
    for path in choose_samples(args.root):
        values = load_csv(path)
        finite = np.isfinite(values)
        differences = np.diff(values, axis=0)
        flatline = np.mean(np.abs(differences) <= np.finfo(float).eps, axis=0)
        scale = np.max(np.abs(values), axis=0)
        saturation = np.mean(np.abs(values) >= (scale * 0.999999), axis=0)
        sampled.append(
            {
                "path": path.relative_to(args.root).as_posix(),
                "samples": int(values.shape[0]),
                "channels": 8,
                "duration_seconds": float(values.shape[0] / 1000.0),
                "nan_or_inf_fraction": float(1.0 - finite.mean()),
                "max_flatline_fraction": float(flatline.max()),
                "max_saturation_fraction": float(saturation.max()),
            }
        )

    first = load_csv(choose_samples(args.root)[0])
    time = np.arange(min(len(first), 3000)) / 1000.0
    view = first[: len(time)]
    envelope = np.sqrt(np.convolve(view[:, 0] ** 2, np.ones(50) / 50, mode="same"))
    centered = view[:, 0] - np.mean(view[:, 0])
    tapered = centered * np.hanning(len(centered))
    frequency = np.fft.rfftfreq(len(tapered), d=1.0 / 1000.0)
    power = np.abs(np.fft.rfft(tapered)) ** 2 / max(len(tapered), 1)
    plot_path = args.output / "libemg_force_raw_envelope_psd.svg"
    write_svg(plot_path, view[:, 0], envelope, frequency, power)

    report = {
        "dataset": "LibEMG Contraction Intensity",
        "status": "ok",
        "sampling_rate_hz": 1000,
        "subjects": 10,
        "csv_files": len(files),
        "sample_strategy": "subjects 1, 5, 10; 20P and 80P; class 1 rep 1",
        "samples": sampled,
        "plot": str(plot_path),
    }
    (args.output / "libemg_force_sanity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "files": len(files), "samples_checked": len(sampled)}))


if __name__ == "__main__":
    main()
