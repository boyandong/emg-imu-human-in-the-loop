from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np


def _emg(sample: dict) -> np.ndarray:
    return np.column_stack([sample["emg"][f"ch{index}"] for index in range(1, 9)]).astype(float)


def _polyline(values: np.ndarray, left: float, top: float, width: float, height: float) -> str:
    values = np.nan_to_num(np.asarray(values, dtype=float))
    low, high = float(values.min()), float(values.max())
    span = max(high - low, np.finfo(float).eps)
    x = np.linspace(left, left + width, len(values))
    y = top + height - (values - low) * height / span
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    return f'<polyline fill="none" stroke="#2166ac" stroke-width="1" points="{points}"/>'


def _write_svg(path: Path, values: np.ndarray) -> None:
    raw = values[:, 0]
    envelope = np.sqrt(np.convolve(raw * raw, np.ones(10) / 10, mode="same"))
    tapered = (raw - raw.mean()) * np.hanning(len(raw))
    psd = np.abs(np.fft.rfft(tapered)) ** 2 / max(len(tapered), 1)
    panels = []
    for index, (title, series) in enumerate((("Raw EMG channel 1", raw), ("50 ms RMS envelope", envelope), ("Periodogram 0-100 Hz", np.log10(psd + np.finfo(float).tiny)))):
        top = 35 + index * 230
        panels += [f'<text x="70" y="{top}" font-size="17" font-family="Arial">{title}</text>', f'<rect x="70" y="{top + 15}" width="900" height="175" fill="white" stroke="#999"/>', _polyline(series, 70, top + 15, 900, 175)]
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="730">' + ''.join(panels) + '</svg>', encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, first = [], None
    with zipfile.ZipFile(args.archive) as handle:
        names = handle.namelist()
        json_count = sum(name.endswith(".json") for name in names)
        if json_count != 612:
            raise ValueError(f"expected 612 user JSON files, found {json_count}")
        for user in (1, 153, 306):
            member = f"EMG-EPN612 Dataset/trainingJSON/user{user}/user{user}.json"
            payload = json.loads(handle.read(member))
            for gesture in ("fist", "open"):
                sample_id, sample = next((key, value) for key, value in payload["trainingSamples"].items() if value["gestureName"] == gesture)
                values = _emg(sample)
                first = values if first is None else first
                delta = np.diff(values, axis=0)
                scale = np.max(np.abs(values), axis=0)
                results.append({
                    "member": member, "trial": sample_id, "gesture": gesture, "samples": len(values), "channels": values.shape[1],
                    "duration_seconds": len(values) / 200.0, "nan_or_inf_fraction": float(1.0 - np.isfinite(values).mean()),
                    "max_flatline_fraction": float(np.mean(np.abs(delta) <= np.finfo(float).eps, axis=0).max()),
                    "max_saturation_fraction": float(np.mean(np.abs(values) >= scale * 0.999999, axis=0).max()),
                })
    plot = args.output / "epn612_raw_envelope_psd.svg"
    _write_svg(plot, first)
    report = {"dataset": "EMG-EPN612", "status": "ok", "sampling_rate_hz": 200, "users": 612,
              "sample_strategy": "trainingJSON users 1, 153, 306; first fist and open trial", "samples": results, "plot": str(plot)}
    (args.output / "epn612_sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "json_files": 612, "samples_checked": len(results)}))


if __name__ == "__main__":
    main()
