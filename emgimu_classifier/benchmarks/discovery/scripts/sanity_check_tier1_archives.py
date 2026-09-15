from __future__ import annotations

import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np


def qc(values: np.ndarray, name: str, rate: float) -> dict:
    delta = np.diff(values, axis=0); scale = np.max(np.abs(values), axis=0)
    return {"member": name, "samples": len(values), "channels": values.shape[1], "duration_seconds": len(values) / rate,
        "nan_or_inf_fraction": float(1.0 - np.isfinite(values).mean()),
        "max_flatline_fraction": float(np.mean(np.abs(delta) <= np.finfo(float).eps, axis=0).max()),
        "max_saturation_fraction": float(np.mean(np.abs(values) >= scale * 0.999999, axis=0).max())}


def polyline(values: np.ndarray, top: int) -> str:
    values = np.nan_to_num(values.astype(float)); low, high = values.min(), values.max(); span = max(float(high-low), 1e-12)
    x = np.linspace(70, 970, len(values)); y = top + 175 - (values-low)*175/span
    return '<polyline fill="none" stroke="#2166ac" stroke-width="1" points="' + ' '.join(f'{a:.2f},{b:.2f}' for a,b in zip(x,y)) + '"/>'


def plot(path: Path, values: np.ndarray, rate: float, label: str = '') -> None:
    raw = values[:, 0]; width = max(1, round(rate*.05)); env = np.sqrt(np.convolve(raw*raw, np.ones(width)/width, mode="same"))
    psd = np.abs(np.fft.rfft((raw-raw.mean())*np.hanning(len(raw))))**2/max(len(raw),1)
    panels=[]
    for i,(title,series) in enumerate((("Raw EMG channel 1",raw),("50 ms RMS envelope",env),(f"Periodogram 0-{rate/2:g} Hz",np.log10(psd+1e-300)))):
        top=80+i*230; panels += [f'<text x="70" y="{top-10}" font-size="17" font-family="Arial">{title}</text>',f'<rect x="70" y="{top}" width="900" height="175" fill="white" stroke="#999"/>',polyline(series,top)]
    caption=f'<text x="20" y="25" font-size="13" font-family="Arial">{escape(label)}</text>'
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="760">'+caption+''.join(panels)+'</svg>',encoding="utf-8")


def manus(archive: Path, output: Path) -> None:
    samples=[]; first=None
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        for user in sorted(np.random.default_rng(20260915).choice([*range(3,17),18],3,replace=False)):
            for speed in ("slow","fast"):
                prefix=f"data/u_{user}/s_1/g_flexext_fist/recording_{speed}_"
                member=next(name for name in names if name.startswith(prefix))
                values=np.loadtxt(BytesIO(z.read(member)),delimiter=",",comments="#")[:,:8]
                svg=output/f'semg_manus_u{user}_{speed}_raw_envelope_psd.svg'
                plot(svg,values,200.0,f'User {user}; session 1; fist flexion-extension; speed {speed}; {len(values)/200:g} s')
                first=values if first is None else first; samples.append({**qc(values,member,200.0),'plot':svg.name})
    svg=output/"semg_manus_raw_envelope_psd.svg";plot(svg,first,200.0)
    (output/"semg_manus_sanity.json").write_text(json.dumps({"dataset":"sEMG-MANUS","status":"ok","sample_strategy":"seed 20260915: three random core users; session 1; fist slow/fast","samples":samples,"plot":str(svg)},indent=2),encoding="utf-8")


def shift(archive: Path, output: Path) -> None:
    samples=[];first=None
    with zipfile.ZipFile(archive) as z:
        for subject in sorted(np.random.default_rng(20260915).choice(21,3,replace=False)):
            for domain in ("training","trial_1"):
                member=f"CIILData-main/ElectrodeShift/subject{subject}/{domain}/R_0_C_0.csv"
                values=np.loadtxt(BytesIO(z.read(member)),delimiter=",")
                svg=output/f'electrode_shift_s{subject}_{domain}_raw_envelope_psd.svg'
                plot(svg,values,200.0,f'Subject {subject}; domain {domain}; class 0; repetition 0; {len(values)/200:g} s')
                first=values if first is None else first;samples.append({**qc(values,member,200.0),'plot':svg.name})
    svg=output/"electrode_shift_raw_envelope_psd.svg";plot(svg,first,200.0)
    (output/"electrode_shift_sanity.json").write_text(json.dumps({"dataset":"LibEMG Electrode Shift","status":"ok","sample_strategy":"seed 20260915: three random subjects; before/after trial_1; close rep 0","samples":samples,"plot":str(svg)},indent=2),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--manus",type=Path,required=True);parser.add_argument("--shift",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);manus(args.manus,args.output);shift(args.shift,args.output)
    print(json.dumps({"status":"ok","datasets":["semg_manus","electrode_shift"],"samples_checked":12}))


if __name__ == "__main__": main()
