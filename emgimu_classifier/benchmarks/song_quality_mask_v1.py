"""One-person raw-ADC audit of a source-frozen F9 candidate quality mask."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from benchmarks.song_quality_observability import raw_formal_windows
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.quality_mask_v1 import SourceCalibratedQualityMask
from emgimu.feature_bank.quality_observability import QualityObservabilityFamily


def evaluate(source: Path, output: Path) -> dict:
    data, raw = {}, {}
    for session in ("S01", "S02", "S03", "S04"):
        folder = source / f"2026-09-18_{session}"
        data[session] = load_session(folder, session, filter_mode="causal")
        raw[session] = raw_formal_windows(folder, session, data[session]["trial"])
        print(f"F9 candidate {session}: {len(raw[session])} frozen formal windows", flush=True)
    source_batch = FeatureBatch(np.concatenate((raw["S01"], raw["S02"])), 250.)
    family = QualityObservabilityFamily(
        adc_min=-8388608, adc_max=8388607, line_frequency_hz=50.,
        pre_highpass_available=True, ring_topology=False,
    ).fit(source_batch)
    source_features = family.transform(source_batch)
    mask = SourceCalibratedQualityMask(source_quantile=.995).fit(
        source_features, family.feature_names)
    frozen = pickle.dumps((family, mask))
    result = {
        "status": "one_person_raw_adc_f9_candidate_not_deployed",
        "source_sessions": ["S01", "S02"],
        "validation_session": "S03", "descriptive_final_session": "S04",
        "source_windows": source_batch.windows,
        "source_sha256": {session: data[session]["audit"]["sha256"]
                          for session in ("S01", "S02")},
        "target_sha256": {session: data[session]["audit"]["sha256"]
                          for session in ("S03", "S04")},
        "source_state_sha256": hashlib.sha256(frozen).hexdigest(),
        "availability": mask.available_,
        "thresholds": {key: values.tolist() for key, values in mask.thresholds_.items()},
        "quality_feature_names": list(mask.feature_names),
        "sessions": {},
        "scope": "Source S01/S02-only F9v2 thresholds; S03/S04 read-only. Raw ADC metadata and pre-highpass input are available for this one-person/day recording. The rule and 0.5 illustrative quality flag are candidates, not selected or deployed. S03 collection readiness failed; S04 was previously inspected. Constant-channel corruption is synthetic, not observed hardware-fault prevalence. No new-user, new-day, re-donning or live safety claim.",
    }
    for session in ("S03", "S04"):
        batch = FeatureBatch(raw[session], 250.)
        clean_features = family.transform(batch)
        clean = mask.transform(clean_features, family.feature_names)
        fault = raw[session].copy()
        fault[:, :, 0] = fault[:, :1, 0]
        synthetic = mask.transform(family.transform(FeatureBatch(fault, 250.)),
                                   family.feature_names)
        labels = np.asarray(data[session]["hand"])
        if not np.array_equal(np.unique(labels), ["fist", "index_pinch", "neutral", "open_hand"]):
            raise AssertionError("Song hand-class inventory changed")
        result["sessions"][session] = {
            "windows": len(clean), "readiness": data[session]["audit"]["readiness"],
            "observed_mean_quality": float(np.mean(clean[:, -3])),
            "observed_min_quality_median": float(np.median(clean[:, -2])),
            "observed_any_bad_channel_fraction": float(np.mean(clean[:, -4] >= 1)),
            "hypothetical_min_quality_below_0_5_fraction": float(np.mean(clean[:, -2] < .5)),
            "by_hand": {hand: {"windows": int(np.sum(labels == hand)),
                               "mean_quality": float(np.mean(clean[labels == hand, -3])),
                               "hypothetical_min_quality_below_0_5_fraction":
                               float(np.mean(clean[labels == hand, -2] < .5))}
                        for hand in sorted(set(labels))},
            "synthetic_ch1_constant": {
                "channel_quality_zero_fraction": float(np.mean(synthetic[:, 0] == 0)),
                "any_bad_channel_fraction": float(np.mean(synthetic[:, -4] >= 1)),
                "minimum_quality_below_0_5_fraction": float(np.mean(synthetic[:, -2] < .5)),
            },
        }
    if frozen != pickle.dumps((family, mask)):
        raise AssertionError("target quality transform mutated source state")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("E:/qxy/emg_meta/emg_meta/data/Song"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/song_real8/QUALITY_MASK_V1.json"))
    args = parser.parse_args()
    result = evaluate(args.source, args.output)
    for session, row in result["sessions"].items():
        print(f"F9 candidate {session}: mean quality={row['observed_mean_quality']:.3f}; "
              f"synthetic constant ch1 detected={row['synthetic_ch1_constant']['channel_quality_zero_fraction']:.3f}",
              flush=True)
