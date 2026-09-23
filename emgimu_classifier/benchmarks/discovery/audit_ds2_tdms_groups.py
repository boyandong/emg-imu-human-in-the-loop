"""Read every public DS2 v8 TDMS group's metadata without loading waveform data.

Requires the optional ``tdms`` extra (npTDMS). Group inventory is provenance
evidence, not a mapping of the aggregate MAT trials or subjective force levels.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import nptdms
import numpy as np


GROUP_NAME = re.compile(r"Group Name(?: #(\d+))?$")
EXPECTED_CHANNELS = ("Sensor 1", "Sensor 2", "Sensor 3")


def audit(archive: Path, output: Path) -> dict:
    rows = []
    property_signatures = Counter()
    file_property_keys = Counter()
    group_property_keys = Counter()
    channel_property_keys = Counter()
    increments = []
    with zipfile.ZipFile(archive) as handle:
        members = sorted((item for item in handle.infolist()
                          if item.filename.lower().endswith(".tdms")),
                         key=lambda item: item.filename)
        for item in members:
            # BytesIO supports the seeks used by npTDMS. read_metadata traverses
            # all TDMS segments but does not decode the three waveform arrays.
            tdms = nptdms.TdmsFile.read_metadata(io.BytesIO(handle.read(item)))
            file_property_keys.update(tdms.properties.keys())
            for ordinal, group in enumerate(tdms.groups()):
                if GROUP_NAME.fullmatch(group.name) is None:
                    raise ValueError(f"unexpected group name: {item.filename}/{group.name}")
                channels = group.channels()
                if tuple(channel.name for channel in channels) != EXPECTED_CHANNELS:
                    raise ValueError(f"unexpected channels: {item.filename}/{group.name}")
                lengths = tuple(len(channel) for channel in channels)
                if len(set(lengths)) != 1:
                    raise ValueError(f"channel lengths differ: {item.filename}/{group.name}")
                group_property_keys.update(group.properties.keys())
                for channel in channels:
                    channel_property_keys.update(channel.properties.keys())
                    property_signatures[tuple(sorted(channel.properties))] += 1
                    if "wf_increment" in channel.properties:
                        increments.append(float(channel.properties["wf_increment"]))
                rows.append({"member": item.filename, "group_ordinal_zero_based": ordinal,
                             "group_name": group.name, "channel_count": len(channels),
                             "samples_per_channel": lengths[0],
                             "group_property_count": len(group.properties)})
    if len(members) != 97 or len(rows) != 3210:
        raise ValueError(f"public v8 group count changed: {len(members)} files / {len(rows)} groups")
    if len(property_signatures) != 1 or len(increments) != 3 * len(rows):
        raise ValueError("channel metadata contract varies or sampling increment missing")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "DS2_TDMS_GROUPS.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lengths = np.asarray([row["samples_per_channel"] for row in rows], dtype=np.int64)
    increments = np.asarray(increments)
    report = {
        "status": "all_tdms_group_metadata_parsed_no_aggregate_mat_or_force_join",
        "archive": str(archive), "parser": f"npTDMS {nptdms.__version__}",
        "reader_reference": "https://nptdms.readthedocs.io/en/stable/reading.html",
        "tdms_members": len(members), "groups": len(rows),
        "channels_per_group": 3, "channel_samples_min": int(lengths.min()),
        "channel_samples_median": float(np.median(lengths)),
        "channel_samples_max": int(lengths.max()),
        "channel_sample_length_top_counts": [
            {"samples": int(samples), "groups": count}
            for samples, count in Counter(lengths.tolist()).most_common(12)],
        "groups_with_under_15000_samples": int(np.sum(lengths < 15000)),
        "groups_with_at_least_15000_samples": int(np.sum(lengths >= 15000)),
        "groups_with_exactly_15000_samples": int(np.sum(lengths == 15000)),
        "group_property_count_distribution": dict(sorted(Counter(
            row["group_property_count"] for row in rows).items())),
        "file_property_keys": dict(sorted(file_property_keys.items())),
        "group_property_keys": dict(sorted(group_property_keys.items())),
        "channel_property_keys": dict(sorted(channel_property_keys.items())),
        "channel_property_signature_count": len(property_signatures),
        "wf_increment_seconds_min": float(increments.min()),
        "wf_increment_seconds_median": float(np.median(increments)),
        "wf_increment_seconds_max": float(increments.max()),
        "csv": csv_path.name,
        "boundary": "Full TDMS segment metadata and channel lengths, not waveform values, were read. Group properties may be empty while filenames/root names suggest movements; neither subjective force level nor inclusion/order in Data_all_Raw.mat is encoded as a verified join. A 3210-to-2863 group/trial count mismatch forbids positional labels. Historical input identity remains unproven."
    }
    (output / "DS2_TDMS_GROUP_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "tdms_members", "groups", "groups_with_at_least_15000_samples",
        "groups_with_exactly_15000_samples", "group_property_count_distribution",
        "wf_increment_seconds_median")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.archive, args.output)
