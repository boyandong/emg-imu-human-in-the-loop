"""Inventory only first-segment file metadata of public DS2 v8 TDMS members.

This deliberately does not infer force levels or join TDMS streams to the
aggregate MAT trial order. NI's TDMS structure documents the lead-in and
length-prefixed object/property fields used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path


NI_FORMAT = "https://www.ni.com/en/support/documentation/supplemental/07/tdms-file-format-internal-structure.html"
MEMBER = re.compile(r"SEMG-(\d{2})/SEMG-(\d{2})_Mv([1-5])(?:_Mv([1-5]))?\.tdms$")


def u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated TDMS metadata integer")
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def tdms_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = u32(data, offset)
    if length > 4096 or offset + length > len(data):
        raise ValueError("invalid TDMS metadata string length")
    return data[offset:offset + length].decode("utf-8"), offset + length


def first_file_name(data: bytes) -> tuple[str, int, int, int]:
    if len(data) < 40 or data[:4] != b"TDSm":
        raise ValueError("TDMS segment lead-in missing")
    toc, version, next_offset, raw_offset = struct.unpack_from("<IIQQ", data, 4)
    if toc & 0x40 or not (toc & 0x02) or version not in (4712, 4713):
        raise ValueError("unsupported first TDMS metadata segment")
    if raw_offset == 0 or raw_offset + 28 > len(data):
        raise ValueError("first TDMS metadata region incomplete")
    count, offset = u32(data, 28)
    if count < 1 or count > 100:
        raise ValueError("unexpected first-segment object count")
    path, offset = tdms_string(data, offset)
    if path != "/":
        raise ValueError("first TDMS object is not the file object")
    raw_index, offset = u32(data, offset)
    if raw_index != 0xFFFFFFFF:
        raise ValueError("file object unexpectedly has a raw data index")
    property_count, offset = u32(data, offset)
    properties = {}
    for _ in range(property_count):
        name, offset = tdms_string(data, offset)
        dtype, offset = u32(data, offset)
        if dtype != 32:
            raise ValueError(f"unsupported first-file property type: {dtype}")
        value, offset = tdms_string(data, offset)
        properties[name] = value
    if "name" not in properties:
        raise ValueError("first TDMS file object has no name property")
    return properties["name"], count, version, int(raw_offset)


def audit(archive: Path, output: Path) -> dict:
    rows = []
    with zipfile.ZipFile(archive) as handle:
        members = [item for item in handle.infolist() if item.filename.lower().endswith(".tdms")]
        for item in members:
            match = MEMBER.fullmatch(item.filename)
            if match is None or match.group(1) != match.group(2):
                raise ValueError(f"unexpected TDMS member path: {item.filename}")
            with handle.open(item) as member:
                lead = member.read(28)
                if len(lead) != 28 or lead[:4] != b"TDSm":
                    raise ValueError(f"bad TDMS lead-in: {item.filename}")
                raw_offset = struct.unpack_from("<Q", lead, 20)[0]
                if raw_offset > 1024 * 1024:
                    raise ValueError(f"TDMS first metadata too large: {item.filename}")
                metadata = lead + member.read(raw_offset)
            name, objects, version, parsed_offset = first_file_name(metadata)
            if parsed_offset != raw_offset:
                raise ValueError(f"TDMS raw offset changed while parsing: {item.filename}")
            rows.append({"member": item.filename, "subject_folder": match.group(1),
                         "movement_filename_indices": ",".join(group for group in match.groups()[2:] if group),
                         "first_segment_file_name": name, "tdms_version": version,
                         "first_segment_objects": objects, "first_metadata_bytes": raw_offset,
                         "member_uncompressed_bytes": item.file_size,
                         "member_crc32": f"{item.CRC:08x}"})
    rows.sort(key=lambda row: row["member"])
    if len(rows) != 97:
        raise ValueError(f"public v8 TDMS member count changed: {len(rows)}")
    movements_by_subject = {}
    for row in rows:
        movements_by_subject.setdefault(row["subject_folder"], set()).update(
            row["movement_filename_indices"].split(","))
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "DS2_TDMS_FIRST_METADATA.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "first_segment_file_metadata_only",
              "archive": str(archive), "archive_bytes_checked": archive.stat().st_size,
              "tdms_members": len(rows),
              "subject_folders": sorted(set(row["subject_folder"] for row in rows)),
              "missing_filename_movement_indices_by_subject": {
                  subject: sorted(set("12345") - indices)
                  for subject, indices in sorted(movements_by_subject.items())
                  if set("12345") - indices},
              "filename_movement_indices": dict(sorted(Counter(
                  index for row in rows for index in row["movement_filename_indices"].split(",")).items())),
              "first_segment_name_values": dict(sorted(Counter(
                  row["first_segment_file_name"] for row in rows).items())),
              "combined_movement_files": [row["member"] for row in rows
                                          if "," in row["movement_filename_indices"]],
              "format_reference": NI_FORMAT,
              "csv": csv_path.name,
              "boundary": "First-segment file name and filename indices are provenance clues, not validated labels for aggregate MAT trials. No per-trial force level, repetition, or subject-to-MAT join is established. No TDMS waveform or later-segment property was decoded. Historical experiment identity remains unproven."}
    (output / "DS2_TDMS_FIRST_METADATA_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("tdms_members", "subject_folders",
                                                  "filename_movement_indices", "combined_movement_files")}),
          flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.archive, args.output)
