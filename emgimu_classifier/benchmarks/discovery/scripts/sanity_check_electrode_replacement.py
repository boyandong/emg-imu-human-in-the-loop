"""Selective native recording sanity; never invent interval or trial labels."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np
import py7zr

from emgimu.datasets.electrode_replacement import MOVEMENTS, load_electrode_replacement_recording, parse_recording_name
from sanity_check_tier1_archives import qc, plot


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[5]
    root = args.root.resolve()
    audit = args.audit.resolve()
    if not root.is_relative_to(workspace) or not audit.is_relative_to(workspace):
        raise ValueError('Output must remain within the project workspace')
    archive = root / 'EMG dataset.7z'
    # Verify the downloaded archive before trusting or extracting its members.
    archive_sha = digest(archive)
    if archive_sha != 'd30a1096ed2522b32bfafcfcb2d9cdf5ccf28e1e21a41dda77824317a8c1b1ff':
        raise ValueError('Official verified archive changed')
    users = sorted(int(x) for x in np.random.default_rng(20260915).choice(np.arange(1, 11), 3, replace=False))
    subset = root / 'native_subset'
    subset.mkdir(exist_ok=True)
    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    with py7zr.SevenZipFile(archive) as z:
        names = z.getnames()
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or '..' in pure.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe archive member')
        recordings = [name for name in names if name.endswith('.txt')]
        identities = {parse_recording_name(name) for name in recordings}
        expected = {(u, m, p) for u in range(1, 11) for m in MOVEMENTS for p in ('P1', 'P2', 'P3') if not (u == 7 and m == 'EX')}
        if identities != expected or len(recordings) != 267:
            raise ValueError('Unexpected native recording inventory')
        selected = [next(n for n in recordings if parse_recording_name(n) == (u, 'PS', p)) for u in users for p in ('P1', 'P3')]
        for name in selected:
            target = (subset / name).resolve()
            if not target.is_relative_to(subset.resolve()):
                raise ValueError('Extraction escaped subset')
            target.parent.mkdir(parents=True, exist_ok=True)
        z.extract(path=subset, targets=selected, recursive=False)
    samples = []
    for name in selected:
        path = subset / name
        subject, movement, position = parse_recording_name(name)
        lines = path.read_text().splitlines()
        bad_rows = [{'line': i + 1, 'observed_numeric_fields': len(line.split())}
                    for i, line in enumerate(lines) if len(line.split()) != 8]
        try:
            recording = load_electrode_replacement_recording(path)
            values = recording.emg
            accepted = True
        except ValueError:
            # Keep full recording rejected; only inspect an intact prefix, never
            # compress time by deleting rows or impute missing channels.
            prefix = lines[:min(5000, bad_rows[0]['line'] - 1)] if bad_rows else []
            if not prefix:
                raise
            values = np.array([[float(v) for v in line.split()] for line in prefix])
            accepted = False
        svg = plots / (path.stem + '.svg')
        label = f'ID{subject}; PS recording intent; {position}; 1000 Hz; intact prefix; interval labels unavailable'
        plot(svg, values[:5000], 1000., label)
        tree = ET.parse(svg)
        if len(tree.findall('.//{http://www.w3.org/2000/svg}polyline')) != 3:
            raise ValueError('Missing sanity plot panel')
        samples.append({**qc(values, name, 1000.), 'subject': subject,
            'full_recording_rows': len(lines), 'full_recording_accepted': accepted,
            'missing_field_rows': bad_rows, 'qc_scope': 'full_recording' if accepted else 'intact_prefix_only',
            'recording_intent': movement, 'position': position,
            'raw_sha256': digest(path), 'plot_sha256': digest(svg), 'plot': str(svg),
            'interval_labels': None, 'repetition_boundaries': None})
    result = {'status': 'sampled_native_sanity_missing_fields_not_trial_annotated',
        'source': 'https://zenodo.org/records/4039550', 'archive_sha256': archive_sha,
        'native_recordings': len(recordings), 'missing_recordings': ['ID7_EX_P1', 'ID7_EX_P2', 'ID7_EX_P3'],
        'seed': 20260915, 'subject_population': list(range(1, 11)), 'selected_subjects': users,
        'conditions': ['P1', 'P3'], 'samples': samples,
        'limitations': ['No native interval labels or repetition boundaries recovered',
            'No few-shot repetition counts inferred', 'Observed-extrema fraction is not a calibrated ADC saturation rate',
            'Eight raw EMG channels; no IMU or common four-class mapping established']}
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'recordings': len(recordings), 'samples_checked': len(samples)}))


if __name__ == '__main__':
    main()
