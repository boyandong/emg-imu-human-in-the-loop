"""Replay fixed, oracle-labelled UniBo bouts through the chunked cue assembler.

This checks transport and sequence-construction parity, not onset detection or
streaming classification accuracy. No model is fitted and no target labels are
used to tune a detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from emgimu.datasets.benchmark import load_benchmark_trial
from .temporal import CuedSequenceAssembler
from .unibo_sequence_temporal import envelope_path


CHUNK_SIZES = (17, 31, 40, 73, 11)


def replay_cued_bout(signal: np.ndarray, trial_id: str, start: int,
                     *, sample_rate_hz: float = 200.) -> np.ndarray:
    """Return the assembled raw samples after deterministic irregular chunks."""
    signal = np.asarray(signal)
    if signal.ndim != 2 or len(signal) < sample_rate_hz:
        raise ValueError('a complete native bout of at least one second is required')
    assembler = CuedSequenceAssembler(sample_rate_hz, signal.shape[1])
    assembler.begin(trial_id, start)
    offset = 0
    chunk = 0
    while offset < len(signal):
        stop = min(len(signal), offset + CHUNK_SIZES[chunk % len(CHUNK_SIZES)])
        assembler.append(signal[offset:stop], start + offset)
        offset = stop
        chunk += 1
    return assembler.finish(trial_id, start + len(signal)).emg[0]


def run(dataset_root: Path, metadata_path: Path, output_path: Path) -> dict:
    metadata_bytes = metadata_path.read_bytes()
    rows = json.loads(metadata_bytes)
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty frozen bout metadata required')
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row['id'] != f"{row['trial']}:{row['start']}:{row['end']}":
            raise ValueError('bout identity and boundaries disagree')
        by_trial[row['trial']].append(row)
    seen = set()
    total_samples = 0
    max_path_error = 0.
    for index, (trial_id, bouts) in enumerate(by_trial.items(), 1):
        first = bouts[0]
        path = (dataset_root / 'trials' / first['user'] /
                f"d{first['day']:02d}" / f"p{first['posture']:02d}" /
                f'{trial_id}.npz')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != first['raw_sha256']:
            raise AssertionError(f'raw trial hash changed: {trial_id}')
        trial = load_benchmark_trial(path, expected_channels=4,
                                     expected_rate_hz=200.)
        if trial.trial_id != trial_id or not trial.benchmark_eligible:
            raise AssertionError(f'trial identity or eligibility changed: {trial_id}')
        for bout in bouts:
            start, end = int(bout['start']), int(bout['end'])
            if not 0 <= start < end <= len(trial.emg):
                raise AssertionError(f'invalid native boundaries: {bout["id"]}')
            if not np.all(trial.hand_label[start:end] == bout['label']):
                raise AssertionError(f'cue labels changed: {bout["id"]}')
            signal = trial.emg[start:end]
            assembled = replay_cued_bout(signal, trial_id, start)
            if not np.array_equal(assembled, signal):
                raise AssertionError(f'chunked native samples changed: {bout["id"]}')
            if not np.isclose(len(assembled) / 200., bout['duration_seconds'], atol=1e-12):
                raise AssertionError(f'native duration changed: {bout["id"]}')
            expected = envelope_path(signal)
            actual = envelope_path(assembled)
            error = float(np.max(np.abs(actual - expected)))
            max_path_error = max(max_path_error, error)
            total_samples += len(signal)
            seen.add(bout['id'])
        if index % 250 == 0:
            print(f'[{index}/{len(by_trial)}] trials, {len(seen)} bouts', flush=True)
    result = {
        'scope': 'fixed-cue chunked input parity; no onset detection or accuracy claim',
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'trial_count': len(by_trial), 'bout_count': len(seen),
        'native_sample_count': total_samples,
        'chunk_pattern_samples': list(CHUNK_SIZES),
        'max_envelope_path_abs_error': max_path_error,
        'all_native_samples_exact': len(seen) == len(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.metadata, args.output)), flush=True)


if __name__ == '__main__':
    main()
