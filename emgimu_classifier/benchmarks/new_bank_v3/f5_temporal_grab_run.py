"""Source-only isolated F5 document formula on frozen GRAB unseen-user trials."""
from __future__ import annotations

import argparse
from pathlib import Path

from .f3b_ces_grab_run import run_candidate
from emgimu.feature_bank.document_temporal_v3 import DocumentTemporalFormV3


HERE = Path(__file__).resolve().parent


def run(data_root: Path) -> dict:
    return run_candidate(data_root, protocol_path=HERE / 'F5_TEMPORAL_GRAB_PROTOCOL.json',
                         candidate_name='F5temporal', candidate_factory=DocumentTemporalFormV3,
                         output_prefix='F5_TEMPORAL_GRAB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1'))
    run(parser.parse_args().data_root)
