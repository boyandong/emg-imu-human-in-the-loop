"""Document F4 native increment; retrospective fixed GRAB unseen-user cohort."""
import argparse
import json
from pathlib import Path
from .f3b_ces_grab_run import run_candidate,sha256
from emgimu.feature_bank.document_spectral_v3 import DocumentSpectralStateV3

HERE=Path(__file__).resolve().parent


def run(data_root):
    result=run_candidate(data_root,protocol_path=HERE/'F4_SPECTRAL_GRAB_PROTOCOL.json',
        candidate_name='F4spectral',candidate_factory=DocumentSpectralStateV3,output_prefix='F4_SPECTRAL_GRAB')
    root=HERE.parents[1]
    result['source_hashes']={p:sha256(root/p) for p in (
        'src/emgimu/feature_bank/document_spectral_v3.py',
        'benchmarks/new_bank_v3/f4_spectral_grab_run.py',
        'benchmarks/new_bank_v3/f3b_ces_grab_run.py')}
    (HERE/'F4_SPECTRAL_GRAB_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,
        default=Path('D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1'))
    run(p.parse_args().data_root)
