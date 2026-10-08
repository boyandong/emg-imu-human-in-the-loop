"""Record hardware-free acceptance for user items5/6/7, with physical gaps explicit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    test = 'tests/test_hardware_preparation_v1.py'
    result = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',test],cwd=ROOT,
                            capture_output=True,text=True)
    if result.returncode:
        print(result.stdout+result.stderr)
        raise SystemExit(result.returncode)
    modules = {5:'src/emgimu/feature_bank/body_frame_v2.py',
               6:'src/emgimu/feature_bank/electrode_layout_v1.py',
               7:'src/emgimu/feature_bank/fault_gate_evaluation_v1.py'}
    evidence = [test,*modules.values(),'src/emgimu/feature_bank/body_frame.py',
                'src/emgimu/feature_bank/core.py','benchmarks/hardware_preparation_acceptance.py']
    payload = {'schema':'hardware_preparation_acceptance_v1',
        'source_sha256':{p:sha(ROOT/p) for p in evidence},'suite_exit_code':result.returncode,
        'suite_summary':result.stdout.strip().splitlines()[-1],
        'items':[
            {'item':5,'module':modules[5],'software_scope':'Explicit g/m/s² and deg/s/rad/s conversion; calibrated relative frame; strict calibration IDs and matched sensor durations.',
             'verified_by':'Independent15-coordinate constant-step SI oracle and unit equivalence; source/evaluation guards.',
             'remaining':'Measured neutral and guided forward calibration, known raw IMU units/axes, native synchronized recordings and physical orientation validation.'},
            {'item':6,'module':modules[6],'software_scope':'Explicit eight measured-channel physical ring map with evidence reference; source/evaluation layout fingerprint; loader-column reorder.',
             'verified_by':'Known channel identity permutation and source-layout mismatch rejection.',
             'remaining':'Actual electrode/pair wiring and circumferential order must be confirmed physically; software cannot authenticate an operator declaration.'},
            {'item':7,'module':modules[7],'software_scope':'Descriptive fault/normal/unknown annotation evaluation; trial-balanced metrics, baseline harm and source-overlap guard.',
             'verified_by':'Independent known confusion rates, duplicate-recording mass and all-unknown unavailable-metric cases.',
             'remaining':'Real fault annotations and independent normal recordings, source-selected gate configuration and downstream application/device validation.'}],
        'physical_validation_proven':False,'default_promoted':False,'completion_proven':False,
        'scope':'Software preparation only; no training, deployment default or hardware accuracy inferred from fixtures.'}
    (ROOT/'feature_bank/HARDWARE_PREPARATION_ACCEPTANCE.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print(result.stdout.strip().splitlines()[-1],flush=True)


if __name__ == '__main__':
    run()
