"""Run the previously optional original-MAT smoke check without training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'feature_bank/OFFICIAL_UNIBO_ADAPTER_ACCEPTANCE_V1.json'
PROTOCOL = ROOT/'benchmarks/new_bank_v3/AUTONOMOUS_CONTINUOUS_UNIBO_V1_PROTOCOL.json'
MEMBER = 'unibo-inail-semg-dataset-main/data/user_1_day_1_posture_1.mat'
TEST = 'tests/test_unibo_adapter.py::UniBoOfficialSampleSmokeTest::test_official_sample_read_only_conversion'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def run():
    if OUTPUT.exists(): raise FileExistsError('Completed original-data acceptance already exists')
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    archive = Path(p['archive'])
    print('1/2 Verify original archive and extract one unchanged source member', flush=True)
    if sha(archive) != p['archive_sha256']: raise ValueError('Original archive changed')
    if MEMBER not in p['members']: raise ValueError('Unregistered original source member')
    with zipfile.ZipFile(archive) as source: raw = source.read(MEMBER)
    tmp = ROOT/'tmp'; tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='official_unibo_acceptance_',dir=tmp) as directory:
        sample = Path(directory)/Path(MEMBER).name; sample.write_bytes(raw)
        env = os.environ.copy(); env['PYTHONPATH'] = 'src;.'
        env['UNIBO_INAIL_SAMPLE'] = str(sample.resolve())
        env['OMP_NUM_THREADS'] = '1'; env['OPENBLAS_NUM_THREADS'] = '1'
        print('2/2 Execute official source conversion test; no fitting', flush=True)
        command = [sys.executable,'-m','pytest',TEST,'-q','-rs']
        completed = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,encoding='utf8')
        if completed.returncode or '1 passed' not in completed.stdout or 'skipped' in completed.stdout:
            raise RuntimeError(completed.stdout + completed.stderr)
        if sample.read_bytes() != raw: raise ValueError('Read-only official sample changed')
    paths = ['src/emgimu/datasets/adapters/unibo_inail.py','src/emgimu/datasets/adapters/base.py',
             'src/emgimu/datasets/benchmark.py','src/emgimu/signal.py','tests/test_unibo_adapter.py',
             'benchmarks/official_unibo_adapter_acceptance_v1.py']
    result = {'schema':'official_unibo_adapter_acceptance_v1', 'protocol_sha256':sha(PROTOCOL),
              'source_archive_sha256':p['archive_sha256'],'source_member':MEMBER,
              'source_member_bytes':len(raw),'source_member_sha256':hashlib.sha256(raw).hexdigest(),
              'source_sha256':{name:sha(ROOT/name) for name in paths},
              'test_node':TEST,'test_exit_code':completed.returncode,'test_summary':completed.stdout.strip().splitlines()[-1],
              'test_stdout':completed.stdout,'test_stderr':completed.stderr,
              'source_sample_immutable':True,'training_performed':False,
              'physical_validation_proven':False,'completion_proven':False,
              'scope':'Single official original MAT adapter conversion and benchmark-schema smoke check. Complements the full-suite run whose environment did not select this optional test; not all-file data validation, sensor compatibility, recognition accuracy or device execution.'}
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(result['test_summary'],flush=True)
    return result


if __name__ == '__main__': run()
