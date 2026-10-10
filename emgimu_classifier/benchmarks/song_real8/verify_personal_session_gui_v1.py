"""No-fit, full-recording parity check for the personal/session GUI service.

The native recording was previously inspected. It includes calibration data;
this is numerical/runtime verification, never an independent efficacy test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import h5py
import numpy as np
from benchmarks.song_real8_study import _filter_emg
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v1 import PersonalSessionStreamV1

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'SONG_PERSONAL_GUI_V1_PROTOCOL.json'
OUT=ROOT/'feature_bank/SONG_PERSONAL_GUI_V1_ACCEPTANCE.json'
SOURCE=HERE/'song_personal_session_v1/source_bank.pkl'
ACCEPTANCE=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'
FILES=[
    'emgimu_classifier/src/emgimu/feature_bank/personal_session_stream_v1.py',
    'emgimu_classifier/tests/test_personal_session_stream_v1.py',
    'emgimu_classifier/benchmarks/song_real8/verify_personal_session_gui_v1.py',
    'collection/emg_meta/emg_meta/emgforce/inference/personal_session_worker.py',
    'collection/emg_meta/emg_meta/emgforce/ui/personal_session_panel.py',
    'collection/emg_meta/emg_meta/emgforce/ui/realtime_inference_page.py',
    'collection/emg_meta/emg_meta/tests/test_personal_session_panel.py']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    old=json.loads((HERE/'SONG_F0_STREAM_V1_PROTOCOL.json').read_text(encoding='utf8'))
    protocol=dict(schema='song_personal_gui_v1_protocol', source_sha256={p:sha(REPO/p) for p in FILES},
        native_path=str(Path(old['source_folder'])/'2026-09-18_S04/session.h5'),
        native_sha256=old['hdf5_sha256']['S04'], arms=['population','session_5shot'],
        chunk_samples=4096, sample_rate_hz=250, window_samples=50, hop_samples=10, consecutive_frames=2,
        source_bank_sha256=sha(SOURCE), offline_acceptance_sha256=sha(ACCEPTANCE),
        profile_sha256={p:sha(HERE/'song_personal_session_v1'/p) for p in ('personal_S03.zip','session_S04_5shot.zip')},
        scope='Previously inspected complete S04 recording, including calibration intervals. Fixed numerical parity only; no fitting, target tuning, accuracy or hardware claims.')
    with PROTOCOL.open('x',encoding='utf8',newline='\n') as f:json.dump(protocol,f,indent=2);f.write('\n')


def run():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(REPO/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if sha(SOURCE)!=p['source_bank_sha256'] or sha(ACCEPTANCE)!=p['offline_acceptance_sha256']:
        raise ValueError('Frozen bank/acceptance changed')
    for name,digest in p['profile_sha256'].items():
        if sha(HERE/'song_personal_session_v1'/name)!=digest:raise ValueError('Frozen profile changed')
    native=Path(p['native_path'])
    if sha(native)!=p['native_sha256']:raise ValueError('Native recording changed')
    with h5py.File(native) as f:raw=f['streams/emg/raw'][:];indices=f['streams/emg/sample_index'][:]
    if not np.array_equal(indices,np.arange(len(raw))):raise ValueError('Native sample continuity differs')
    filtered=_filter_emg(raw,'causal');records=[]
    for arm in p['arms']:
        workflow=load_workflow(SOURCE,ACCEPTANCE)
        service=PersonalSessionStreamV1(workflow,user_id='Song',session_id='S04',channel_ids=workflow.channels)
        if arm=='session_5shot':
            service.command(dict(op='profiles',personal=str(HERE/'song_personal_session_v1/personal_S03.zip'),
                session=str(HERE/'song_personal_session_v1/session_S04_5shot.zip')))
        source_before=pickle.dumps(workflow.bank);personal_before=pickle.dumps(service.personal)
        session_before=pickle.dumps(service.session)
        if service.session is None:weights=workflow.bank.policy_.population
        else:weights=service.session.fusion_state.weights
        service.command(dict(op='recognize'));count=0;error=0.;candidate=active=None;consecutive=0
        for start in range(0,len(raw),p['chunk_samples']):
            result=service.ingest(raw[start:start+p['chunk_samples']],indices[start:start+p['chunk_samples']])
            if 'probabilities' not in result:continue
            ends=np.asarray(result['output_sample_indices']);ids=np.array([f'verification:{i:09}' for i in ends])
            batch=FeatureBatch(np.stack([filtered[end-49:end+1] for end in ends]),250.)
            readout=workflow.bank.predict_providers(batch,ids,window_offsets=np.zeros(len(ids),int),user_id=service.user)
            expected=sum(float(weights[i])*readout['probabilities'][name] for i,name in enumerate(workflow.bank.providers_))
            expected/=expected.sum(1,keepdims=True)
            difference=float(np.max(abs(expected-result['probabilities'])));error=max(error,difference)
            if difference>1e-12:raise ValueError('Window probabilities differ from independently weighted frozen providers')
            confirmed=[]
            for probability in expected:
                label=workflow.bank.class_names_[int(probability.argmax())]
                consecutive=consecutive+1 if label==candidate else 1;candidate=label
                if consecutive>=2:active=label
                confirmed.append(active)
            if confirmed!=result['confirmed_labels']:raise ValueError('Chronological two-confirmation labels differ')
            count+=len(ends)
        if (pickle.dumps(workflow.bank)!=source_before or pickle.dumps(service.personal)!=personal_before
                or pickle.dumps(service.session)!=session_before):raise ValueError('Source or calibration profile mutated')
        records.append(dict(arm=arm,windows=count,maximum_probability_error=error,confirmation_exact=True,
            source_and_profiles_immutable=True,personal_trials=service.info()['personal_trials'],session_trials=service.info()['session_trials']))
        print(f'{arm}: {count} live windows verified; max error {error:.3g}',flush=True)
    output=dict(schema='song_personal_gui_v1_acceptance', protocol_sha256=sha(PROTOCOL),
        source_sha256=p['source_sha256'],records=records,
        gui_flow='Qt subprocess tests: guided personal enrollment, new-session calibration, ZIP+calibration-input persistence, replay, live input, disconnect, identity rejection and backend switching.',
        default_promoted=False,physical_validation_proven=False,completion_proven=False,
        scope='Opt-in six-provider GUI lifecycle. Full native-recording window parity, not native trial efficacy or hardware validation. F7/F8/F9 remain context outputs; normalization does not replace classifier inputs; quality rejection and full F0-F9 integration remain open.')
    OUT.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
