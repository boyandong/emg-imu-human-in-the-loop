"""Source-only raw F9 mask with paired native-recording corruption replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import h5py
import numpy as np
from benchmarks.song_real8_study import load_session,window_starts,parse_label,_filter_emg
from benchmarks.song_real8.song_personal_session_workflow_v1 import windows,CLASSES,CHANNELS
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.source_quality_gate_v1 import SourceQualityGateV1
from emgimu.feature_bank.fault_gate_evaluation_v1 import evaluate_fault_gate

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT.parent
PROTOCOL=HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json';RESULT=HERE/'SONG_RAW_QUALITY_V1_RESULTS.json';OUT=HERE/'song_raw_quality_v1'
SCENARIOS=('unmodified','dropout_ch3','flat_ch3','transport_rail_ch3','flat26_ch3','line50_source_rms','low5_source_rms','gain2_ch3')
KNOWN_FAULTS=SCENARIOS[1:5]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):return value.decode('utf8') if isinstance(value,bytes) else str(value)


def native(session,protocol):
    folder=Path(protocol['source_folder'])/f'2026-09-18_{session}'
    item=load_session(folder,session,filter_mode='causal');path=folder/'session.h5'
    if sha(path)!=protocol['hdf5_sha256'][session]:raise ValueError('Native recording changed')
    with h5py.File(path) as f:raw=f['streams/emg/raw'][:];trials=f['trials'][:];imu_indices=f['streams/imu/emg_sample_index'][:]
    starts=[];ids=[]
    for row in trials:
        if (text(row['trial_kind'])!='formal' or not bool(row['valid']) or text(row['completion_status'])!='completed'):continue
        left,right=int(row['stable_start_sample']),int(row['stable_end_sample'])
        if not 0<=left<right<=len(raw):continue
        for start in window_starts(left,right):
            if np.searchsorted(imu_indices,int(start)+49,side='right')<22:continue
            starts.append(int(start));ids.append(f"{session}:{int(row['trial_id'])}")
    if ids!=item['trial'].tolist():raise ValueError('Raw reconstruction differs from native filtered window axis')
    item['batch']=FeatureBatch(item['batch'].emg,250.)
    return item,raw,np.asarray(starts),FeatureBatch(np.stack([raw[s:s+50] for s in starts]),250.)


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Quality protocol already frozen')
    old=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    files=['emgimu_classifier/src/emgimu/feature_bank/source_quality_gate_v1.py',
        'emgimu_classifier/src/emgimu/feature_bank/personal_session_stream_v2.py',
        'emgimu_classifier/tests/test_source_quality_gate_v1.py','emgimu_classifier/tests/test_personal_session_stream_v2.py',
        'emgimu_classifier/benchmarks/song_real8/song_raw_quality_v1.py',
        'collection/emg_meta/emg_meta/emgforce/protocol.py',
        'collection/emg_meta/emg_meta/emgforce/inference/personal_session_worker_v2.py',
        'collection/emg_meta/emg_meta/emgforce/ui/realtime_inference_page_v2.py',
        'collection/emg_meta/emg_meta/emgforce/ui/main_window.py',
        'collection/emg_meta/emg_meta/tests/test_quality_page_v2.py',
        'emgimu_classifier/src/emgimu/feature_bank/document_quality_v3.py',
        'emgimu_classifier/src/emgimu/feature_bank/calibration.py',
        'emgimu_classifier/src/emgimu/feature_bank/fault_gate_evaluation_v1.py']
    artifacts=[HERE/'song_personal_session_v1/source_bank.pkl',HERE/'song_personal_session_v1/personal_S03.zip',
               HERE/'song_personal_session_v1/session_S04_5shot.zip',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json']
    p=dict(schema='song_raw_quality_v1',source_folder=old['source_folder'],hdf5_sha256=old['hdf5_sha256'],
        scenarios=SCENARIOS,known_injected_fault_scenarios=KNOWN_FAULTS,modes=['off','structural','soft'],
        source_sessions=['S01','S02'],target_session='S04',source_quantile=.995,adc_range=[-8388608,8388607],
        adc_range_provenance='Acquisition signed24_be big-endian signed24 transport extrema; not a measured analogue-front-end clipping bound',
        rules='Fixed half-count zero/flat tolerances. Source raw pre-software-highpass F9v3 references and0.995 quantile soft thresholds. Structural invalid>=0.5 zero/flat ratio or>=0.1 near transport rail. Structural mode rejects any invalid measured channel. Soft mode uses channel min for F0/F2ac and mean for remaining providers; Unknown only at zero effective mass. No target tuning.',
        source_sha256={f:sha(REPO/f) for f in files},artifact_sha256={f.relative_to(REPO).as_posix():sha(f) for f in artifacts},
        primary='Structural mode catches all four explicit injected fault types at native-trial granularity and rejects no otherwise-correct unmodified trial; this is a synthetic check, not physical fault validation. Preserve soft-routing harms/failures and all modes. Off/default behavior remains unchanged.',
        scope='Previously inspected same-person/day S04 fixed124 evaluation trials; original source-frozen six providers and20+20 personal/current calibration remain fixed. Inject into the full raw recording before causal filtering. Unmodified native data have UNKNOWN hardware-fault status, not normal labels; 50Hz/5Hz/gain perturbations are not assumed faults. No real fault false-positive rate, mechanical-force, hardware accuracy, live latency or default promotion proof.',default_promoted=False)
    with PROTOCOL.open('x',encoding='utf8',newline='\n') as f:json.dump(p,f,indent=2);f.write('\n')
    print('Frozen raw-quality/native corruption and GUI protocol; no native data loaded',flush=True)


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Existing quality result will not be overwritten')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in p['source_sha256'].items():
        if sha(REPO/name)!=digest:raise ValueError('Frozen quality source changed: '+name)
    for name,digest in p['artifact_sha256'].items():
        if sha(REPO/name)!=digest:raise ValueError('Frozen source-model artifact changed')
    source=[native(s,p) for s in p['source_sessions']]
    raw_source=FeatureBatch(np.concatenate([v[3].emg for v in source]),250.)
    source_ids=np.concatenate([v[0]['trial'] for v in source])
    gate=SourceQualityGateV1(channel_ids=CHANNELS,adc_range=tuple(p['adc_range']),
        adc_range_provenance=p['adc_range_provenance'],source_quantile=p['source_quantile']).fit(raw_source,source_ids)
    OUT.mkdir();gate_path=OUT/'source_gate.pkl';gate_path.write_bytes(pickle.dumps(gate));gate_before=pickle.dumps(gate)
    original=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    w=load_workflow(HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    personal=w.load_profile(HERE/'song_personal_session_v1/personal_S03.zip',user_id='Song')
    session=w.load_profile(HERE/'song_personal_session_v1/session_S04_5shot.zip',user_id='Song',session_id='S04',personal=personal)
    frozen=pickle.dumps((w.bank,personal,session))
    item,raw,starts,_=native('S04',p);mask=np.isin(item['trial'],original['evaluation_ids']);starts=starts[mask]
    ids=item['trial'][mask];offsets=np.zeros(len(ids),int)
    for t in np.unique(ids):offsets[ids==t]=np.arange(np.sum(ids==t))
    trial_ids=tuple(np.unique(ids));y=np.asarray(original['evaluation_labels'])
    if list(trial_ids)!=original['evaluation_ids']:raise ValueError('Fixed target axis differs')
    amplitude=np.median(np.sqrt(np.mean(raw_source.emg.astype(float)**2,axis=1)),axis=0)
    rows=[];cells=[];arrays=dict(trial_ids=np.asarray(trial_ids),labels=y,class_names=np.asarray(CLASSES));baseline=None
    for scenario in SCENARIOS:
        altered=raw.astype(float).copy()
        if scenario=='dropout_ch3':altered[:,2]=0
        elif scenario=='flat_ch3':altered[:,2]=float(np.median(raw_source.emg[:,:,2]))
        elif scenario=='transport_rail_ch3':altered[:,2]=p['adc_range'][1]
        elif scenario=='flat26_ch3':
            for start in starts:altered[start:start+26,2]=altered[start,2]
        elif scenario in ('line50_source_rms','low5_source_rms'):
            hz=50 if scenario.startswith('line') else 5
            altered+=amplitude[None,:]*np.sin(2*np.pi*hz*np.arange(len(raw))/250.)[:,None]
        elif scenario=='gain2_ch3':altered[:,2]*=2
        filtered=_filter_emg(altered,'causal')
        eb=FeatureBatch(np.stack([filtered[s:s+50] for s in starts]),250.)
        rb=FeatureBatch(np.stack([altered[s:s+50] for s in starts]),250.)
        readout=w.bank.predict_providers(eb,ids,window_offsets=offsets,user_id='Song')
        providers=readout['probabilities'];weights=session.fusion_state.weights
        if scenario=='unmodified':
            baseline=sum(float(weights[i])*providers[g] for i,g in enumerate(w.bank.providers_));baseline=baseline.argmax(1)==np.array([CLASSES.index(c) for c in y])
            reference=w.predict(eb,ids,window_offsets=offsets,user_id='Song',session_id='S04',personal=personal,session=session,
                observed_channel_ids=CHANNELS,preprocessing_id=w.preprocessing_id)
            arrays['unmodified_reference_probabilities']=reference['probabilities']
        for mode in p['modes']:
            decision=gate.decide(providers,weights,rb,ids,observed_channel_ids=CHANNELS,class_names=CLASSES,mode=mode,
                provider_trial_ids={g:readout['trial_ids'] for g in providers})
            rejected=decision['rejected'];predicted=np.asarray(decision['labels']);correct=predicted==y
            annotations=np.repeat('fault' if scenario in KNOWN_FAULTS else 'unknown',len(y))
            evaluated=evaluate_fault_gate(rejected,annotations,trial_ids,source_trial_ids=gate.source_trials)
            accepted=~rejected
            cell=dict(scenario=scenario,mode=mode,trials=len(y),rejected=int(rejected.sum()),coverage=float(accepted.mean()),
                accuracy_unknown_wrong=float(correct.mean()),accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
                baseline_correct_unmodified_rejected=int(np.sum(baseline&rejected)) if scenario=='unmodified' else None,
                fault_annotation='synthetic_known_fault' if scenario in KNOWN_FAULTS else 'unknown_hardware_fault_status',
                gate_evaluation=evaluated)
            cells.append(cell)
            for t,label,pred,reject,q,bad in zip(trial_ids,y,predicted,rejected,decision['probabilities'],decision['bad_channel_count']):
                rows.append(dict(scenario=scenario,mode=mode,trial_id=t,label=label,predicted_label=pred,
                    rejected=bool(reject),bad_channel_count=int(bad),**{'p_'+c:float(q[i]) for i,c in enumerate(CLASSES)}))
            arrays[scenario+'_'+mode+'_probabilities']=decision['probabilities']
            arrays[scenario+'_'+mode+'_channel_quality']=decision['channel_quality']
        print(f'{scenario}: all3 gate modes complete; classifier/profiles unchanged',flush=True)
    lookup={(c['scenario'],c['mode']):c for c in cells}
    guards=dict(all_four_injected_fault_types_rejected=all(lookup[(s,'structural')]['rejected']==124 for s in KNOWN_FAULTS),
        no_correct_unmodified_predictions_rejected=lookup[('unmodified','structural')]['baseline_correct_unmodified_rejected']==0)
    with (OUT/'predictions.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'readouts.npz',**arrays)
    if pickle.dumps(gate)!=gate_before or pickle.dumps((w.bank,personal,session))!=frozen:raise ValueError('Gate/source/profile mutation')
    r=dict(schema=p['schema'],protocol_sha256=sha(PROTOCOL),gate_path=gate_path.relative_to(ROOT).as_posix(),gate_sha256=sha(gate_path),
        gate_policy_id=gate.policy_id,source_trials=list(gate.source_trials),source_windows=raw_source.windows,
        source_amplitude=amplitude.tolist(),evaluation_ids=list(trial_ids),evaluation_labels=y.tolist(),cells=cells,
        primary_guards=guards,primary_pass=all(guards.values()),source_state_immutable=True,default_promoted=False,
        physical_validation_proven=False,completion_proven=False,scope=p['scope'],
        artifact_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in OUT.iterdir()})
    RESULT.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
