"""Native target calibration cost for immutable F0/F7 and late-fusion experiments."""
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.epn612 import load_epn612_windows, GESTURE_TO_LABEL
from benchmarks.new_bank_v3.emg_window_bank_v1 import sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'EMG_CALIBRATION_BURDEN_V1.json'
TABLE=HERE/'EMG_CALIBRATION_BURDEN_V1.csv'
RUNS=('EMG_F0_F7_BANK_V1','EMG_CALIBRATED_FUSION_V1')


def run():
    sources={}; payloads={}; archive=None; archive_digest=None
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding='utf8'))
    for name in RUNS:
        p=read(HERE/(name+'_PROTOCOL.json'));r=read(HERE/(name+'_RESULTS.json'))
        if r['protocol_sha256']!=sha(HERE/(name+'_PROTOCOL.json')): raise ValueError('Protocol changed')
        if archive is None: archive=Path(p['archive']);archive_digest=p['archive_sha256']
        if Path(p['archive'])!=archive or p['archive_sha256']!=archive_digest: raise ValueError('Archive identity differs')
        payloads[name]=(p,r)
    if sha(archive)!=archive_digest: raise ValueError('Native archive changed')
    users=sorted({b['user'] for _,r in payloads.values() for b in r['blocks']})
    print('1/2 Read native window counts and complete stored EMG lengths; no model fitting',flush=True)
    data=load_epn612_windows(archive,users=users)
    ids,counts=np.unique(data.trials,return_counts=True);represented=dict(zip(ids.tolist(),counts.tolist()))
    window_samples=data.batch.emg.shape[1];sample_rate=data.batch.sample_rate_hz
    duration=window_samples/sample_rate
    reserved={t for _,r in payloads.values() for b in r['blocks'] for values in b['reserved_ids'].values() for t in values}
    ledger={}
    with zipfile.ZipFile(archive) as handle:
        for user in users:
            member=f'EMG-EPN612 Dataset/trainingJSON/user{user}/user{user}.json'
            raw=json.loads(handle.read(member));rate=raw['generalInfo']['samplingFrequencyInHertz']
            if rate!=sample_rate: raise ValueError('Raw recording sample rate differs')
            for name,sample in raw['trainingSamples'].items():
                trial=f'trainingJSON:user{user}:{name}'
                if trial not in reserved: continue
                lengths=[len(sample['emg'][f'ch{c}']) for c in range(1,9)]
                if len(set(lengths))!=1 or not lengths[0]: raise ValueError('Raw EMG channel lengths differ')
                ledger[trial]={'subject':user,'native_gesture':sample['gestureName'],
                    'class_label':GESTURE_TO_LABEL[sample['gestureName']], 'samples_per_channel':lengths[0],
                    'channels':8,'sample_rate_hz':rate,'full_recording_seconds':lengths[0]/rate,
                    'represented_windows':represented[trial],'represented_signal_seconds':represented[trial]*duration}
    if set(ledger)!=reserved: raise ValueError('Missing native reserved trial')
    rows=[];selections=[]
    for name,(p,r) in payloads.items():
        for b in r['blocks']:
            reserved_ids=[t for values in b['reserved_ids'].values() for t in values]
            for shots in p['budgets']:
                entry=next((c for c in b['calibrations'] if c['shots']==shots),None)
                cal_ids=entry['ids'] if entry else []
                if len(cal_ids)!=6*shots or set(cal_ids)&set(b['evaluation_ids']): raise ValueError('Calibration cost provenance mismatch')
                if entry and [ledger[t]['class_label'] for t in cal_ids]!=entry['labels']: raise ValueError('Native calibration label differs')
                selections.append({'run_id':name,'subject':b['user'],'shots_per_class':shots,
                    'calibration_ids':cal_ids,'reserved_ids':reserved_ids,'evaluation_ids':b['evaluation_ids']})
                for arm in r['scores'][str(shots)]:
                    uses_cal=(arm in ('F7','F0_F7')) if name=='EMG_F0_F7_BANK_V1' else arm.startswith('reliability')
                    used=cal_ids if uses_cal else []
                    windows=sum(ledger[t]['represented_windows'] for t in used)
                    rows.append({'run_id':name,'feature_bank':arm,'dataset':'EPN612','subject':b['user'],
                        'shots_per_class':shots,'classes':6,'used_calibration_trials':len(used),
                        'used_calibration_windows':windows,'window_samples':window_samples,'sample_rate_hz':sample_rate,
                        'used_signal_seconds':windows*duration,
                        'used_trials_full_recording_seconds':sum(ledger[t]['full_recording_seconds'] for t in used),
                        'reserved_calibration_trials':len(reserved_ids),
                        'reserved_signal_seconds':sum(ledger[t]['represented_signal_seconds'] for t in reserved_ids),
                        'reserved_full_recording_seconds':sum(ledger[t]['full_recording_seconds'] for t in reserved_ids),
                        'requires_all_native_gestures':bool(used),'requires_target_force':'N/A',
                        'requires_target_posture':'N/A','requires_each_session':'N/A',
                        'hardware_setup_seconds':'N/A','guided_prompt_rest_seconds':'N/A','device_wall_time_seconds':'N/A',
                        'scope':'Target-only calibration burden. Used extracted signal differs from full stored recording duration; neither measures physical setup, guided rest or elapsed session time. Reserved trials are excluded from evaluation even when unused.'})
    with TABLE.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for path in (Path(__file__),ROOT/'src/emgimu/datasets/epn612.py',ROOT/'tests/test_emg_calibration_burden_v1.py'):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    out={'schema':'emg_calibration_burden_v1','archive_sha256':archive_digest,'source_sha256':sources,
         'table_sha256':sha(TABLE),'native_reserved_trial_ledger':ledger,'calibration_selections':selections,
         'records':rows,'few_second_calibration_proven':False,'physical_wall_time_proven':False,
         'scope':'Descriptive native cost accounting of completed experiments; no model fitting or protocol changes. Source-training cost is excluded. No cross-session, controlled-force/posture or own-device timing evidence.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print(f'2/2 Saved {len(rows)} method-specific cost rows and {len(ledger)} native reserved trial durations',flush=True)


if __name__=='__main__':run()
