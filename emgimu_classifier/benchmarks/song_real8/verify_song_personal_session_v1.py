"""Verify native workflow delivery without retraining or changing source models."""
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from benchmarks.song_real8.song_personal_session_workflow_v1 import ROOT,HERE,RESULT,PROTOCOL,OUT,sha,CLASSES,CHANNELS,PREPROCESSING,windows,select,score
from benchmarks.song_real8_study import load_session
from emgimu.feature_bank.core import FeatureBatch
from emgimu.feature_bank.personal_session_workflow_v1 import PersonalSessionWorkflowV1

ACCEPTANCE=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'


def run():
    r=json.loads(RESULT.read_text(encoding='utf8'));p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    if sha(PROTOCOL)!=r['protocol_sha256']:raise ValueError('Protocol changed')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed')
    for path_key,hash_key in [('source_bank_path','source_bank_sha256'),('prediction_path','prediction_sha256'),
                               ('context_path','context_sha256'),('personal_profile_path','personal_profile_sha256')]:
        if sha(ROOT/r[path_key])!=r[hash_key]:raise ValueError('Delivery bytes changed')
    bank=pickle.loads((ROOT/r['source_bank_path']).read_bytes());before=pickle.dumps(bank)
    config=dict(channel_ids=CHANNELS,preprocessing_id=PREPROCESSING,rest_label='neutral',
        quality_options={'line_frequency_hz':50,'pre_highpass_available':False})
    workflow=PersonalSessionWorkflowV1(bank,**config)
    personal=workflow.load_profile(ROOT/r['personal_profile_path'],user_id='Song');long_before=pickle.dumps(personal)
    item=load_session(Path(p['source_folder'])/'2026-09-18_S04','S04',filter_mode='causal')
    if item['audit']['sha256']!=p['hdf5_sha256']['S04']:raise ValueError('Native recording changed')
    item['batch']=FeatureBatch(item['batch'].emg,250.)
    eb,ids,offsets=windows(item,r['evaluation_ids']);kwargs=dict(window_offsets=offsets,user_id='Song',session_id='S04',
        observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
    readout=bank.predict_providers(eb,ids,window_offsets=offsets,user_id='Song');provider=readout['probabilities']
    provider_path=ROOT/'feature_bank/SONG_PERSONAL_SESSION_PROVIDER_READOUTS_V1.npz'
    np.savez_compressed(provider_path,trial_ids=np.asarray(r['evaluation_ids']),class_names=np.asarray(CLASSES),
        labels=np.asarray(r['evaluation_labels']),**provider)
    with (ROOT/r['prediction_path']).open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=44*len(r['evaluation_ids']):raise ValueError('Missing native trial cells')
    checked=0;maximum=0.;profile_paths=[]
    with np.load(ROOT/r['context_path'],allow_pickle=False) as arrays:
        for shots in p['current_shots_per_class']:
            session=None
            if shots:
                path=OUT/f'session_S04_{shots}shot.zip';profile_paths.append(path)
                expected=next(v for v in r['session_roundtrips'] if v['shots']==shots)
                if sha(path)!=expected['profile_sha256']:raise ValueError('Session bytes changed')
                session=workflow.load_profile(path,user_id='Song',session_id='S04',personal=personal)
                trial_labels={t:np.unique(item['hand'][item['trial']==t]).item() for t in np.unique(item['trial'])}
                all_ids=np.array(sorted(trial_labels));all_y=np.array([trial_labels[t] for t in all_ids])
                selected,_=select(all_ids,all_y,shots,p['selection_seed'])
                if tuple(all_ids[selected])!=session.calibration_trials:raise ValueError('Nested calibration IDs differ')
                if set(session.calibration_trials)&set(r['evaluation_ids']):raise ValueError('Native trial leakage')
            output=workflow.predict(eb,ids,personal=personal,session=session,**kwargs)
            np.testing.assert_array_equal(output['quality_observations'],arrays[f'shots{shots}_quality'])
            np.testing.assert_array_equal(workflow.normalized_view(eb,personal=personal,session=session,
                user_id='Song',session_id='S04',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING).emg,
                arrays[f'shots{shots}_normalized_emg'])
            for name,modes in output['anchor_coordinates'].items():
                for mode,x in modes.items():np.testing.assert_array_equal(x,arrays[f'shots{shots}_{name}_{mode}'])
            population=bank.policy_.population;personal_weights=personal.fusion_state.fusion_state.weights
            session_weights=personal_weights if session is None else session.fusion_state.weights
            expected_arms={'F0':provider['F0'],'population':sum(w*provider[k] for k,w in zip(bank.providers_,population)),
                'uniform':np.mean(list(provider.values()),axis=0),
                'personal':sum(w*provider[k] for k,w in zip(bank.providers_,personal_weights)),
                'session':sum(w*provider[k] for k,w in zip(bank.providers_,session_weights))}
            for omitted in bank.providers_:
                weights=np.array(session_weights);weights[bank.providers_.index(omitted)]=0.;weights/=weights.sum()
                expected_arms['session_minus_'+omitted]=sum(w*provider[k] for k,w in zip(bank.providers_,weights))
            for arm,expected in expected_arms.items():
                selected_rows=[v for v in rows if int(v['shots'])==shots and v['arm']==arm]
                if [v['trial_id'] for v in selected_rows]!=r['evaluation_ids']:raise ValueError('Evaluation order changed')
                q=np.array([[float(v['p_'+c]) for c in CLASSES] for v in selected_rows])
                error=float(np.max(abs(q-expected)));maximum=max(maximum,error)
                if error>1e-12:raise ValueError('Native fusion differs')
                metrics=score(np.array(r['evaluation_labels']),q);cell=next(c for c in r['cells'] if c['shots']==shots and c['arm']==arm)
                for key in ('macro_f1','accuracy','log_loss','brier'):
                    if abs(metrics[key]-cell[key])>1e-12:raise ValueError('Native metric differs')
                checked+=1
    if before!=pickle.dumps(bank) or long_before!=pickle.dumps(personal):raise ValueError('Verification mutated state')
    dependencies=[Path(__file__),RESULT,PROTOCOL,ROOT/'src/emgimu/feature_bank/personal_session_cli_v1.py',
        ROOT/'tests/test_personal_session_cli_v1.py',ROOT/'tests/test_song_personal_session_v1_delivery.py']
    artifacts=[ROOT/r['source_bank_path'],ROOT/r['prediction_path'],ROOT/r['context_path'],ROOT/r['personal_profile_path'],provider_path,*profile_paths]
    out={'schema':'song_personal_session_acceptance_v1','workflow_config':config,'workflow_contract_id':workflow.contract_id,
        'source_bank_id':bank.bank_id_,'source_bank_sha256':sha(ROOT/r['source_bank_path']),
        'provider_readouts_path':provider_path.relative_to(ROOT).as_posix(),
        'source_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in dependencies},
        'artifact_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in artifacts},
        'checked_native_cells':checked,'checked_trial_probabilities':len(rows),'maximum_probability_error':maximum,
        'source_state_immutable':True,'personal_state_immutable':True,'physical_validation_proven':False,
        'default_promoted':False,'completion_proven':False,
        'scope':'Source-frozen six-provider Song lifecycle; all44 cells/5456 native trial predictions and persisted context outputs reverified without fitting. Long-term20 trials plus current0/4/8/20 are separate costs. No GUI calibration integration, full F0-F9 composition, learned anchor/context classifier, quality gate, native anatomical F6 or live efficacy claim.'}
    ACCEPTANCE.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print(f'Native workflow verified: {checked} cells/{len(rows)} predictions, maximum error{maximum:.3g}; no fitting',flush=True)


if __name__=='__main__':run()
