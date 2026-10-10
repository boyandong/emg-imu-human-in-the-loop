"""Frozen same-trial F7/F8/raw-F9 composition and all provider removals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from benchmarks.song_real8.song_raw_quality_v1 import native
from benchmarks.song_real8.song_personal_session_workflow_v1 import windows, select, aggregate, score, CHANNELS, PREPROCESSING
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow
from emgimu.feature_bank.personal_session_stream_v2 import load_gate
from emgimu.feature_bank.personal_session_decision_v1 import PersonalSessionDecisionV1

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT.parent
PROTOCOL=HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json';RESULT=HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json'
OUT=HERE/'song_integrated_decision_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def arms(providers):
    definitions={'baseline':dict(use_anchor=False,use_session_routing=False),
        'F7_long':dict(anchor_mode='long_term',use_session_routing=False),
        'F7_local':dict(anchor_mode='local',use_session_routing=False),
        'F7_blended':dict(use_session_routing=False),
        'F8_only':dict(use_anchor=False),'F7_F8':{},'F7_F8_F9_structural':dict(quality_mode='structural'),
        'F7_standalone':dict(use_session_routing=False)}
    definitions.update({'full_minus_'+name:dict(available=tuple(g for g in providers if g!=name),quality_mode='structural') for name in providers})
    return definitions


def prepare():
    if PROTOCOL.exists():raise FileExistsError('Integrated protocol already frozen')
    parent=json.loads((HERE/'SONG_RAW_QUALITY_V1_PROTOCOL.json').read_text(encoding='utf8'))
    previous=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_PROTOCOL.json').read_text(encoding='utf8'))
    files=['src/emgimu/feature_bank/personal_session_decision_v1.py',
           'src/emgimu/feature_bank/personal_session_decision_cli_v1.py',
           'tests/test_personal_session_decision_v1.py','benchmarks/song_real8/song_integrated_decision_v1.py']
    dependencies=dict(previous['source_sha256'])
    dependencies.update({name:sha(ROOT/name) for name in files})
    dependencies['src/emgimu/feature_bank/source_quality_gate_v1.py']=sha(ROOT/'src/emgimu/feature_bank/source_quality_gate_v1.py')
    dependencies['src/emgimu/feature_bank/personal_session_stream_v2.py']=sha(ROOT/'src/emgimu/feature_bank/personal_session_stream_v2.py')
    artifacts=[HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
               HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json',HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json']
    p=dict(schema='song_integrated_decision_v1',source_folder=parent['source_folder'],hdf5_sha256=parent['hdf5_sha256'],
        anchor_mix=.5,current_shots=[0,1,2,5],selection_seed=previous['selection_seed'],
        temperature='Mean pairwise calibrated class-prototype distance, separately for each family and chosen long/local/budget-blended mode. Collapse gives uniform. No evaluation fitting. F2ac anchor uses exact affine-invariant distances on trial-balanced centered F2a SPD matrices; remaining heads use source-standardized Euclidean coordinates.',
        routing='Same-user calibration only: mean normalized prototype drift, mean absolute class-pair geometry shift, and named F0/F1/F4-specific shift where available. F2ac uses affine-invariant drift/geometry. Normalize by long-term mean class distance; degenerate geometry is inactive. Multiply existing reliability weights by max(exp(-clip(mean risk,0,5)),.05); renormalize available providers. No target selection.',
        split='Reuse original S03 long20 and S04 nested4/8/20 calibration, same124 evaluation trials. Zero current shots still consumes20 long-term trials. All data previously inspected, developmental retrospective comparison.',
        primary='Fixed five-shot full F7_F8_F9_structural versus baseline on identical124 trials: lower loss and Brier, nonworse macro-F1 and every class recall. Preserve all branches and removals even when guards fail; no target-chosen alternative/default promotion.',
        source_sha256=dependencies,artifact_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in artifacts},
        scope='Operational F7/F8/raw-F9 around six frozen EMG window providers, not every document subfamily, anatomical F6, full-bout F5, future-free continuous inference or device efficacy. Same single operator-identified user/day, inspected sessions and cued stable trial windows. New profile schema/policy identity cannot silently reuse old archives.',default_promoted=False)
    with PROTOCOL.open('x',encoding='utf8',newline='\n') as f:json.dump(p,f,indent=2);f.write('\n')
    print('Frozen F7/F8/raw-F9 candidate; no native data loaded',flush=True)


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Existing integrated experiment cannot be overwritten')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,digest in {**p['source_sha256'],**p['artifact_sha256']}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen integrated dependency changed: '+name)
    old=json.loads((HERE/'SONG_PERSONAL_SESSION_V1_RESULTS.json').read_text(encoding='utf8'))
    base=load_workflow(HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    gate=load_gate(HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    w=PersonalSessionDecisionV1(base,anchor_mix=p['anchor_mix'],gate=gate);source_before=pickle.dumps((base.bank,gate))
    long,_,_,_=native('S03',p);current,raw,starts,_=native('S04',p)
    kwargs=dict(user_id='Song',observed_channel_ids=CHANNELS,preprocessing_id=PREPROCESSING)
    OUT.mkdir();policy_path=OUT/'decision_policy.json'
    policy_path.write_text(json.dumps(dict(anchor_mix=p['anchor_mix'],source_bank_id=base.bank.bank_id_,base_contract_id=base.contract_id,
                                          gate_policy_id=gate.policy_id),indent=2)+'\n',encoding='utf8',newline='\n')
    eb,ei,eo=windows(current,old['evaluation_ids']);ey=np.asarray(old['evaluation_labels'])
    mask=np.isin(current['trial'],old['evaluation_ids']);raw_batch=eb.__class__(np.stack([raw[s:s+50] for s in starts[mask]]),250.)
    b,ids,o=windows(long,old['personal_calibration_ids']);labels=dict(zip(long['trial'],long['hand']))
    labels={t:labels[t] for t in np.unique(ids)}
    personal=w.enroll_user(b,ids,labels,window_offsets=o,session_id='S03',forbidden_evaluation_trials=old['evaluation_ids'],**kwargs)
    pp=OUT/'personal_S03.zip';w.save_profile(personal,pp);personal=w.load_profile(pp,user_id='Song');personal_before=pickle.dumps(personal)
    _,cy,cids=aggregate(current['batch'].emg.mean(1),current)
    cells=[];rows=[];profiles={};readouts={};roundtrips=[]
    for shots in p['current_shots']:
        session=None
        if shots:
            ci,_=select(cids,cy,shots,p['selection_seed']);b,ids,o=windows(current,cids[ci])
            session=w.calibrate_session(b,ids,dict(zip(cids[ci],cy[ci])),window_offsets=o,personal=personal,session_id='S04',
                forbidden_evaluation_trials=old['evaluation_ids'],**kwargs)
            path=OUT/f'session_S04_{shots}shot.zip';w.save_profile(session,path)
            session=w.load_profile(path,user_id='Song',session_id='S04',personal=personal)
            profiles[str(shots)]=dict(path=path.relative_to(ROOT).as_posix(),calibration_ids=list(session.base.calibration_trials),routing={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in session.routing.items()})
        state_before=pickle.dumps(session)
        for arm,options in arms(base.bank.providers_).items():
            result=w.predict(eb,ei,window_offsets=eo,personal=personal,session=session,session_id='S04',raw_batch=raw_batch,**kwargs,**options)
            q=result['probabilities']
            if result['quality_decision'] is not None and result['quality_decision']['rejected'].any():raise ValueError('Native unmodified rejection differs from previous experiment')
            if arm=='F7_standalone':
                q=sum(float(result['weights'][i])*result['anchor_probabilities'][n] for i,n in enumerate(base.bank.providers_));q/=q.sum(1,keepdims=True)
            cells.append(dict(shots=shots,arm=arm,long_term_calibration_trials=20,current_calibration_trials=4*shots,
                              anchor_enabled=result['anchor_enabled'],session_routing_enabled=result['session_routing_enabled'],**score(ey,q)))
            for t,y,prob in zip(old['evaluation_ids'],ey,q):rows.append(dict(shots=shots,arm=arm,trial_id=t,label=y,**{'p_'+c:float(prob[i]) for i,c in enumerate(base.bank.classes_)}))
            readouts[f'shots{shots}_{arm}_probabilities']=q
            if arm=='F7_F8':
                for n in base.bank.providers_:
                    readouts[f'shots{shots}_{n}_anchor_distances']=result['anchor_distances'][n]
                    readouts[f'shots{shots}_{n}_anchor_probabilities']=result['anchor_probabilities'][n]
                readouts[f'shots{shots}_weights']=np.array(result['weights'])
        restored=w.load_profile(pp,user_id='Song')
        a=w.predict(eb,ei,window_offsets=eo,personal=restored,session=session,session_id='S04',**kwargs)
        error=float(abs(a['probabilities']-readouts[f'shots{shots}_F7_F8_probabilities']).max())
        if error>1e-12 or pickle.dumps(personal)!=personal_before or pickle.dumps(session)!=state_before:raise ValueError('Profiles mutated or persistence changed probabilities')
        roundtrips.append(dict(shots=shots,maximum_probability_error=error))
        print(f'Current{shots}shot/class: all14 integrated/control/removal arms complete',flush=True)
    with (OUT/'predictions.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'readouts.npz',**readouts)
    lookup={(c['shots'],c['arm']):c for c in cells};a=lookup[(5,'baseline')];b=lookup[(5,'F7_F8_F9_structural')]
    guards=dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],
                nonworse_all_class_recall=all(b['recall'][c]>=a['recall'][c] for c in base.bank.classes_))
    if pickle.dumps((base.bank,gate))!=source_before:raise ValueError('Source model/gate mutation')
    r=dict(schema=p['schema'],protocol_sha256=sha(PROTOCOL),policy_path=policy_path.relative_to(ROOT).as_posix(),policy_sha256=sha(policy_path),
        decision_policy_id=w.policy_id,evaluation_ids=old['evaluation_ids'],evaluation_labels=old['evaluation_labels'],personal_calibration_ids=old['personal_calibration_ids'],
        reserved_current_ids=old['reserved_current_ids'],profiles=profiles,cells=cells,roundtrips=roundtrips,primary_guards=guards,primary_pass=all(guards.values()),
        source_and_profiles_immutable=True,default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=p['scope'],
        artifact_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in OUT.iterdir()})
    RESULT.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
