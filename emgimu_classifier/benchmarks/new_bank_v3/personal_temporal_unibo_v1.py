"""New full-bout personal/session fusion; frozen G5 predictions are reused."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from emgimu.datasets.benchmark import load_benchmark_trial
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1,PersonalTemporalBoutsV1
from emgimu.feature_bank.unibo_sequence_temporal import bout_weights
from emgimu.feature_bank.unibo_study import _metrics

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'PERSONAL_TEMPORAL_UNIBO_V1_PROTOCOL.json'
RESULT=HERE/'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json'
OUT=HERE/'personal_temporal_unibo_v1'
CLASSES=('neutral','index_pinch','fist','open_hand')
FILES=['src/emgimu/feature_bank/personal_temporal_bouts_v1.py',
    'src/emgimu/feature_bank/complete_bout_window_adapter_v1.py',
    'src/emgimu/feature_bank/personal_temporal_cli_v1.py',
    'src/emgimu/feature_bank/document_path_v3.py','src/emgimu/feature_bank/temporal.py',
    'src/emgimu/feature_bank/core.py','src/emgimu/feature_bank/unibo_sequence_temporal.py',
    'src/emgimu/feature_bank/unibo_study.py','src/emgimu/feature_bank/screening.py',
    'src/emgimu/datasets/benchmark.py','src/emgimu/state.py','src/emgimu/datasets/unibo_baseline.py',
    'tests/test_personal_temporal_bouts_v1.py','tests/test_personal_temporal_cli_v1.py',
    'tests/test_personal_temporal_native_oracles_v1.py',
    'benchmarks/new_bank_v3/personal_temporal_unibo_v1.py']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    parent=HERE/'F5_PATH_UNIBO_PROTOCOL.json';p=json.loads(parent.read_text())
    external={str(HERE/'F5_PATH_UNIBO_RESULTS.json'):sha(HERE/'F5_PATH_UNIBO_RESULTS.json')}
    for folder in ('frozen_source','frozen_final'):
        for name in ('bout_metadata.json','split_trial_ids.json','heldout_predictions.npz','run_manifest.json'):
            path=Path(p[folder])/name;external[str(path)]=sha(path)
    state=Path(p['frozen_source'])/'fitted_states.pkl';external[str(state)]=sha(state)
    protocol=dict(schema='personal_temporal_unibo_v1_protocol',parent_protocol_sha256=sha(parent),
        external_sha256=external,source_sha256={name:sha(ROOT/name) for name in FILES},
        classes=CLASSES,current_shots=[0,1,2,5],source_profile_trials=20,
        selection_seed='20261011_personal_temporal_unibo_v1',temporal_mix=.25,band_fraction=.1,
        source_rule='Reuse exact frozen G5 probabilities/models; never refit G5 or temperatures. Long-term temporal profile uses the same20 frozen source candidate bouts/user from Days1-5. Source history overlaps the personalized G5 training history and is disclosed, not a generic-to-new-user trial.',
        calibration_rule='For each user and each Day6/7/8, reserve5 complete bouts/class deterministically by SHA256(seed|boutid), requiring distinct recording trials across all20. Nested1/2/5 shots/class. Exclude ALL bouts of every reserved recording from evaluation at all budgets, including0; fit local templates only on selected calibration bouts.',
        temporal='All native samples ->32 equal-duration RMS bins -> envelope/(L2+1e-10). Per-class medoids with cumulative Euclidean Sakoe-Chiba band3, shortest equal-cost path, divided by selected path length. Temperature=mean inter-class medoid DTW distance. Order1/2 signatures without absolute time; long-calibration mean/std reused by local class means; Euclidean class-geometry temperature.',
        fusion='Fixed .75 base+.25 DTW or signature; full=.75base+.125DTW+.125signature. Long/local temporal probability blend .5/.5. Include base softened toward uniform as a same-mixture control. No target policy/temperature search.',
        primary='Day6 five-shot full versus base: lower equal-user loss/Brier, nonworse pooled macroF1 and at least5/7 user loss wins. Preserve all budgets, all11 arms, all three day axes and all provider/long-local controls. Days7/8 descriptive, no selection.',
        scope='Previously inspected UniBo four-channel200Hz native oracle-labelled complete bouts. Native calibration/evaluation recording trials disjoint. Separate eight-channel250Hz window-adapter/CLI tests do not prove real eight-channel complete-action accuracy. Song stable-only excerpts are ineligible. Estimated boundaries stay estimated; output waits for interval end. No automatic segmentation or device/default efficacy claim.',
        base_cost='Existing G5 models were trained on personalized Days1-5 history. Reported additional20/0/4/8/20 trial costs concern temporal registration, not total source history. DTW_local at positive budgets mathematically uses only current medoids; the lifecycle still requires a long-term profile. Signature local/blended uses long-term standardization.',
        default_promoted=False)
    with PROTOCOL.open('x',encoding='utf8',newline='\n') as f:
        json.dump(protocol,f,indent=2);f.write('\n')


def dtw_oracle(a,b,band):
    local=np.linalg.norm(a[:,None]-b[None],axis=2)
    total=np.full((len(a)+1,len(b)+1),np.inf);length=np.zeros(total.shape,int);total[0,0]=0.
    for i in range(1,len(a)+1):
        for j in range(max(1,i-band),min(len(b),i+band)+1):
            cost,n=min(((total[i-1,j-1],length[i-1,j-1]),(total[i-1,j],length[i-1,j]),
                (total[i,j-1],length[i,j-1])),key=lambda v:(v[0],v[1]))
            total[i,j]=cost+local[i-1,j-1];length[i,j]=n+1
    return total[-1,-1]/length[-1,-1]


def path_and_signature(batch):
    paths=[];signatures=[]
    for raw in batch.sequences:
        envelope=np.stack([np.sqrt(np.einsum('tc,tc->c',v.astype(float),v.astype(float))/len(v))
            for v in np.array_split(raw,32)])
        path=envelope/(np.linalg.norm(envelope,axis=1,keepdims=True)+1e-10)
        first=np.zeros(path.shape[1]);second=np.zeros((path.shape[1],path.shape[1]))
        for change in np.diff(path,axis=0):
            second+=np.outer(first,change)+.5*np.outer(change,change);first+=change
        paths.append(path);signatures.append(np.r_[first,second.ravel()])
    return np.stack(paths),np.stack(signatures)


def oracle_read(paths,signature,profile,band):
    def probability(distance,geometry):
        temperature=geometry[np.triu_indices(len(geometry),1)].mean()
        if temperature<=1e-10:return np.full(distance.shape,1/distance.shape[1])
        z=-distance/temperature;z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)
    distance=np.array([[dtw_oracle(x,t,band) for t in profile.templates] for x in paths])
    geometry=np.array([[dtw_oracle(a,b,band) for b in profile.templates] for a in profile.templates])
    z=(signature-profile.signature_mean)/profile.signature_scale
    d=np.linalg.norm(z[:,None]-profile.signature_prototypes[None],axis=2)
    g=np.linalg.norm(profile.signature_prototypes[:,None]-profile.signature_prototypes[None],axis=2)
    return probability(distance,geometry),probability(d,g)


def verify_profile(batch,labels,profile,personal=None):
    paths,signature=path_and_signature(batch)
    for i,c in enumerate(CLASSES):
        positions=[j for j,t in enumerate(batch.trial_ids) if labels[t]==c]
        d=np.array([[dtw_oracle(paths[a],paths[b],3) for b in positions] for a in positions])
        # Runtime fills one triangle then mirrors it. Keep that tie rule when
        # arithmetic on the two DP traversals differs at floating precision.
        d=np.triu(d,1);d+=d.T
        chosen=positions[int(d.sum(1).argmin())]
        assert profile.medoid_trial_ids[i]==batch.trial_ids[chosen]
        np.testing.assert_allclose(profile.templates[i],paths[chosen],rtol=0,atol=1e-14)
    mean=signature.mean(0) if personal is None else personal.signature_mean
    scale=signature.std(0) if personal is None else personal.signature_scale
    scale=np.where(scale>1e-10,scale,1.)
    np.testing.assert_allclose(profile.signature_mean,mean,rtol=0,atol=1e-14)
    np.testing.assert_allclose(profile.signature_scale,scale,rtol=0,atol=1e-14)
    y=np.array([labels[t] for t in batch.trial_ids])
    expected=np.stack([((signature-mean)/scale)[y==c].mean(0) for c in CLASSES])
    np.testing.assert_allclose(profile.signature_prototypes,expected,rtol=0,atol=1e-10)


def select(rows,seed):
    selected=[];used=set()
    for label in range(4):
        ordered=sorted([r for r in rows if r['label']==label],key=lambda r:hashlib.sha256((seed+'|'+r['id']).encode()).digest())
        chosen=[]
        for r in ordered:
            if r['trial'] in used:continue
            used.add(r['trial']);chosen.append(r)
            if len(chosen)==5:break
        if len(chosen)!=5:raise ValueError('Five distinct recording trials/class unavailable')
        selected.extend(chosen)
    return selected


def costs(arm,shots):
    if arm in ('base','base_uniform'):return (0,0)
    if arm in ('DTW_long','signature_long','base_DTW_long'):return (20,0)
    if arm=='DTW_local' and shots:return (0,4*shots)
    return (20,4*shots)


def score(y,q,weights):
    result=_metrics(y,q,weights)
    prediction=q.argmax(1)
    result['per_class_recall']={c:float(weights[(y==i)&(prediction==i)].sum()/weights[y==i].sum())
        for i,c in enumerate(CLASSES)}
    return result


def run():
    if RESULT.exists() or OUT.exists():raise FileExistsError('Native experiment already exists; never overwrite/rerun')
    p=json.loads(PROTOCOL.read_text());parent=json.loads((HERE/'F5_PATH_UNIBO_PROTOCOL.json').read_text())
    assert sha(HERE/'F5_PATH_UNIBO_PROTOCOL.json')==p['parent_protocol_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['external_sha256'].items():assert sha(Path(name))==digest,name
    source=Path(parent['frozen_source']);final=Path(parent['frozen_final'])
    splits=json.loads((source/'split_trial_ids.json').read_text())
    metadata=json.loads((source/'bout_metadata.json').read_text())+json.loads((final/'bout_metadata.json').read_text())
    by_id={r['id']:r for r in metadata};assert len(by_id)==len(metadata)
    wanted={i for s in splits for i in s['source_template_candidates']}|{r['id'] for r in metadata if r['day'] in (6,7,8)}
    files={f.stem:f for f in (Path(parent['dataset_root'])/'trials').rglob('*.npz')}
    raw={};by_file={}
    for i in wanted:by_file.setdefault(by_id[i]['trial'],[]).append(by_id[i])
    print('1/3 Read complete native envelopes; retain source G5 probabilities',flush=True)
    for trial_id,rows in by_file.items():
        path=files[trial_id];digest=sha(path);trial=load_benchmark_trial(path,expected_channels=4,expected_rate_hz=200.)
        assert trial.benchmark_eligible
        for r in rows:
            assert digest==r['raw_sha256'] and np.all(trial.hand_label[r['start']:r['end']]==r['label'])
            raw[r['id']]=trial.emg[r['start']:r['end']].copy()
    base={}
    for folder in (source,final):
        with np.load(folder/'heldout_predictions.npz',allow_pickle=False) as z:
            for i,t in enumerate(z['bout_ids']):
                assert by_id[str(t)]['label']==int(z['labels'][i]);base[str(t)]=z['G5'][i].copy()
    old=json.loads((HERE/'F5_PATH_UNIBO_RESULTS.json').read_text())
    old_medoids={v['user']:v['medoid_ids'] for v in old['source_selection']}
    def batch(rows):
        return TemporalBoutBatchV1(tuple(raw[r['id']] for r in rows),tuple(r['id'] for r in rows),
            tuple(r['trial'] for r in rows),tuple(r['start'] for r in rows),200.,('EMG1','EMG2','EMG3','EMG4'),
            'unibo_native_processed200_complete_bout_v1','complete_cued')
    def labels(rows):return {r['id']:CLASSES[r['label']] for r in rows}
    OUT.mkdir();arrays={};blocks=[];aggregate={};artifacts={}
    for split in splits:
        user=split['user'];long_rows=[by_id[i] for i in split['source_template_candidates']]
        assert len(long_rows)==20 and all(r['day']<=5 for r in long_rows)
        config=dict(source_bank_id=sha(source/'fitted_states.pkl')+'|'+user,sample_rate_hz=200.,
            channel_ids=['EMG1','EMG2','EMG3','EMG4'],preprocessing_id='unibo_native_processed200_complete_bout_v1',
            class_names=list(CLASSES),temporal_mix=p['temporal_mix'],band_fraction=p['band_fraction'])
        w=PersonalTemporalBoutsV1(**config)
        config_path=OUT/(user+'_config.json');config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf8',newline='\n')
        personal=w.enroll(batch(long_rows),labels(long_rows),user_id=user,session_id='Days1to5')
        verify_profile(batch(long_rows),labels(long_rows),personal)
        assert list(personal.medoid_trial_ids)==old_medoids[user]
        pp=OUT/(user+'_personal.zip');w.save_profile(personal,pp);personal=w.load_profile(pp,user_id=user)
        immutable=pickle.dumps((w,personal))
        for day in (6,7,8):
            rows=[r for r in metadata if r['user']==user and r['day']==day]
            reserved=select(rows,p['selection_seed']);reserved_records={r['trial'] for r in reserved}
            evaluation=[r for r in rows if r['trial'] not in reserved_records]
            assert evaluation and not {r['trial'] for r in long_rows}&{r['trial'] for r in evaluation}
            b=batch(evaluation);paths,signature=path_and_signature(b);q=np.stack([base[r['id']] for r in evaluation])
            truth=np.array([r['label'] for r in evaluation]);weights=bout_weights(evaluation);weights/=weights.sum()
            long_dtw,long_sig=oracle_read(paths,signature,personal,w.band)
            for shots in p['current_shots']:
                selected=[] if shots==0 else [r for c in range(4) for r in [v for v in reserved if v['label']==c][:shots]]
                session=None;sp=None
                if selected:
                    session=w.enroll(batch(selected),labels(selected),user_id=user,session_id=f'Day{day}',personal=personal,
                        forbidden_trial_ids=b.trial_ids,forbidden_recording_ids=b.recording_ids)
                    verify_profile(batch(selected),labels(selected),session,personal)
                    sp=OUT/f'{user}_Day{day}_{shots}shot.zip';w.save_profile(session,sp);session=w.load_profile(sp,user_id=user)
                frozen=pickle.dumps(session)
                result=w.predict(b,personal=personal,session=session,user_id=user,session_id=f'Day{day}',
                    base_probabilities=q,base_trial_ids=b.trial_ids,base_class_names=CLASSES)
                local_dtw,local_sig=(long_dtw,long_sig) if session is None else oracle_read(paths,signature,session,w.band)
                expected=dict(DTW_long=long_dtw,signature_long=long_sig,DTW_local=local_dtw,
                    DTW_blended=(long_dtw+local_dtw)/2,signature_blended=(long_sig+local_sig)/2,base=q)
                for name in ('DTW_long','DTW_blended','signature_blended'):
                    expected['base_'+name]=.75*q+.25*expected[name]
                expected['base_full']=.75*q+.125*(expected['DTW_blended']+expected['signature_blended'])
                expected['base_uniform']=.75*q+.25/4
                key=f'{user}_d{day}_s{shots}';error=0.;scores={}
                for arm,values in result['arms'].items():
                    error=max(error,float(np.max(abs(values-expected[arm]))));assert error<=1e-10,(key,arm,error)
                    arrays[key+'_'+arm]=values;scores[arm]=score(truth,values,weights)
                    phase='validation' if day==6 else 'descriptive_final'
                    aggregate.setdefault((phase,shots,arm),[]).append((user,truth,values,weights))
                np.testing.assert_allclose(result['signature_coordinates'],signature,rtol=0,atol=1e-13)
                assert pickle.dumps((w,personal))==immutable and pickle.dumps(session)==frozen
                blocks.append(dict(key=key,user=user,day=day,shots=shots,evaluation_ids=list(b.trial_ids),
                    evaluation_recordings=list(b.recording_ids),labels=truth.tolist(),weights=weights.tolist(),
                    reserved_ids=[r['id'] for r in reserved],reserved_recordings=sorted(reserved_records),
                    calibration_ids=[r['id'] for r in selected],personal_profile=pp.relative_to(ROOT).as_posix(),
                    session_profile=None if sp is None else sp.relative_to(ROOT).as_posix(),
                    oracle_max_probability_error=error,session_template_enabled=shots>0,
                    predictive_calibration_cost={a:dict(long_term=costs(a,shots)[0],current=costs(a,shots)[1]) for a in scores},
                    lifecycle_requires_long20_except_base_controls=True,scores=scores))
            print(f'2/3 {user} Day{day}: {len(evaluation)} evaluation bouts; four budgets/11 arms verified',flush=True)
    pooled=[]
    for (phase,shots,arm),values in aggregate.items():
        y=np.concatenate([v[1] for v in values]);q=np.concatenate([v[2] for v in values]);weights=np.concatenate([v[3] for v in values])
        pooled.append(dict(phase=phase,shots=shots,arm=arm,**score(y,q,weights)))
    lookup={(r['phase'],r['shots'],r['arm']):r for r in pooled}
    a,b=lookup['validation',5,'base'],lookup['validation',5,'base_full']
    user_wins=sum(v['scores']['base_full']['log_loss']<v['scores']['base']['log_loss'] for v in blocks if v['day']==6 and v['shots']==5)
    guards=dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],
        nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],at_least5_of7_user_loss_wins=user_wins>=5)
    with (OUT/'readouts.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    artifacts={f.relative_to(ROOT).as_posix():sha(f) for f in OUT.iterdir() if f.is_file()}
    result=dict(schema='personal_temporal_unibo_v1',protocol_sha256=sha(PROTOCOL),artifact_sha256=artifacts,
        blocks=blocks,pooled=pooled,primary_guards=guards,primary_pass=all(guards.values()),primary_user_loss_wins=user_wins,
        independent_DP_medoids_probabilities_and_signatures=True,source_and_profiles_immutable=True,
        source_G5_refitted=False,default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=p['scope'])
    with RESULT.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print('3/3 '+json.dumps(dict(blocks=len(blocks),primary_guards=guards,user_loss_wins=user_wins)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
