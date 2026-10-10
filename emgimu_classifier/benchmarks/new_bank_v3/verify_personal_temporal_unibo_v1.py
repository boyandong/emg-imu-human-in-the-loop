"""Read-only reconstruction of every retained temporal fusion cell and split."""
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json'
ARMS=('DTW_long','signature_long','DTW_local','DTW_blended','signature_blended',
      'base','base_DTW_long','base_DTW_blended','base_signature_blended','base_full','base_uniform')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf8'))


def metrics(y,q,w,classes):
    w=w/w.sum();prediction=q.argmax(1);recall={};f1={}
    for i,c in enumerate(classes):
        tp=w[(y==i)&(prediction==i)].sum();support=w[y==i].sum();guessed=w[prediction==i].sum()
        recall[c]=float(tp/support) if support else 0.
        f1[c]=float(2*tp/(support+guessed)) if support+guessed else 0.
    return dict(accuracy=float(w[y==prediction].sum()),macro_f1=float(np.mean(list(f1.values()))),
        log_loss=float(-np.sum(w*np.log(np.maximum(q[np.arange(len(y)),y],np.finfo(float).eps)))),
        brier=float(np.sum(w*((q-np.eye(len(classes))[y])**2).mean(1))),per_class_recall=recall,
        per_class_f1=f1)


def check_scores(saved,y,q,w,classes):
    result=metrics(y,q,w,classes)
    for key in ('accuracy','macro_f1','log_loss','brier'):
        np.testing.assert_allclose(saved[key],result[key],rtol=0,atol=1e-12)
    for field,actual in (('per_class_recall',saved['per_class_recall']),
                         ('per_class_f1',json.loads(saved['per_class_f1_json']))):
        # Saved G5 metrics use the established display labels.
        values=list(actual.values())
        np.testing.assert_allclose(values,list(result[field].values()),rtol=0,atol=1e-12)


def profile(path,user,classes,trial_ids,personal=None):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('manifest.json'));payload=z.read('profile.pkl')
    assert hashlib.sha256(payload).hexdigest()==manifest['payload_sha256']
    p=pickle.loads(payload)
    assert p.profile_id==manifest['profile_id'] and p.user_id==user and list(p.trial_ids)==trial_ids
    assert p.templates.shape==(len(classes),32,4) and p.signature_prototypes.shape==(len(classes),20)
    if personal is None:assert p.kind=='personal' and p.personal_profile_id is None
    else:
        assert p.kind=='session' and p.personal_profile_id==personal.profile_id
        np.testing.assert_array_equal(p.signature_mean,personal.signature_mean)
        np.testing.assert_array_equal(p.signature_scale,personal.signature_scale)
        assert set(p.recording_ids).isdisjoint(personal.recording_ids)
    return p


def verify():
    protocol=HERE/'PERSONAL_TEMPORAL_UNIBO_V1_PROTOCOL.json';result=HERE/'PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json'
    p,r=load(protocol),load(result)
    assert sha(protocol)==r['protocol_sha256']
    assert sha(HERE/'F5_PATH_UNIBO_PROTOCOL.json')==p['parent_protocol_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['external_sha256'].items():assert sha(Path(name))==digest,name
    for name,digest in r['artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    parent=load(HERE/'F5_PATH_UNIBO_PROTOCOL.json')
    metadata={v['id']:v for side in ('frozen_source','frozen_final')
        for v in load(Path(parent[side])/'bout_metadata.json')}
    splits={v['user']:v for v in load(Path(parent['frozen_source'])/'split_trial_ids.json')}
    frozen_base={}
    for side in ('frozen_source','frozen_final'):
        with np.load(Path(parent[side])/'heldout_predictions.npz',allow_pickle=False) as z:
            frozen_base.update({str(t):q for t,q in zip(z['bout_ids'],z['G5'])})
    classes=p['classes'];assert classes==['neutral','index_pinch','fist','open_hand']
    groups={};pooled={};predictions=0;max_error=0.
    archive=HERE/'personal_temporal_unibo_v1/readouts.npz'
    with np.load(archive,allow_pickle=False) as arrays:
        assert len(r['blocks'])==84 and len(arrays.files)==84*11
        for b in r['blocks']:
            user,day,shots=b['user'],b['day'],b['shots'];key=b['key']
            assert key==f'{user}_d{day}_s{shots}' and shots in (0,1,2,5) and day in (6,7,8)
            groups.setdefault((user,day),[]).append(b)
            selected=[];used=set()
            for c in range(4):
                candidates=sorted([v for v in metadata.values() if v['user']==user and v['day']==day and v['label']==c],
                    key=lambda v:hashlib.sha256((p['selection_seed']+'|'+v['id']).encode()).digest())
                chosen=[]
                for v in candidates:
                    if v['trial'] in used:continue
                    used.add(v['trial']);chosen.append(v['id'])
                    if len(chosen)==5:break
                assert len(chosen)==5;selected.extend(chosen)
            assert selected==b['reserved_ids'] and sorted(used)==b['reserved_recordings']
            expected=[] if not shots else [t for c in range(4) for t in [i for i in selected if metadata[i]['label']==c][:shots]]
            assert b['calibration_ids']==expected
            ids=b['evaluation_ids'];records=b['evaluation_recordings'];y=np.array(b['labels']);w=np.array(b['weights'])
            assert len(ids)==len(set(ids)) and records==[metadata[t]['trial'] for t in ids]
            assert set(records).isdisjoint(used) and len(ids)>0
            assert set(ids)=={v['id'] for v in metadata.values() if v['user']==user and v['day']==day and v['trial'] not in used}
            np.testing.assert_array_equal(y,[metadata[t]['label'] for t in ids])
            independent=[]
            for t in ids:
                v=metadata[t];same_record=[i for i in ids if metadata[i]['trial']==v['trial']]
                labels={metadata[i]['label'] for i in same_record}
                independent.append(1/(len(set(records))*len(labels)*sum(metadata[i]['label']==v['label'] for i in same_record)))
            np.testing.assert_allclose(w,independent,rtol=0,atol=1e-14)
            long=profile(ROOT/b['personal_profile'],user,classes,splits[user]['source_template_candidates'])
            assert set(records).isdisjoint(long.recording_ids)
            if shots:
                local=profile(ROOT/b['session_profile'],user,classes,expected,long)
                assert local.session_id==f'Day{day}' and set(records).isdisjoint(local.recording_ids)
            else:assert b['session_profile'] is None and not b['session_template_enabled']
            q={a:arrays[key+'_'+a] for a in ARMS}
            np.testing.assert_array_equal(q['base'],np.stack([frozen_base[t] for t in ids]))
            for a in ARMS:
                values=q[a];assert values.shape==(len(ids),4) and np.isfinite(values).all() and (values>=0).all()
                np.testing.assert_allclose(values.sum(1),1.,rtol=0,atol=1e-12)
                check_scores(b['scores'][a],y,values,w,classes);predictions+=len(ids)
                pooled.setdefault(('validation' if day==6 else 'descriptive_final',shots,a),[]).append((y,values,w))
                cost=(0,0) if a in ('base','base_uniform') else ((20,0) if a in ('DTW_long','signature_long','base_DTW_long') else ((0,4*shots) if a=='DTW_local' and shots else (20,4*shots)))
                assert b['predictive_calibration_cost'][a]==dict(long_term=cost[0],current=cost[1])
            for a in ('DTW_long','DTW_blended','signature_blended'):
                np.testing.assert_allclose(q['base_'+a],.75*q['base']+.25*q[a],rtol=0,atol=1e-14)
            np.testing.assert_allclose(q['base_full'],.75*q['base']+.125*(q['DTW_blended']+q['signature_blended']),rtol=0,atol=1e-14)
            np.testing.assert_allclose(q['base_uniform'],.75*q['base']+.25/4,rtol=0,atol=1e-14)
            if not shots:
                np.testing.assert_array_equal(q['DTW_long'],q['DTW_local'])
                np.testing.assert_array_equal(q['DTW_long'],q['DTW_blended'])
            max_error=max(max_error,b['oracle_max_probability_error'])
        for g in groups.values():
            assert len(g)==4
            assert all(b['evaluation_ids']==g[0]['evaluation_ids'] for b in g)
    assert len(groups)==21 and len(r['pooled'])==88 and max_error<=1e-10
    for v in r['pooled']:
        values=pooled[v['phase'],v['shots'],v['arm']]
        check_scores(v,np.concatenate([a for a,b,c in values]),np.concatenate([b for a,b,c in values]),np.concatenate([c for a,b,c in values]),classes)
    index={(v['phase'],v['shots'],v['arm']):v for v in r['pooled']}
    a,b=index['validation',5,'base'],index['validation',5,'base_full']
    wins=sum(v['scores']['base_full']['log_loss']<v['scores']['base']['log_loss'] for v in r['blocks'] if v['day']==6 and v['shots']==5)
    guards=dict(lower_log_loss=b['log_loss']<a['log_loss'],lower_brier=b['brier']<a['brier'],
        nonworse_macro_f1=b['macro_f1']>=a['macro_f1'],at_least5_of7_user_loss_wins=wins>=5)
    assert r['primary_guards']==guards and r['primary_pass']==all(guards.values()) and r['primary_user_loss_wins']==wins
    assert not any(r[k] for k in ('source_G5_refitted','default_promoted','physical_validation_proven','completion_proven'))
    return dict(schema='personal_temporal_unibo_acceptance_v1',verifier_sha256=sha(Path(__file__)),
        protocol_sha256=sha(protocol),result_sha256=sha(result),blocks=84,arm_cells=924,pooled_cells=88,
        retained_predictions=predictions,matched_evaluation_bouts=predictions//44,
        independent_metrics_splits_fusion_costs_profiles_verified=True,
        native_inrun_DP_medoid_signature_probability_oracle_max_error=max_error,
        primary_guards=guards,primary_pass=all(guards.values()),user_loss_wins=wins,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=p['scope'])


if __name__=='__main__':
    result=verify();OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))
