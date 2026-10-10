"""No-fit independent energy FSM, source logits, geometry and metric audit."""
import argparse
import csv
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from emgimu.datasets.roam_cued_intervals_v1 import load_roam_cued_intervals,CLASSES
from emgimu.feature_bank.personal_temporal_bouts_v1 import TemporalBoutBatchV1
from emgimu.feature_bank.native_bout_window_adapter_v2 import native_windows
from benchmarks.new_bank_v3.roam_native_continuous_v1 import ROOT,HERE,OUT,PROTOCOL,RESULT,ARMS,ACTIVE
from benchmarks.new_bank_v3.roam_native_joint_v1 import sha,write,build,GROUPS
from benchmarks.new_bank_v3.verify_roam_native_joint_v1 import manual_source,temporal_oracle
from benchmarks.new_bank_v3.verify_calibration_rest_continuous_unibo_v1 import thresholds,independent_intervals,independent_matches
from benchmarks.song_real8.verify_integrated_decision_v1 import covariance_oracle,probability,routing_oracle

RECEIPT=ROOT/'feature_bank/ROAM_NATIVE_CONTINUOUS_V1_ACCEPTANCE.json'


def independent_metrics(records,arm):
    total=sum(len(r['references']) for r in records);detected=sum(len(r['events']) for r in records)
    true=np.zeros(3,int);pred=np.zeros(3,int);correct=np.zeros(3,int);q=[];y=[];on=[];off=[]
    for r in records:
        for ref in r['references']:true[CLASSES.index(ref['class_name'])]+=1
        for e in r['events']:pred[CLASSES.index(e['labels'][arm])]+=1
        for m in r['matches']:
            ref=r['references'][m['reference_index']];e=r['events'][m['event_index']]
            yi=CLASSES.index(ref['class_name']);y.append(yi);q.append(e['probabilities'][arm])
            correct[yi]+=int(e['labels'][arm]==ref['class_name'])
            on.append(abs(m['onset_error_s']));off.append(abs(m['offset_error_s']))
    n=len(y);prob=np.asarray(q,float).reshape(-1,3);yy=np.array(y,int)
    recall={c:float(correct[i]/true[i]) if true[i] else 0. for i,c in enumerate(CLASSES) if c in ACTIVE}
    f1=[2*correct[i]/(true[i]+pred[i]) if true[i]+pred[i] else 0. for i,c in enumerate(CLASSES) if c in ACTIVE]
    return dict(references=total,detections=detected,matched=n,missed=total-n,unmatched_detections=detected-n,
        correct=int(correct.sum()),detection_recall=n/total if total else 0.,detection_precision=n/detected if detected else 0.,
        end_to_end_success=float(correct.sum()/total) if total else 0.,active_event_macro_f1=float(np.mean(f1)),
        active_recall=recall,conditional_accuracy=float(correct.sum()/n) if n else None,
        conditional_log_loss=float(-np.log(np.maximum(prob[np.arange(n),yy],1e-15)).sum()/n) if n else None,
        conditional_brier=float(np.square(prob-np.eye(3)[yy]).sum()/(3*n)) if n else None,
        onset_cue_mae_s=float(sum(on)/n) if n else None,offset_cue_mae_s=float(sum(off)/n) if n else None,
        eof_censored_recordings=sum(int(r['censored_end']) for r in records))


def window_oracle(bank,batch,personal,session):
    b,ids,_,_=native_windows(batch,40,8)
    axis,raw=manual_source(bank.families_,bank.models_,b,ids,bank.temperatures_)
    z={}
    for g in GROUPS:
        pieces=[]
        for family in bank.families_[g]:
            v=family.transform(b);pieces.append(np.stack([v[ids==t].mean(0) for t in axis]))
        v=np.concatenate(pieces,axis=1);scaler,_=bank.models_[g]
        v-=scaler.mean_.astype(v.dtype);v/=scaler.scale_.astype(v.dtype);z[g]=v
    p=personal.window.decision;s=None if session is None else session.window.decision
    covariance=covariance_oracle(b,ids);heads={}
    for g in GROUPS:
        if g=='F2ac':prototypes=p.spd_prototypes if s is None else s.spd_blended
        else:
            anchor=p.base.anchors[g] if s is None else s.base.anchors[g]['blended']
            prototypes=anchor.prototypes_[[list(anchor.classes_).index(c) for c in CLASSES]]
        anchor,_=probability(covariance if g=='F2ac' else z[g],prototypes,g=='F2ac')
        heads[g]=.5*raw[g]+.5*anchor
    weights=np.asarray(p.base.fusion_state.fusion_state.weights if s is None else s.base.fusion_state.weights,float)
    if s is not None:
        long={g:p.base.anchors[g].prototypes_ for g in GROUPS}
        local={g:s.base.anchors[g]['local'].prototypes_ for g in GROUPS}
        weights*=routing_oracle(long,local,p.spd_prototypes,s.spd_local,p,s)
    weights/=weights.sum()
    window=sum(weights[i]*heads[g] for i,g in enumerate(GROUPS));window/=window.sum(1,keepdims=True)
    source=sum(bank.policy_.population[i]*raw[g] for i,g in enumerate(GROUPS));source/=source.sum(1,keepdims=True)
    return source,window


def close(actual,expected):
    assert actual.keys()==expected.keys()
    for k,v in expected.items():
        if isinstance(v,dict):close(actual[k],v)
        elif v is None:assert actual[k] is None,k
        elif isinstance(v,(int,bool)):assert actual[k]==v,k
        else:assert abs(actual[k]-v)<1e-12,(k,actual[k],v)


def verify():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));r=json.loads(RESULT.read_text(encoding='utf8'))
    assert r['protocol_sha256']==sha(PROTOCOL) and sha(Path(p['archive']))==p['archive_sha256']
    for rel,h in {**p['source_sha256'],**p['artifact_sha256'],**r['artifacts_sha256']}.items():assert sha(ROOT/rel)==h,rel
    records=json.loads((OUT/'recordings.json').read_text(encoding='utf8'))
    assert len(records)==60 and len({(v['user'],v['shots'],v['member']) for v in records})==60
    assert len({v['member'] for v in records})==20
    assert {v['shots'] for v in records}=={0,1,2}
    bank=pickle.loads((HERE/'roam_native_joint_v1/source_bank.pkl').read_bytes());w=build(bank)
    before=pickle.dumps(w);threshold_error=probability_error=0.;event_count=0;unique_refs=set();native_receipts={}
    with zipfile.ZipFile(p['archive']) as z:
        for row in records:
            u=row['user'];user=f'ROAM_s{u}';d=load_roam_cued_intervals(z,row['member']);x=np.concatenate(d.batch.sequences)
            assert row['native_samples']==len(x)==d.receipt['native_samples'] and row['member_sha256']==d.receipt['member_sha256']
            native_receipts[row['member']]=d.receipt
            refs=[v for v in d.receipt['cue_intervals'] if v['class_name'] in ACTIVE]
            assert refs==row['references'];unique_refs.update(v['trial_id'] for v in refs)
            personal=w.load_profile(ROOT/row['personal_path'],user_id=user)
            session=None if row['session_path'] is None else w.load_profile(ROOT/row['session_path'],user_id=user,session_id='hanging',personal=personal)
            saved=pickle.dumps((personal,session));profile=session or personal
            assert profile.profile_id==row['detector_profile_id']
            for pf in (personal,session):
                if pf is None:continue
                assert row['member'] not in pf.calibration.recording_ids
                assert set(bank.policy_.source_trials).isdisjoint(pf.calibration.trial_ids)
                assert pf.calibration_cost['counted_once'] and not pf.calibration_cost['resampled']
                original=load_roam_cued_intervals(z,next(iter(set(pf.calibration.recording_ids)))).batch
                for t,v in zip(pf.calibration.trial_ids,pf.calibration.sequences):
                    np.testing.assert_array_equal(v,original.sequences[original.trial_ids.index(t)])
            assert row['unique_calibration_trials']==len(personal.calibration.trial_ids)+(0 if session is None else len(session.calibration.trial_ids))==6+3*row['shots']
            rest=np.concatenate([v for t,v in zip(profile.calibration.trial_ids,profile.calibration.sequences) if dict(profile.calibration_labels)[t]=='relax'])
            on,off,width=thresholds(rest,200.)
            threshold_error=max(threshold_error,abs(on-profile.detector.on_),abs(off-profile.detector.off_))
            assert profile.detector.width_==width and profile.detector.policy==(.08,.12,.1,1.,30.)
            bounds,censored=independent_intervals(x,on,off)
            assert bounds==[[e['start'],e['end']] for e in row['events']] and censored==row['censored_end']
            pairs=independent_matches(bounds,[dict(v,native_label=v['class_name']) for v in refs])
            assert [{k:v for k,v in m.items() if k!='label'} for m in pairs]==row['matches']
            for ordinal,e in enumerate(row['events']):
                event_count+=1;a,b=e['start'],e['end']
                assert e['trial_id']==row['member']+f':estimated{ordinal}' and e['recording_id']==row['member']
                assert e['available_at_sample_index']==b and e['boundary_kind']=='estimated' and e['quality_policy']=='off'
                assert e['class_names']==list(CLASSES) and e['personal_profile_id']==personal.profile_id
                assert e['session_profile_id']==(None if session is None else session.profile_id)
                batch=TemporalBoutBatchV1((x[a:b],),(e['trial_id'],),(row['member'],),(a,),200.,d.batch.channel_ids,d.batch.preprocessing_id,'estimated')
                source,window=window_oracle(bank,batch,personal,session)
                ld,ls=temporal_oracle(batch,personal.temporal);dtw,sig=ld,ls
                if session is not None:
                    sd,ss=temporal_oracle(batch,session.temporal);dtw=.5*(ld+sd);sig=.5*(ls+ss)
                expected=dict(source_window=source[0],window_full=window[0],joint_full=(.75*window+.125*dtw+.125*sig)[0])
                for arm,q in expected.items():
                    q=q/q.sum();stored=np.array(e['probabilities'][arm]);assert stored.shape==(3,) and np.isfinite(stored).all()
                    probability_error=max(probability_error,float(abs(q-stored).max()))
                    assert e['labels'][arm]==CLASSES[int(q.argmax())]
                assert e['unrepresented_tail_samples']==(b-a-40)%8
            assert pickle.dumps((personal,session))==saved
    assert len(unique_refs)==100 and [native_receipts[v['member']] for v in p['native_recordings']]==p['native_recordings']
    assert pickle.dumps(w)==before and max(probability_error,threshold_error)<1e-12
    assert len(r['cells'])==90 and len(r['aggregates'])==27
    for cell in r['cells']:
        selected=[x for x in records if x['user']==cell['user'] and x['shots']==cell['shots']]
        close(cell['metrics'],independent_metrics(selected,cell['arm']))
        assert cell['unique_calibration_trials']==6+3*cell['shots']
    for cell in r['aggregates']:
        users=range(19,24) if cell['phase']=='validation' else range(24,29) if cell['phase']=='descriptive_final' else range(19,29)
        close(cell['metrics'],independent_metrics([x for x in records if x['user'] in users and x['shots']==cell['shots']],cell['arm']))
    with (OUT/'events.csv').open(encoding='utf8',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==event_count*3
    expected_csv={}
    for row in records:
        refs={m['event_index']:row['references'][m['reference_index']] for m in row['matches']}
        for i,e in enumerate(row['events']):
            for arm in ARMS:expected_csv[row['user'],row['shots'],row['member'],e['trial_id'],arm]=(e,refs.get(i))
    seen=set()
    for row in csvrows:
        key=int(row['user']),int(row['shots']),row['member'],row['event_id'],row['arm'];assert key not in seen;seen.add(key)
        e,ref=expected_csv[key];assert int(row['start'])==e['start'] and int(row['end'])==e['end']
        assert row['reference_id']==('' if ref is None else ref['trial_id'])
        assert row['reference_label']==('' if ref is None else ref['class_name'])
        assert row['predicted']==e['labels'][row['arm']]
        assert json.loads(row['probabilities_json'])==e['probabilities'][row['arm']]
    assert seen==set(expected_csv)
    aggregate={(v['phase'],v['shots'],v['arm']):v['metrics'] for v in r['aggregates']}
    base=aggregate['validation',2,'source_window'];joint=aggregate['validation',2,'joint_full']
    cells={(v['user'],v['shots'],v['arm']):v['metrics'] for v in r['cells']}
    wins=sum(cells[u,2,'joint_full']['correct']>cells[u,2,'source_window']['correct'] for u in range(19,24))
    guards=dict(higher_end_to_end_success=joint['correct']>base['correct'],
        nonworse_active_recalls=all(joint['active_recall'][c]>=base['active_recall'][c] for c in ACTIVE),
        nonworse_active_event_f1=joint['active_event_macro_f1']>=base['active_event_macro_f1'],
        lower_conditional_log_loss=joint['conditional_log_loss'] is not None and joint['conditional_log_loss']<base['conditional_log_loss'],
        lower_conditional_brier=joint['conditional_brier'] is not None and joint['conditional_brier']<base['conditional_brier'],at_least3_user_success_wins=wins>=3)
    assert guards==r['primary_guards'] and all(guards.values())==r['primary_pass'] and wins==r['validation_user_success_wins']
    assert not r['classifier_refitted'] and not r['detector_refitted'] and r['source_profiles_unchanged']
    assert not r['default_promoted'] and not r['physical_validation_proven'] and not r['completion_proven']
    return dict(schema='roam_native_continuous_v1_acceptance',protocol_sha256=sha(PROTOCOL),result_sha256=sha(RESULT),
        verifier_sha256=sha(Path(__file__)),recording_budget_blocks=60,distinct_query_recordings=20,independent_active_references=100,
        cells=90,detected_event_budget_rows=event_count,csv_probability_rows=event_count*3,
        maximum_independent_probability_error=probability_error,maximum_rest_threshold_error=threshold_error,
        independent_native_energy_FSM=True,independent_logits_anchor_routing_path_and_metric_equations=True,
        all_misses_and_unmatched_detections_retained=True,calibration_query_recordings_disjoint=True,
        primary_guards=guards,primary_pass=all(guards.values()),validation_user_success_wins=wins,
        classifier_refitted=False,detector_refitted=False,read_only_no_fitting=True,
        physiological_boundary_ground_truth=False,physical_validation_proven=False,default_promoted=False,completion_proven=False,
        two_shot_summary=[v for v in r['aggregates'] if v['shots']==2],scope=r['scope'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-receipt',action='store_true');args=p.parse_args()
    receipt=verify()
    if args.write_receipt:write(RECEIPT,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('two_shot_summary','scope')}))
