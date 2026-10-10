"""No-fit native oracle for integrated F7 geometry, F8 weights and every removal."""
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.linalg import eigvalsh
from benchmarks.song_real8.song_raw_quality_v1 import native
from benchmarks.song_real8.song_personal_session_workflow_v1 import windows, CHANNELS, PREPROCESSING
from emgimu.feature_bank.personal_session_decision_cli_v1 import load_decision_workflow

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'SONG_INTEGRATED_DECISION_V1_PROTOCOL.json';RESULT=HERE/'SONG_INTEGRATED_DECISION_V1_RESULTS.json'
OUT=ROOT/'feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def covariance_oracle(batch,ids):
    x=batch.emg.astype(float);x=x-x.mean(1,keepdims=True)
    covariance=np.stack([v.T@v/(len(v)-1) for v in x]);trace=np.trace(covariance,axis1=1,axis2=2)
    covariance=.95*covariance+.05*trace[:,None,None]/8*np.eye(8)
    covariance=covariance/(np.trace(covariance,axis1=1,axis2=2)[:,None,None]+1e-12)+1e-10*np.eye(8)
    return np.stack([covariance[ids==t].mean(0) for t in np.unique(ids)])


def distance(x,p,spd=False):
    if not spd:return np.stack([np.linalg.norm(x-v,axis=1) for v in p],axis=1)
    return np.array([[np.linalg.norm(np.log(eigvalsh(v,prototype))) for prototype in p] for v in x])


def probability(x,p,spd=False):
    d=distance(x,p,spd);g=distance(p,p,spd);temperature=float(g[np.triu_indices(len(p),1)].mean())
    if temperature<=1e-10:return np.full(d.shape,1/len(p)),d
    logits=-d/temperature;logits-=logits.max(1,keepdims=True);q=np.exp(logits);q/=q.sum(1,keepdims=True)
    return q,d


def prototypes(w,item,ids):
    b,wi,o=windows(item,ids);readout=w.bank.predict_providers(b,wi,window_offsets=o,user_id='Song')
    labels=dict(zip(item['trial'],item['hand']));y=np.array([labels[t] for t in readout['trial_ids']])
    # Source standardization preserves float32; prototype fitting then casts
    # trial coordinates to float64 before the equal-trial class mean.
    z={g:np.stack([v[y==c].astype(float).mean(0) for c in w.bank.classes_]) for g,v in readout['source_standardized_features'].items()}
    covariance=covariance_oracle(b,wi);sp=np.stack([covariance[y==c].mean(0) for c in w.bank.classes_])
    return z,sp,b,wi


def routing_oracle(long,local,spd_long,spd_local,p,s):
    pairs=list(combinations(range(len(spd_long)),2));classes=tuple(p.base.anchors['F0'].classes_)
    ref=p.base.session_reference.reference_;entries=[s.base.descriptor['classes'][str(c)] for c in classes]
    def ratio(value,scale):return float(value/scale) if scale>1e-10 else 0.
    log=np.array([float(ref[c]['log_scale']) for c in classes])
    specific={'F0':ratio(np.mean([abs(e['log_global_activation_shift']) for e in entries]),np.mean([abs(log[i]-log[j]) for i,j in pairs]))
              +float(np.mean([np.mean(abs(np.array(e['channel_quality_score_shift']))) for e in entries]))}
    for group,key,field in [('F1','pattern','scale_pattern_residual_norm'),('F4abc','log_bands','mean_absolute_log_band_residual')]:
        geom=np.mean([np.linalg.norm(ref[classes[i]][key]-ref[classes[j]][key]) if group=='F1' else np.abs(ref[classes[i]][key]-ref[classes[j]][key]).mean() for i,j in pairs])
        specific[group]=ratio(np.mean([e[field] for e in entries]),geom)
    risks=[];phi=[]
    for name in long:
        a,b=long[name],local[name]
        residual=np.linalg.norm(b-a,axis=1);cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-10)
        geometry=np.array([np.linalg.norm(b[i]-b[j])-np.linalg.norm(a[i]-a[j]) for i,j in pairs])
        phi.extend(residual);phi.extend(cosine);phi.extend(geometry)
        if name=='F2ac':
            a,b=spd_long,spd_local
            between=np.array([distance(a[i:i+1],a[j:j+1],True).item() for i,j in pairs])
            changed=np.array([distance(b[i:i+1],b[j:j+1],True).item() for i,j in pairs])-between
            shift=np.array([distance(a[i:i+1],b[i:i+1],True).item() for i in range(len(a))])
        else:
            between=np.array([np.linalg.norm(a[i]-a[j]) for i,j in pairs]);changed=geometry;shift=residual
        terms=[ratio(shift.mean(),between.mean()),ratio(abs(changed).mean(),between.mean())]
        if name in specific:terms.append(specific[name])
        risks.append(np.mean(terms))
    risks=np.array(risks)
    np.testing.assert_allclose(s.routing['descriptor'],phi,rtol=0,atol=1e-11)
    np.testing.assert_allclose(s.routing['risks'],risks,rtol=0,atol=1e-11)
    return np.maximum(np.exp(-np.minimum(risks,5)),.05)


def run():
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));r=json.loads(RESULT.read_text(encoding='utf8'))
    assert sha(PROTOCOL)==r['protocol_sha256']
    for name,digest in {**p['source_sha256'],**p['artifact_sha256'],**r['artifact_sha256']}.items():assert sha(ROOT/name)==digest,name
    w=load_decision_workflow(HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
        ROOT/r['policy_path'],RESULT,HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    personal=w.load_profile(HERE/'song_integrated_decision_v1/personal_S03.zip',user_id='Song')
    before=pickle.dumps((w.bank,w.gate,personal))
    long,_,_,_=native('S03',p);current,_,_,_=native('S04',p)
    lp,ls,_,_=prototypes(w,long,r['personal_calibration_ids'])
    np.testing.assert_allclose(personal.spd_prototypes,ls,rtol=0,atol=1e-12)
    for name in lp:np.testing.assert_allclose(personal.base.anchors[name].prototypes_,lp[name],rtol=0,atol=1e-12)
    eb,ei,eo=windows(current,r['evaluation_ids']);evcov=covariance_oracle(eb,ei)
    readout=w.bank.predict_providers(eb,ei,window_offsets=eo,user_id='Song');source=readout['probabilities']
    with (HERE/'song_integrated_decision_v1/predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    with (HERE/'song_personal_session_v1/predictions.csv').open(encoding='utf8',newline='') as f:old=list(csv.DictReader(f))
    arrays=np.load(HERE/'song_integrated_decision_v1/readouts.npz',allow_pickle=False)
    maximum=0.;checked=0
    for shots in (0,1,2,5):
        session=None;local=lp;localspd=ls;blended=lp;blendspd=ls;multipliers=np.ones(len(lp))
        if shots:
            session=w.load_profile(ROOT/r['profiles'][str(shots)]['path'],user_id='Song',session_id='S04',personal=personal)
            local,localspd,_,_=prototypes(w,current,r['profiles'][str(shots)]['calibration_ids'])
            beta=5/(5+shots);blended={name:beta*lp[name]+(1-beta)*local[name] for name in lp};blendspd=beta*ls+(1-beta)*localspd
            np.testing.assert_allclose(session.spd_local,localspd,rtol=0,atol=1e-12)
            np.testing.assert_allclose(session.spd_blended,blendspd,rtol=0,atol=1e-12)
            for name in lp:
                np.testing.assert_allclose(session.base.anchors[name]['local'].prototypes_,local[name],rtol=0,atol=1e-12)
                np.testing.assert_allclose(session.base.anchors[name]['blended'].prototypes_,blended[name],rtol=0,atol=1e-12)
            multipliers=routing_oracle(lp,local,ls,localspd,personal,session)
        saved=pickle.dumps(session)
        baseweights=np.array(personal.base.fusion_state.fusion_state.weights if session is None else session.base.fusion_state.weights)
        for cell in [c for c in r['cells'] if c['shots']==shots]:
            arm=cell['arm'];active=tuple(name for name in lp if arm!='full_minus_'+name)
            weights=baseweights.copy();anchor_mode='long' if arm=='F7_long' else 'local' if arm=='F7_local' else 'blend'
            anchor_protos=lp if anchor_mode=='long' else local if anchor_mode=='local' else blended
            spd_protos=ls if anchor_mode=='long' else localspd if anchor_mode=='local' else blendspd
            anchors={}
            for name in active:
                anchors[name],d=probability(evcov if name=='F2ac' else readout['source_standardized_features'][name],spd_protos if name=='F2ac' else anchor_protos[name],name=='F2ac')
                if arm=='F7_F8':
                    np.testing.assert_allclose(d,arrays[f'shots{shots}_{name}_anchor_distances'],rtol=0,atol=1e-10)
                    np.testing.assert_allclose(anchors[name],arrays[f'shots{shots}_{name}_anchor_probabilities'],rtol=0,atol=1e-12)
            if arm not in ('baseline','F7_long','F7_local','F7_blended','F7_standalone') and shots:weights*=multipliers
            weights=np.array([weights[list(lp).index(name)] for name in active]);weights/=weights.sum()
            heads=source if arm in ('baseline','F8_only') else anchors if arm=='F7_standalone' else {name:.5*source[name]+.5*anchors[name] for name in active}
            q=sum(weights[i]*heads[name] for i,name in enumerate(active));q/=q.sum(1,keepdims=True)
            selected=[v for v in rows if int(v['shots'])==shots and v['arm']==arm]
            assert [v['trial_id'] for v in selected]==r['evaluation_ids']
            reference=np.array([[float(v['p_'+c]) for c in w.bank.classes_] for v in selected])
            maximum=max(maximum,float(abs(q-reference).max()));checked+=len(selected)
            np.testing.assert_allclose(q,reference,rtol=0,atol=1e-12)
            np.testing.assert_allclose(q,arrays[f'shots{shots}_{arm}_probabilities'],rtol=0,atol=1e-12)
            if arm=='baseline':
                previous=[v for v in old if int(v['shots'])==shots and v['arm']=='session']
                expected=np.array([[float(v['p_'+c]) for c in w.bank.classes_] for v in previous])
                np.testing.assert_allclose(q,expected,rtol=0,atol=1e-12)
        assert pickle.dumps(session)==saved
        print(f'{shots}shot: independent prototypes, generalized SPD eigenvalues, F8 weights and14 arms verified',flush=True)
    assert checked==6944 and maximum<1e-12 and pickle.dumps((w.bank,w.gate,personal))==before
    sources=[Path(__file__),PROTOCOL,RESULT,ROOT/'tests/test_integrated_decision_delivery_v1.py',ROOT/'tests/test_integrated_decision_cli_v1.py']
    out=dict(schema='song_integrated_decision_acceptance_v1',source_sha256={v.relative_to(ROOT).as_posix():sha(v) for v in sources},
        artifact_sha256=r['artifact_sha256'],checked_cells=56,checked_trial_probabilities=checked,maximum_probability_error=maximum,
        independent_covariance_prototype_oracle=True,independent_generalized_eigenvalue_oracle=True,independent_session_routing_oracle=True,
        previous_baseline_parity=True,source_and_profiles_immutable=True,primary_pass=r['primary_pass'],default_promoted=False,
        physical_validation_proven=False,completion_proven=False,scope=p['scope'])
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':run()
