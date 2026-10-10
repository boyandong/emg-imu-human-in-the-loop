"""No-refit independent CSP, F4d, prototype/routing and all76 native-cell checks."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.linalg import eigvalsh
from benchmarks.song_real8.song_extended_window_v1 import load_extended
from benchmarks.song_real8.song_personal_session_workflow_v1 import aggregate, windows, score
from benchmarks.song_real8.song_raw_quality_v1 import native
from benchmarks.song_real8.verify_integrated_decision_v1 import covariance_oracle, probability, routing_oracle
from benchmarks.song_real8.verify_song_raw_quality_v1 import channel_masks
from emgimu.feature_bank.personal_session_cli_v1 import load_workflow

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'SONG_EXTENDED_WINDOW_V1_PROTOCOL.json'
RESULT = HERE/'SONG_EXTENDED_WINDOW_V1_RESULTS.json'
OUT = ROOT/'feature_bank/SONG_EXTENDED_WINDOW_ACCEPTANCE_V1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectral(w, batch):
    x = batch.emg.astype(float)
    x = (x-x.mean(1, keepdims=True))*np.hanning(x.shape[1])[None, :, None]
    harmonics = np.exp(-2j*np.pi*np.outer(np.arange(26), np.arange(50))/50)
    power = abs(np.einsum('kt,ntc->nkc', harmonics, x))**2/50
    frequency = np.arange(26)*5.
    values = []
    for i, (low, high) in enumerate(w.bands):
        mask = (frequency >= low) & (frequency <= high if i == len(w.bands)-1 else frequency < high)
        values.append(np.log(power[:, mask].sum(1)+1e-10))
    return np.concatenate(values, 1).astype(np.float32)


def prototypes(w, item, ids, profile):
    b, wi, o = windows(item, ids)
    readout = w.bank.predict_providers(b, wi, window_offsets=o, user_id='Song')
    labels = dict(zip(item['trial'], item['hand']))
    y = np.array([labels[t] for t in readout['trial_ids']])
    means = {name:np.stack([x.astype(float)[y == c].mean(0) for c in w.bank.classes_])
        for name,x in readout['source_standardized_features'].items()}
    cov = covariance_oracle(b,wi)
    matrices = np.stack([cov[y == c].mean(0) for c in w.bank.classes_])
    log_bands = spectral(w,b).astype(float)
    reference = np.stack([log_bands[wi == t].mean(0) for t in np.unique(wi)]).mean(0)
    np.testing.assert_allclose(profile.spectral_reference,reference,rtol=0,atol=2e-6)
    return means,matrices


def run():
    p = json.loads(PROTOCOL.read_text(encoding='utf8'))
    r = json.loads(RESULT.read_text(encoding='utf8'))
    assert sha(PROTOCOL) == r['protocol_sha256']
    for name,digest in {**p['source_sha256'],**p['artifact_sha256'],**r['artifact_sha256']}.items():
        assert sha(ROOT/name) == digest,name
    w = load_extended(ROOT/r['source_bank_path'],ROOT/r['policy_path'],
        HERE/'song_raw_quality_v1/source_gate.pkl',HERE/'SONG_RAW_QUALITY_V1_RESULTS.json')
    before = pickle.dumps(w)
    old = load_workflow(HERE/'song_personal_session_v1/source_bank.pkl',ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')
    for g in old.bank.providers_:
        assert pickle.dumps((old.bank.families_[g],old.bank.models_[g],old.bank.temperatures_[g])) == pickle.dumps((w.bank.families_[g],w.bank.models_[g],w.bank.temperatures_[g]))
    source = [native(s,p)[0] for s in ('S01','S02')]
    x = np.concatenate([v['batch'].emg for v in source]).astype(float)
    wi = np.concatenate([v['trial'] for v in source])
    wy = np.concatenate([v['hand'] for v in source])
    family = w.bank.families_['F2b'][0]
    assert set(family.source_trial_ids_) == set(wi) == set(w.bank.policy_.source_trials)
    moment = np.stack([v.T@v/(np.trace(v.T@v)+1e-10) for v in x])
    max_eigen = 0.
    for i,c in enumerate(family.classes_):
        positive = moment[wy == c].mean(0)
        total = positive+moment[wy != c].mean(0)+1e-5*np.eye(8)
        eigen = eigvalsh(positive,total)
        vectors,values = family.filters_[:,i*4:i*4+4],family.eigenvalues_[i]
        error = float(np.max(abs(positive@vectors-(total@vectors)*values)))
        max_eigen = max(max_eigen,error)
        assert error < 1e-10
        np.testing.assert_allclose(values,np.r_[eigen[-2:],eigen[:2]],rtol=0,atol=1e-10)
    variance = np.stack([np.var(v@family.filters_,axis=0) for v in x])
    csp = np.log(variance/(variance.sum(1,keepdims=True)+1e-10)+1e-10).astype(np.float32)
    np.testing.assert_allclose(family.transform(source[0]['batch']),csp[:len(source[0]['trial'])],rtol=0,atol=2e-6)
    source_features = np.stack([csp[wi == t].mean(0) for t in np.unique(wi)])
    scaler = w.bank.models_['F2b'][0]
    np.testing.assert_allclose(scaler.mean_,source_features.astype(float).mean(0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(scaler.var_,source_features.astype(float).var(0),rtol=0,atol=1e-12)
    assert all(not set(f['fit_ids']) & set(f['ids']) for f in r['source_oof'])
    for f in r['source_oof']:
        logits = np.log(np.maximum(f['raw'],1e-15))/r['source_temperature']
        q = np.exp(logits-logits.max(1,keepdims=True));q /= q.sum(1,keepdims=True)
        np.testing.assert_array_equal(q,np.asarray(f['calibrated']))
    losses = np.array([r['source_oof_losses'][g] for g in w.bank.providers_])
    population = np.exp(-losses+losses.min());population /= population.sum()
    np.testing.assert_allclose(w.bank.policy_.population,population,rtol=0,atol=1e-15)
    long,_,_,_ = native('S03',p)
    current,_,_,raw = native('S04',p)
    personal = w.load_profile(HERE/'song_extended_window_v1/personal_S03.zip',user_id='Song')
    long_means,long_cov = prototypes(w,long,r['personal_calibration_ids'],personal)
    for g in w.bank.providers_:
        np.testing.assert_allclose(personal.decision.base.anchors[g].prototypes_,long_means[g],rtol=0,atol=1e-12)
    np.testing.assert_allclose(personal.decision.spd_prototypes,long_cov,rtol=0,atol=1e-12)
    eb,ei,eo = windows(current,r['evaluation_ids'])
    mask = np.isin(current['trial'],r['evaluation_ids'])
    readout = w.bank.predict_providers(eb,ei,window_offsets=eo,user_id='Song')
    matrices = covariance_oracle(eb,ei)
    _,channels,bad = channel_masks(raw.emg[mask],ei,w.gate)
    context = spectral(w,eb).astype(float)-personal.spectral_reference
    expected_context = np.stack([context[ei == t].mean(0) for t in np.unique(ei)])
    maximum,checked = 0.,0
    old_readouts = HERE/'song_integrated_decision_v1/readouts.npz'
    with np.load(HERE/'song_extended_window_v1/readouts.npz',allow_pickle=False) as saved, np.load(old_readouts,allow_pickle=False) as original:
        for shots in p['current_shots']:
            session = None
            local,local_cov = long_means,long_cov
            if shots:
                session = w.load_profile(ROOT/r['profiles'][str(shots)]['path'],user_id='Song',session_id='S04',personal=personal)
                local,local_cov = prototypes(w,current,r['profiles'][str(shots)]['calibration_ids'],session)
                count = np.array([sum(dict(zip(current['trial'],current['hand']))[t] == c for t in r['profiles'][str(shots)]['calibration_ids']) for c in w.bank.classes_])
                beta = np.array(personal.base.calibration_counts)/(np.array(personal.base.calibration_counts)+count)
                for g in w.bank.providers_:
                    np.testing.assert_allclose(session.decision.base.anchors[g]['local'].prototypes_,local[g],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(session.decision.base.anchors[g]['blended'].prototypes_,beta[:,None]*long_means[g]+(1-beta[:,None])*local[g],rtol=0,atol=1e-12)
                np.testing.assert_allclose(session.decision.spd_blended,beta[:,None,None]*long_cov+(1-beta[:,None,None])*local_cov,rtol=0,atol=1e-12)
                routing = routing_oracle(long_means,local,long_cov,local_cov,personal.decision,session.decision)
                np.testing.assert_allclose(saved[f'shots{shots}_spectral_session_minus_long'],session.spectral_reference-personal.spectral_reference,rtol=0,atol=1e-12)
            else:
                beta = np.ones(4);routing = np.ones(7)
            provider_anchors = {}
            for g in w.bank.providers_:
                prototype = beta[:,None]*long_means[g]+(1-beta[:,None])*local[g]
                if g == 'F2ac':
                    cov = beta[:,None,None]*long_cov+(1-beta[:,None,None])*local_cov
                    provider_anchors[g],_ = probability(matrices,cov,spd=True)
                else:
                    provider_anchors[g],_ = probability(readout['source_standardized_features'][g],prototype)
            baseweights = np.array(personal.base.fusion_state.fusion_state.weights if session is None else session.base.fusion_state.weights)
            np.testing.assert_allclose(saved[f'shots{shots}_spectral_window_minus_long'],context,rtol=0,atol=2e-6)
            np.testing.assert_allclose(saved[f'shots{shots}_spectral_trial_minus_long'],expected_context,rtol=0,atol=2e-6)
            for cell in [c for c in r['cells'] if c['shots'] == shots]:
                name = cell['arm']
                if name.startswith('old_six'):
                    key = 'baseline' if name == 'old_six_baseline' else 'F7_F8'
                    expected = original[f'shots{shots}_{key}_probabilities']
                elif name == 'CSP_only':
                    expected = readout['probabilities']['F2b']
                elif name == 'seven_uniform':
                    expected = np.mean(list(readout['probabilities'].values()),axis=0)
                else:
                    active = tuple(g for g in w.bank.providers_ if not name.startswith('full_minus_') or g != name[len('full_minus_'):])
                    indices = [w.bank.providers_.index(g) for g in active]
                    weights = np.array(w.bank.policy_.population if name == 'seven_population' else baseweights)[indices]
                    anchor = name not in ('seven_reliability','seven_F8','seven_minus_F7','seven_population')
                    router = session is not None and name not in ('seven_reliability','seven_minus_F8','seven_population')
                    if router:
                        weights *= routing[indices]
                    weights /= weights.sum()
                    q = np.stack([(.5*readout['probabilities'][g]+.5*provider_anchors[g]) if anchor else readout['probabilities'][g] for g in active],axis=1)
                    structural = name == 'seven_full_structural' or name in ('seven_minus_F7','seven_minus_F8') or name.startswith('full_minus_')
                    effective = np.tile(weights,(len(q),1))
                    if structural:
                        effective *= (~bad.any(1))[:,None]
                    rejected = effective.sum(1) <= 1e-10
                    effective[rejected] = weights
                    effective /= effective.sum(1,keepdims=True)
                    expected = np.einsum('ng,ngc->nc',effective,q)
                    expected /= expected.sum(1,keepdims=True)
                actual = saved[f'shots{shots}_{name}_probabilities']
                error = float(abs(actual-expected).max());maximum=max(maximum,error)
                assert error < 1e-12,(shots,name,error)
                metric = score(np.asarray(r['evaluation_labels']),actual)
                for k in ('macro_f1','accuracy','log_loss','brier'):
                    assert abs(metric[k]-cell[k]) < 1e-12
                checked += len(actual)
            print(f'{shots}shot: CSP/F4d, calibrated prototypes, routing and19 arms independently verified',flush=True)
    assert checked == 9424 and pickle.dumps(w) == before
    output = dict(schema='song_extended_window_acceptance_v1',protocol_sha256=sha(PROTOCOL),result_sha256=sha(RESULT),
        verifier_sha256=sha(Path(__file__)),artifact_sha256=r['artifact_sha256'],checked_cells=len(r['cells']),
        checked_trial_probabilities=checked,maximum_probability_error=maximum,maximum_CSP_eigen_residual=max_eigen,
        independent_source_CSP_eigen_and_variance=True,independent_direct_DFT_context=True,
        independent_trial_balanced_prototypes_and_routing=True,previous_six_baseline_parity=True,
        source_and_profiles_immutable=True,primary_pass=r['primary_pass'],F4d_context_only=True,
        default_promoted=False,physical_validation_proven=False,completion_proven=False,scope=r['scope'])
    OUT.write_text(json.dumps(output,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__ == '__main__':
    run()
