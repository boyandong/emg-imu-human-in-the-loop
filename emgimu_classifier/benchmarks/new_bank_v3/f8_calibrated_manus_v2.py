"""Source-user OOF temperature calibration, applied to frozen F8 evaluation blocks."""
import argparse
from dataclasses import fields
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from benchmarks.new_bank_v3.f8_router_manus_v1 import extract, sha
from emgimu.datasets.semg_manus import ManusWindows, load_semg_manus_windows
from emgimu.feature_bank.manus_study import GESTURES, _metrics
from emgimu.feature_bank.families import LocalDetailFamily
from emgimu.feature_bank.document_scale_v3 import DocumentScalePatternV3
from emgimu.feature_bank.spec_spatial_v3 import SpecSpdTangentV3
from emgimu.feature_bank.relative_spectrum import LogBandEnergyFamily
from emgimu.feature_bank.force_nested_oof import temperature_probability
from emgimu.feature_bank.session_fusion_v2 import PROVIDERS, fuse_disjoint_session_predictions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE/'F8_CALIBRATED_MANUS_V2_PROTOCOL.json'
PARENT = HERE/'F8_ROUTER_MANUS_V1_RESULTS.json'


def prepare():
    old = json.loads((HERE/'F8_ROUTER_MANUS_V1_PROTOCOL.json').read_text())
    p = {'version':'f8-calibrated-manus-v2', 'archive':old['archive'], 'archive_sha256':old['archive_sha256'],
         'parent_sha256':sha(PARENT), 'users':old['users'], 'providers':list(PROVIDERS),
         'temperature_candidates':[.25,.5,.75,1.,1.25,1.5,2.,3.,4.],
         'source':'Session1 only; leave one source user out. Feature family and scaler/classifier fit only other source users in each fold. Equal trial-mean coordinates.',
         'selection':'Minimum pooled source-OOF log loss per provider; ties prefer nearest log-temperature to zero, then smaller temperature. No target calibration/evaluation labels for temperature selection.',
         'readout':'Apply temperatures to saved full-source provider probabilities. Same24 target calibration/evaluation blocks and same frozen F8 weights; checked identity and class-axis fusion.',
         'classifier_C':old['classifier_C'], 'max_iter':old['max_iter'], 'seed':old['seed'],
         'scope':'Previously inspected MANUS eight-channel sessions, retrospective versioned calibrated routing comparison. TD24 is not Rest-fitted document F0. No own-device or default-promotion claim.'}
    if PROTOCOL.exists():
        raise FileExistsError('Refuse protocol replacement')
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n',encoding='utf8')


def take(data, mask):
    idx = np.flatnonzero(mask)
    return ManusWindows(**{f.name:data.batch.take(idx) if f.name == 'batch' else getattr(data,f.name)[idx]
                           for f in fields(ManusWindows)})


def run():
    p = json.loads(PROTOCOL.read_text()); parent = json.loads(PARENT.read_text())
    if sha(PARENT) != p['parent_sha256'] or sha(Path(p['archive'])) != p['archive_sha256']:
        raise ValueError('Frozen parent or archive changed')
    source = load_semg_manus_windows(p['archive'],users=p['users'],sessions=(1,),gestures=GESTURES)
    factories = {'TD24':LocalDetailFamily, 'pattern':DocumentScalePatternV3,
                 'SPD':SpecSpdTangentV3, 'log_bands':LogBandEnergyFamily}
    folds = []; oof = []
    for held in p['users']:
        train = take(source,source.users != held); validation = take(source,source.users == held)
        train_ids = sorted(set(train.trials)); held_ids = sorted(set(validation.trials))
        if set(train_ids) & set(held_ids):
            raise ValueError('Source OOF trial overlap')
        families = {name:factory().fit(train.batch) for name,factory in factories.items()}
        x,y,_,_ = extract(families,train); vx,vy,_,vids = extract(families,validation)
        probabilities = {}
        for name in PROVIDERS:
            model = make_pipeline(StandardScaler(),LogisticRegression(C=p['classifier_C'],class_weight='balanced',
                max_iter=p['max_iter'],random_state=p['seed'])).fit(x[name],y)
            if not np.array_equal(model[-1].classes_,np.arange(6)) or np.max(model[-1].n_iter_) >= p['max_iter']:
                raise ValueError('Incomplete class axis or nonconverged source model')
            before = pickle.dumps((families[name],model))
            probabilities[name] = model.predict_proba(vx[name])
            if before != pickle.dumps((families[name],model)):
                raise ValueError('Held-source prediction changed fitted state')
        for i,trial in enumerate(vids):
            oof.append({'trial':trial,'user':held,'label':int(vy[i]),
                        'raw_probability':{n:probabilities[n][i].tolist() for n in PROVIDERS}})
        folds.append({'held_user':held,'train_users':sorted(set(train.users.tolist())),
                      'train_trials':train_ids,'held_trials':held_ids})
        print(f'Source user{held} OOF fold complete',flush=True)
    target_ids = {t for b in parent['blocks'] for t in b['calibration_trials']+b['evaluation_trials']}
    if len({r['trial'] for r in oof}) != len(oof) or {r['trial'] for r in oof} != set(source.trials) or set(source.trials) & target_ids:
        raise ValueError('OOF coverage or source/target separation failed')
    temperatures = {}; selection = {}; labels = np.array([r['label'] for r in oof])
    for name in PROVIDERS:
        raw = np.array([r['raw_probability'][name] for r in oof]); trials = []
        for temperature in p['temperature_candidates']:
            probability = temperature_probability(raw,temperature)
            loss = float(-np.log(np.maximum(probability[np.arange(len(labels)),labels],1e-15)).mean())
            trials.append({'temperature':temperature,'log_loss':loss})
        best = min(trials,key=lambda r:(r['log_loss'],abs(np.log(r['temperature'])),r['temperature']))
        temperatures[name] = best['temperature']; selection[name] = trials
    blocks = []; scores = {}
    for b in parent['blocks']:
        experts = np.stack([temperature_probability(np.array(b['providers'][i]),temperatures[name])
                            for i,name in enumerate(PROVIDERS)])
        routed = fuse_disjoint_session_predictions(experts,b['weights'], source_trials=b['source_trials'],
            calibration_trials=b['calibration_trials'],evaluation_trials=b['evaluation_trials'],
            provider_names=PROVIDERS,provider_classes=[tuple(range(6))]*4,classes=tuple(range(6)))
        blocks.append({**b,'uncalibrated_probabilities':b['probabilities'], 'providers':experts.tolist(),
                       'probabilities':{'TD24':experts[0].tolist(),'uniform':experts.mean(axis=0).tolist(),'F8':routed.tolist()}})
    for phase in ('validation','descriptive_final'):
        for shots in (1,2):
            group = [b for b in blocks if b['phase'] == phase and b['shots'] == shots]
            y = np.concatenate([b['labels'] for b in group])
            scores[f'{phase}_{shots}shot'] = {name:_metrics(y,np.concatenate([b['probabilities'][name] for b in group]))
                                             for name in ('TD24','uniform','F8')}
    paths = [Path(__file__),ROOT/'src/emgimu/feature_bank/force_nested_oof.py',
             ROOT/'src/emgimu/feature_bank/session_fusion_v2.py',ROOT/'benchmarks/new_bank_v3/f8_router_manus_v1.py']
    result = {'protocol_sha256':sha(PROTOCOL),'parent_sha256':sha(PARENT),
        'source_hashes':{f.relative_to(ROOT).as_posix():sha(f) for f in paths},
        'source_folds':folds,'source_oof':oof,'temperature_selection':selection,'temperatures':temperatures,
        'blocks':blocks,'scores':scores,'scope':p['scope'],'default_promoted':False,'completion_proven':False}
    (HERE/'F8_CALIBRATED_MANUS_V2_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'temperatures':temperatures,'F8_losses':{k:v['F8']['log_loss'] for k,v in scores.items()}}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
