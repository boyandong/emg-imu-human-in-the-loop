"""Export both unchanged Song source fits to the data-free collection runtime."""
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT.parent
COLLECTION=REPO/'collection/emg_meta/emg_meta'
OUT=ROOT/'feature_bank/SONG_GUI_V2_ACCEPTANCE.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    source=ROOT/'feature_bank/models/song_f0_250hz_v1.pkl'
    acceptance=json.loads((ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json').read_text(encoding='utf8'))
    if sha(source)!=acceptance['package_sha256']:raise ValueError('Source checkpoint changed')
    runtime=pickle.loads(source.read_bytes());before=pickle.dumps(runtime)
    result_path=HERE/'F0_REST_MODEL_RESULTS.json';result=json.loads(result_path.read_text(encoding='utf8'))
    protocol_path=HERE/'SONG_F0_STREAM_V1_PROTOCOL.json';protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    stream_result=HERE/'SONG_F0_STREAM_V1_RESULTS.json';sr=json.loads(stream_result.read_text(encoding='utf8'))
    if sha(protocol_path)!=sr['protocol_sha256']:raise ValueError('Frozen stream protocol changed')
    reference_path=HERE/'SONG_F0_STREAM_V1_EMISSIONS.npz'
    if sha(reference_path)!=sr['emissions_sha256']:raise ValueError('Frozen stream emissions changed')
    with (HERE/'F0_REST_MODEL_TRIAL_PREDICTIONS.csv').open(encoding='utf8',newline='') as f:trial_rows=list(csv.DictReader(f))
    sys.path.insert(0,str(COLLECTION))
    from emgforce.inference.song_local import SongLocalRuntime,SongOnlineDecision,LABELS,discover_song_models
    bundles=[];records=[]
    for arm,(family,model) in runtime.models_.items():
        directory=COLLECTION/'model_assets'/('song_f0_'+('rest' if arm=='rest_only_threshold' else 'pooled')+'_frozen_v2')
        directory.mkdir(parents=True,exist_ok=True)
        artifact=directory/'song_f0_model.json';manifest_path=directory/'song_manifest.json'
        payload={'format_version':2,'model_kind':'song_real8_causal_f0_logistic','sample_rate_hz':250,'channels':8,
            'window_samples':50,'hop_samples':10,'classes':list(model[-1].classes_),
            'filter':{'highpass_hz':40.,'highpass_order':4,'notches_hz':[50.,100.],'notch_q':30.,
                      'implementation':'causal_sosfilt_zero_initial_state_continuous'},
            'f0_thresholds':family.thresholds_.tolist(),'standard_scaler_mean':model[0].mean_.tolist(),
            'standard_scaler_scale':model[0].scale_.tolist(),'logistic_coef':model[-1].coef_.tolist(),
            'logistic_intercept':model[-1].intercept_.tolist(),
            'stream_policy':{'consecutive_frames':2,'online_event_threshold':0.,
                'confirmation':'consecutive_argmax_initial_unknown','standard_scaler_dtype':'float32_inplace','probability_dtype':'float64'}}
        selected=[v for v in trial_rows if v['session']=='S03' and v['arm']==arm]
        native=list(runtime.classes_)
        accuracy=float(np.mean([v['hand']==native[np.argmax([float(v['p_'+c]) for c in native])] for v in selected]))
        manifest={'format_version':2,'algorithm_id':'song_real8_local_v1',
            'model_id':'Song250 frozen '+arm,'artifact':artifact.name,
            'model_status':'frozen_source_one_person_one_day_not_live_validated',
            'threshold_arm':arm,'validation_trial_accuracy':accuracy,
            'validation_trial_macro_f1':result['metrics']['S03'][arm]['macro_f1'],
            'source_checkpoint_sha256':sha(source),'source_original_state_sha256':acceptance['source_original_state_sha256'][arm],
            'stream_protocol_sha256':sha(protocol_path),'default_promoted':False,
            'scope':'Both fixed source arms exported without fitting; single user/day retrospective data, readiness failures, stable-cue and hardware limitations persist. Stream confirmation reduces switching but slightly harms F1; user selection is required.'}
        # Reverification may reuse identical exported bytes, never replace a different bundle.
        encoded=(json.dumps(payload,indent=2)+'\n').encode('utf8')
        if artifact.exists() and artifact.read_bytes()!=encoded:raise ValueError('Existing GUI model differs')
        artifact.write_bytes(encoded);manifest['sha256']=sha(artifact)
        encoded=(json.dumps(manifest,indent=2)+'\n').encode('utf8')
        if manifest_path.exists() and manifest_path.read_bytes()!=encoded:raise ValueError('Existing GUI manifest differs')
        manifest_path.write_bytes(encoded)
        bundles.append({'arm':arm,'directory':directory.relative_to(REPO).as_posix(),
            'model_sha256':sha(artifact),'manifest_sha256':sha(manifest_path)})
        exported=SongLocalRuntime(directory)
        for label,value in [('thresholds',family.thresholds_),('scaler_mean',model[0].mean_),('scaler_scale',model[0].scale_),('coef',model[-1].coef_),('intercept',model[-1].intercept_)]:
            if not np.array_equal(getattr(exported,label),value):raise ValueError('Exported source parameters differ')
    with np.load(reference_path,allow_pickle=False) as reference:
        for session in protocol['target_sessions']:
            path=Path(protocol['source_folder'])/f'2026-09-18_{session}'/'session.h5'
            if sha(path)!=protocol['hdf5_sha256'][session]:raise ValueError('Native HDF5 changed')
            with h5py.File(path) as f:raw=f['streams/emg/raw'][:]
            for entry in bundles:
                arm=entry['arm'];exported=SongLocalRuntime(REPO/entry['directory']);frames=[]
                for start in range(0,len(raw),4096):
                    block=raw[start:start+4096];gap,items=exported.ingest(block,np.arange(start,start+len(block)))
                    if gap:raise ValueError('False gap in contiguous recording')
                    frames+=items
                ends=np.array([v[0] for v in frames]);q=np.stack([v[1] for v in frames])
                cols=[LABELS.index(c) for c in runtime.classes_]
                error=float(np.max(abs(q[:,cols]-reference[session+'_'+arm+'_probabilities'])))
                if error>1e-12 or not np.array_equal(ends,reference[session+'_ends']):raise ValueError('GUI stream differs from frozen native replay')
                decision=SongOnlineDecision(exported.consecutive_frames)
                confirmed=[]
                for probability in q:
                    label,_=decision.step(probability,exported.online_event_threshold)
                    confirmed.append(-1 if label is None else runtime.classes_.index(label))
                if not np.array_equal(confirmed,reference[session+'_'+arm+'_confirmed']):raise ValueError('GUI confirmation differs')
                records.append({'session':session,'arm':arm,'emissions':len(frames),'maximum_probability_error':error,'confirmation_exact':True})
                print(f'GUI {session}/{arm}: source parameters and every stream frame verified',flush=True)
    if before!=pickle.dumps(runtime):raise ValueError('Export changed source runtime')
    discovered={v.directory.resolve() for v in discover_song_models(COLLECTION/'models',include_builtin=True)}
    if not all((REPO/v['directory']).resolve() in discovered for v in bundles):raise ValueError('Bundled GUI model discovery failed')
    sources=[Path(__file__),result_path,protocol_path,stream_result,reference_path,
        COLLECTION/'emgforce/inference/song_local.py',COLLECTION/'emgforce/inference/song_worker.py',
        COLLECTION/'emgforce/ui/realtime_inference_page.py',COLLECTION/'tests/test_song_frozen_v2.py']
    out={'schema':'song_gui_v2_acceptance','source_sha256':{v.relative_to(REPO).as_posix():sha(v) for v in sources},
        'source_checkpoint_sha256':sha(source),'bundles':bundles,'records':records,'source_parameters_exact':True,
        'source_state_immutable':True,'builtin_discovery_verified':True,'default_promoted':False,
        'physical_validation_proven':False,'completion_proven':False,
        'scope':'Two source-frozen JSON models load in the existing collection page without classifier Python package or source recordings. All126800 GUI runtime stream frames match frozen replay below1e-12; two-confirmation labels agree exactly. Qt worker/page simulated-input tests are separate from physical device efficacy. Legacy v1 rules remain unchanged.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':run()
