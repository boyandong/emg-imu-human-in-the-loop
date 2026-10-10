"""Verify GUI package provenance, source parameters and native replay coverage."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gui_bundles_bind_source_models_and_full_native_replay():
    path=ROOT/'feature_bank/SONG_GUI_V2_ACCEPTANCE.json'
    acceptance=json.loads(path.read_text(encoding='utf8'))
    for name,digest in acceptance['source_sha256'].items():assert sha(REPO/name)==digest
    source=ROOT/'feature_bank/models/song_f0_250hz_v1.pkl'
    assert sha(source)==acceptance['source_checkpoint_sha256']
    runtime=pickle.loads(source.read_bytes())
    assert acceptance['source_parameters_exact'] and acceptance['source_state_immutable']
    assert acceptance['builtin_discovery_verified']
    assert not any(acceptance[k] for k in ('default_promoted','physical_validation_proven','completion_proven'))
    assert {r['arm'] for r in acceptance['bundles']}==set(runtime.models_)
    for entry in acceptance['bundles']:
        directory=REPO/entry['directory'];artifact=directory/'song_f0_model.json';manifest=directory/'song_manifest.json'
        assert sha(artifact)==entry['model_sha256'] and sha(manifest)==entry['manifest_sha256']
        m=json.loads(manifest.read_text(encoding='utf8'));v=json.loads(artifact.read_text(encoding='utf8'))
        assert m['sha256']==sha(artifact) and m['source_checkpoint_sha256']==sha(source)
        assert m['threshold_arm']==entry['arm'] and not m['default_promoted']
        family,model=runtime.models_[entry['arm']]
        assert v['classes']==list(model[-1].classes_)
        for name,expected in [('f0_thresholds',family.thresholds_),('standard_scaler_mean',model[0].mean_),
                              ('standard_scaler_scale',model[0].scale_),('logistic_coef',model[-1].coef_),
                              ('logistic_intercept',model[-1].intercept_)]:
            np.testing.assert_array_equal(v[name],expected)
        assert (v['channels'],v['sample_rate_hz'],v['window_samples'],v['hop_samples'])==(8,250,50,10)
        assert v['stream_policy']['consecutive_frames']==2 and v['stream_policy']['online_event_threshold']==0.
    records=acceptance['records'];assert len(records)==4 and sum(r['emissions'] for r in records)==126800
    frozen=json.loads((ROOT/'benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json').read_text(encoding='utf8'))
    expected={(r['session'],r['arm']):r['emissions'] for r in frozen['records']}
    assert {(r['session'],r['arm']):r['emissions'] for r in records}==expected
    assert all(r['confirmation_exact'] and r['maximum_probability_error']<1e-12 for r in records)
    index=json.loads((ROOT/'feature_bank/delivery/INDEX.json').read_text(encoding='utf8'))
    pointer=index['song_gui_v2_acceptance']
    assert (ROOT/'feature_bank/delivery'/pointer['path']).resolve()==path
    assert pointer['sha256']==sha(path)
