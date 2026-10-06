import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.grabmyo_crossday import run as grab


def test_document_spectral_native_trial_pairing_and_frozen_readback():
    root=Path(__file__).resolve().parents[1];here=root/'benchmarks/new_bank_v3'
    protocol=here/'F4_SPECTRAL_GRAB_PROTOCOL.json'
    result=json.loads((here/'F4_SPECTRAL_GRAB_RESULTS.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert result['protocol_sha256']==sha(protocol)
    predictions=here/'F4_SPECTRAL_GRAB_PREDICTIONS.csv'
    assert result['prediction_sha256']==sha(predictions)
    for path,digest in result['source_hashes'].items():assert sha(root/path)==digest
    assert result['f0v2_parent_replay_max_abs_error']==0.
    assert result['feature_dimensions']=={'F0v2':48,'F4spectral':72}
    source=set(result['source_trial_ids']);validation=set(result['validation_trial_ids']);final=set(result['final_trial_ids'])
    assert (len(source),len(validation),len(final))==(112,56,56)
    assert not(source&validation or source&final or validation&final)
    with predictions.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==224
    classes=np.asarray(grab.GESTURES)
    for phase in ('validation','final'):
        ids=[]
        for arm in ('F0v2','F0v2+F4spectral'):
            selected=[r for r in rows if r['phase']==phase and r['arm']==arm]
            ids.append({r['trial_id'] for r in selected})
            assert len(selected)==len(ids[-1])==56
            p=np.asarray([[float(r[f'p_{c}']) for c in classes] for r in selected])
            score=grab.score(np.asarray([int(r['gesture']) for r in selected]),p,classes,
                             np.asarray([int(r['subject']) for r in selected]))
            np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12)
            for metric in ('macro_f1','log_loss','brier','minimum_subject_macro_f1'):
                np.testing.assert_allclose(score[metric],result['scores'][arm][phase][metric],atol=1e-12)
        assert ids[0]==ids[1]
    # Observed validation benefit reverses in the inspected descriptive final.
    assert result['scores']['F0v2+F4spectral']['final']['macro_f1']<result['scores']['F0v2']['final']['macro_f1']
    assert result['scores']['F0v2+F4spectral']['final']['log_loss']>result['scores']['F0v2']['final']['log_loss']
