import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.grabmyo_crossday import run as grab


def test_f1_document_native_increment_is_trial_matched_and_source_frozen():
    root=Path(__file__).resolve().parents[1];here=root/'benchmarks/new_bank_v3'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=here/'F1_SCALE_GRAB_PROTOCOL.json'
    result=json.loads((here/'F1_SCALE_GRAB_RESULTS.json').read_text())
    path=here/'F1_SCALE_GRAB_PREDICTIONS.csv'
    assert result['protocol_sha256']==sha(protocol)
    assert result['prediction_sha256']==sha(path)
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    assert result['f0v2_parent_replay_max_abs_error']==0.
    assert result['feature_dimensions']=={'F0v2':48,'F1scale':8}
    assert result['scores']['F0v2+F1scale']['dimension']==112
    source=set(result['source_trial_ids']);val=set(result['validation_trial_ids']);final=set(result['final_trial_ids'])
    assert (len(source),len(val),len(final))==(112,56,56)
    assert not(source&val or source&final or val&final)
    with path.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==224
    classes=np.asarray(grab.GESTURES)
    for phase in ('validation','final'):
        trial_sets=[]
        for arm in ('F0v2','F0v2+F1scale'):
            selected=[r for r in rows if r['phase']==phase and r['arm']==arm]
            trial_sets.append({r['trial_id'] for r in selected})
            assert len(selected)==len(trial_sets[-1])==56
            p=np.array([[float(r[f'p_{c}']) for c in classes] for r in selected])
            score=grab.score(np.array([int(r['gesture']) for r in selected]),p,classes,
                             np.array([int(r['subject']) for r in selected]))
            np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12)
            for metric in ('macro_f1','log_loss','brier','minimum_subject_macro_f1'):
                np.testing.assert_allclose(score[metric],result['scores'][arm][phase][metric],atol=1e-12)
        assert trial_sets[0]==trial_sets[1]
        base=result['scores']['F0v2'][phase];addition=result['scores']['F0v2+F1scale'][phase]
        assert addition['macro_f1']<base['macro_f1'] and addition['log_loss']>base['log_loss']
