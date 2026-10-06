import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def test_f5_complete_native_replay_is_trial_separated_and_reproducible():
    root=Path(__file__).resolve().parents[1];here=root/'benchmarks/new_bank_v3'
    result=json.loads((here/'F5_PATH_UNIBO_RESULTS.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['protocol_sha256']==sha(here/'F5_PATH_UNIBO_PROTOCOL.json')
    assert result['row_sha256']==sha(here/'F5_PATH_UNIBO_ROWS.csv')
    for path,digest in result['source_hashes'].items():assert sha(root/path)==digest
    assert result['rows']==5091
    assert result['dtw_dimension']==4 and result['signature_dimension']==20
    assert not result['classifier_fit'] and not result['target_labels_used_for_fitting']
    source={r['user']:r for r in result['source_selection']}
    assert set(source)=={f'u{i:02}' for i in range(1,8)}
    for row in source.values():
        assert len(set(row['candidate_ids']))==20
        assert len(row['medoid_ids'])==4
        assert set(row['medoid_ids'])<=set(row['candidate_ids'])
        assert row['classes']==[0,1,2,3] and row['source_band']==3
    with (here/'F5_PATH_UNIBO_ROWS.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len({r['id'] for r in rows})==5091
    for row in rows:
        candidates=source[row['user']]['candidate_ids']
        assert row['trial'] not in {i.split(':',1)[0] for i in candidates}
        assert float(row['duration_seconds'])>=1.
        assert int(row['day']) in ({6} if row['phase']=='validation' else {7,8})
        distances=[float(row[f'dtw_{i}']) for i in range(4)]
        baseline=[float(row[f'legacy_dtw_{i}']) for i in range(4)]
        signature=[float(row[f'signature_{i}']) for i in range(20)]
        assert np.isfinite(distances+baseline+signature).all()
        assert min(distances)>=0.
        assert int(row['prediction'])==int(np.argmin(distances))
        assert int(row['legacy_prediction'])==int(np.argmin(baseline))
    for phase,total in (('validation',1700),('descriptive_final',3391)):
        selected=[r for r in rows if r['phase']==phase];reported=result['by_phase'][phase]
        assert len(selected)==total==reported['bouts']
        assert len({r['user'] for r in selected})==reported['users']==7
        assert sum(r['prediction']!=r['legacy_prediction'] for r in selected)==reported['changed_nearest_template_decisions']
        for prefix in ('new','legacy'):
            key='prediction' if prefix=='new' else 'legacy_prediction'
            assert np.mean([r[key]==r['label'] for r in selected])==reported[f'{prefix}_bout_accuracy']
    assert max(float(r['signature_legacy_max_abs_difference']) for r in rows)==result['max_signature_legacy_abs_difference']
