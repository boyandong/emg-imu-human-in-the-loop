from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/song_real8'


def test_full_recording_stream_grid_probabilities_confirmation_and_trial_mass():
    protocol=HERE/'SONG_F0_STREAM_V1_PROTOCOL.json';p=json.loads(protocol.read_text(encoding='utf8'))
    result=json.loads((HERE/'SONG_F0_STREAM_V1_RESULTS.json').read_text(encoding='utf8'))
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==result['protocol_sha256']
    for name,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/p['package_path']).read_bytes()).hexdigest()==p['package_sha256']
    array_path=HERE/'SONG_F0_STREAM_V1_EMISSIONS.npz'
    assert hashlib.sha256(array_path.read_bytes()).hexdigest()==result['emissions_sha256']
    assert result['source_state_immutable'] and len(result['records'])==4
    assert not any(result[k] for k in ['default_promoted','physical_validation_proven','completion_proven'])
    with np.load(array_path,allow_pickle=False) as a:
        for row in result['records']:
            key=row['session']+'_'+row['arm'];ends=a[row['session']+'_ends'];truth=a[row['session']+'_truth'];trials=a[row['session']+'_trials']
            np.testing.assert_array_equal(ends,np.arange(49,row['raw_samples'],10))
            q=a[key+'_probabilities'];raw=a[key+'_raw'];confirmed=a[key+'_confirmed']
            assert q.shape==(len(ends),4) and row['emissions']==len(ends)
            np.testing.assert_allclose(q.sum(1),1.,atol=1e-12)
            np.testing.assert_array_equal(raw,q.argmax(1))
            stable=-1;candidate=None;count=0;expected=[]
            for label in raw:
                if label==stable:candidate=None;count=0
                else:
                    count=count+1 if label==candidate else 1;candidate=label
                    if count>=p['confirmations']:stable=int(label);candidate=None;count=0
                expected.append(stable)
            np.testing.assert_array_equal(confirmed,expected)
            assert row['one_pass_max_probability_error']<1e-12 and row['unscored_emissions']==int(np.sum(truth<0))
            mask=truth>=0;unique=np.unique(trials[mask]);counts=Counter(trials[mask]);weights=np.array([1/counts[t] for t in trials[mask]])
            for name,pred in [('raw',raw),('confirmed',confirmed)]:
                y=truth[mask];v=pred[mask];m=row[name];cm=np.zeros((4,5))
                for t,c,w in zip(y,v,weights):cm[t,4 if c<0 else c]+=w
                np.testing.assert_allclose(cm,m['trial_balanced_confusion'],atol=1e-12)
                den=cm.sum(1)+cm[:,:4].sum(0);f=np.divide(2*np.diag(cm[:,:4]),den,out=np.zeros(4),where=den>0)
                assert abs(m['trial_balanced_macro_f1']-f.mean())<1e-12
                assert abs(m['trial_balanced_accuracy']-np.sum(weights*(y==v))/len(unique))<1e-12
                assert m['eligible_trials']==len(unique) and m['eligible_windows']==int(np.sum(mask))
                holds={int(t):pred[(trials==t)&mask] for t in unique}
                correct=sum(bool(np.all(v==truth[(trials==t)&mask])) for t,v in holds.items())
                assert m['whole_stable_hold_correct']==correct
                assert m['within_stable_trial_switches']==sum(int(np.sum(np.diff(v)!=0)) for v in holds.values())
