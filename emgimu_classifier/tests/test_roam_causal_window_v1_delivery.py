import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/new_bank_v3'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(runs,size):
    out=np.full(size,-999,dtype=int);previous=0
    for a,b,label in runs:
        assert a==previous and a<b<=size
        out[a:b]=label;previous=b
    assert previous==size
    return out


def test_native_provenance_full_sample_coverage_and_independent_metrics():
    p=json.loads((HERE/'ROAM_CAUSAL_WINDOW_V1_PROTOCOL.json').read_text(encoding='utf8'))
    r=json.loads((HERE/'ROAM_CAUSAL_WINDOW_V1_RESULTS.json').read_text(encoding='utf8'))
    assert sha(HERE/'ROAM_CAUSAL_WINDOW_V1_PROTOCOL.json')==r['protocol_sha256']
    for path,digest in p['source_sha256'].items(): assert sha(ROOT/path)==digest
    emissions=HERE/'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv'
    assert sha(emissions)==r['emissions_sha256']
    grouped=defaultdict(list)
    with emissions.open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream): grouped[row['native_file']].append(row)
    assert len(grouped)==len(r['records'])==40
    assert r['source_windows']==4177 and len(r['source_bout_ids'])==162
    assert len(r['source_native_sha256'])==18 and not set(grouped)&set(r['source_native_sha256'])
    assert set(p['source_users']).isdisjoint(p['validation_users']+p['descriptive_final_users'])
    assert all(abs(v-r['source_windows']/3)<1e-8 for v in r['source_class_weight_mass'].values())
    assert r['source_state_immutable'] and not r['default_promoted'] and not r['physical_validation_proven']
    for record in r['records']:
        n=record['samples'];truth=expand(record['truth_rle'],n);pred=expand(record['prediction_rle'],n)
        rows=grouped[record['native_file']]
        ends=np.array([int(row['emission_sample']) for row in rows])
        probability=np.array([[float(row[f'p_{c}']) for c in range(3)] for row in rows])
        np.testing.assert_array_equal(ends,np.arange(39,n,10))
        assert len(rows)==record['emissions']
        assert record['unknown_warmup_samples']==39 and np.all(pred[:39]==-1)
        expected=np.repeat(probability,10,axis=0)[:n-39]
        # Final hold can be shorter than a hop; no future extrapolation or
        # confident class invented for the initialization prefix.
        assert len(expected)==n-39
        np.testing.assert_array_equal(pred[39:],expected.argmax(1))
        y=truth[39:];called=pred[39:];f=[]
        for c in range(3):
            tp=np.sum((y==c)&(called==c));den=np.sum(y==c)+np.sum(called==c)
            f.append(2*tp/den if den else 0)
        metrics={'macro_f1':sum(f)/3,'accuracy':np.mean(y==called),
                 'log_loss':-np.mean(np.log(np.clip(expected[np.arange(len(y)),y],np.finfo(float).eps,1))),
                 'brier':np.mean((expected-np.eye(3)[y])**2)}
        for key,value in metrics.items(): assert abs(value-record['scores'][key])<1e-10
        events=record['transition_hold']['events'];onsets=np.flatnonzero(np.diff(truth))+1
        assert len(events)==len(onsets)==8
        for event,onset in zip(events,onsets):
            assert event['eligible'] and event['onset_sample']==onset
            left,right,stop=event['reaction_start'],event['reaction_stop'],event['maintenance_stop']
            assert left==onset-100 and right==onset+100
            new,old=truth[onset],truth[onset-1]
            reaction=pred[left:right];hold=pred[right:stop]
            correct=bool(new in reaction and set(reaction)<=set([old,new]) and np.all(hold==new))
            assert event['correct']==correct
            assert event['maintenance_error_samples']==sum(hold!=new)
            assert event['maintenance_switches']==sum(hold[1:]!=hold[:-1])
        assert sum(e['correct'] for e in events)==record['transition_hold']['correct_transitions']
    for phase,conditions in r['summaries'].items():
        for posture,summary in conditions.items():
            group=[rec for rec in r['records'] if rec['phase']==phase and (posture=='ALL' or rec['posture']==posture)]
            assert len(group)==summary['recordings']
            assert abs(sum(rec['scores']['macro_f1'] for rec in group)/len(group)-summary['equal_recording_macro_f1'])<1e-12
            total=sum(rec['transition_hold']['eligible_transitions'] for rec in group)
            correct=sum(rec['transition_hold']['correct_transitions'] for rec in group)
            assert summary['eligible_transitions']==total and summary['correct_transitions']==correct
            assert summary['transition_hold_accuracy']==correct/total
