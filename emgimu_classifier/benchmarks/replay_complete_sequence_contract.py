"""Read-only source artifacts; all new audit output stays in workspace."""
import hashlib
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
from emgimu.feature_bank.unibo_sequence_temporal import load_bouts,predict_pair
from emgimu.feature_bank.force_nested_oof import fit_temperature,temperature_probability

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset',type=Path)
parser.add_argument('parent',type=Path)
parser.add_argument('--audit-output',type=Path,default=Path('feature_bank/results/complete_sequence_contract_replay.json'))
args = parser.parse_args()
dataset, parent = args.dataset, args.parent
files = ('fitted_states.pkl','heldout_predictions.npz','source_calibration_predictions.npz','bout_metadata.json','run_manifest.json')
hashes = {name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in files}
states,temps = pickle.loads((parent/'fitted_states.pkl').read_bytes())
before = pickle.dumps(states)
print('Reload native complete bouts; replay saved source and Day6 models without refitting',flush=True)
bouts = load_bouts(dataset)
metadata = [{k:v for k,v in b.items() if k not in ('path','windows')} for b in bouts]
assert metadata == json.loads((parent/'bout_metadata.json').read_text())
collected = {'G5':[],'DTW':[]}; ids=[]; arrays=0
with np.load(parent/'source_calibration_predictions.npz',allow_pickle=False) as saved:
    for user in sorted(states):
        cal = [b for b in bouts if b['user']==user and b['day']==5]
        ev = [b for b in bouts if b['user']==user and b['day']==6]
        source = predict_pair(states[user][0],cal)
        target = predict_pair(states[user][1],ev)
        y = np.array([b['label'] for b in cal])
        for name in source:
            np.testing.assert_array_equal(source[name],saved[f'{user}_{name}'])
            np.testing.assert_allclose(fit_temperature(source[name],y),temps[user][name],rtol=0,atol=1e-12)
            collected[name].append(temperature_probability(target[name],temps[user][name])); arrays+=2
        ids += [b['id'] for b in ev]
with np.load(parent/'heldout_predictions.npz',allow_pickle=False) as saved:
    np.testing.assert_array_equal(ids,saved['bout_ids'])
    for name,parts in collected.items():
        np.testing.assert_array_equal(np.concatenate(parts),saved[name])
assert before == pickle.dumps(states)
assert hashes == {name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in files}
audit = {'status':'ok','prediction_arrays_recomputed':arrays,'source_temperatures_recomputed':14,
    'source_states_immutable':True,'parent_files_unchanged':True,'native_bouts':len(bouts),'target_bouts':len(ids),
    'maximum_error':0.,'classifier_refitted':False,'parent_sha256':hashes,
    'scope':'new complete-sequence input guard preserves saved native full-bout G5/DTW predictions; no short-window or hardware accuracy claim'}
args.audit_output.write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit),flush=True)
