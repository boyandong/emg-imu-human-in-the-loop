"""Check source hashes, copied table contents and available explicit trial split lists."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def verify(source_root:Path, results:Path)->dict:
    provenance=json.loads((results/'provenance.json').read_text(encoding='utf-8'))
    checked_rows=0
    for entry in provenance:
        source=source_root/entry['run_id']/entry['artifact']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'changed source: {source}')
        with source.open(encoding='utf-8-sig',newline='') as handle:expected=list(csv.DictReader(handle))
        with (results/entry['artifact']).open(encoding='utf-8',newline='') as handle:
            actual=[r for r in csv.DictReader(handle) if r['run_id']==entry['run_id']]
        if len(actual)!=len(expected):raise ValueError(f'row count mismatch: {source}')
        for a,b in zip(actual,expected):
            if any(a[key]!=value for key,value in b.items()):raise ValueError(f'copied value mismatch: {source}')
        checked_rows+=len(actual)
    split_checks=0
    for path in (results/'manifests').glob('*__split_trial_ids.json'):
        data=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data,dict):
            if set(data.get('train',[]))&set(data.get('validation',[])):raise ValueError(f'train leakage: {path}')
            split_checks+=1
        else:
            for row in data:
                if set(row.get('calibration',[]))&set(row.get('evaluation',[])):raise ValueError(f'calibration leakage: {path}')
                split_checks+=1
    return {'status':'ok','source_artifacts_checked':len(provenance),'rows_checked':checked_rows,
            'explicit_split_checks':split_checks,'scope':'copied values/hash integrity and available explicit trial lists; not a complete scientific requirement audit'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source_root',type=Path);parser.add_argument('results',type=Path)
    args=parser.parse_args();result=verify(args.source_root,args.results)
    (args.results/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
