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
    def check_partitions(data,path):
        count=0
        if isinstance(data,dict):
            partitions=[set(data[key]) for key in ('train','validation','test','calibration','evaluation') if key in data and isinstance(data[key],list)]
            if any(a&b for i,a in enumerate(partitions) for b in partitions[i+1:]):raise ValueError(f'split leakage: {path}')
            count+=int(len(partitions)>=2)
            count+=sum(check_partitions(value,path) for value in data.values() if isinstance(value,(dict,list)))
        elif isinstance(data,list):
            count+=sum(check_partitions(value,path) for value in data if isinstance(value,(dict,list)))
        return count
    for path in (results/'manifests').glob('*split_trial_ids.json'):
        data=json.loads(path.read_text(encoding='utf-8'))
        split_checks+=check_partitions(data,path)
    return {'status':'ok','source_artifacts_checked':len(provenance),'rows_checked':checked_rows,
            'explicit_split_checks':split_checks,'scope':'copied values/hash integrity and available explicit trial lists; not a complete scientific requirement audit'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source_root',type=Path);parser.add_argument('results',type=Path)
    args=parser.parse_args();result=verify(args.source_root,args.results)
    (args.results/'integrity_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
