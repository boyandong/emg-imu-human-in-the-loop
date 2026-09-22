"""Verify discovery evidence without rehashing multi-GB immutable archives."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import numpy as np


def verify(root):
    required=('DATASET_INVENTORY.md','DATASET_CANDIDATES.csv','FAILURE_BENCHMARK_MATRIX.md',
      'BENCHMARK_SELECTION_REPORT.md','GESTURE_ONTOLOGY.md','SENSOR_LAYOUTS.md',
      'DATASET_MANIFEST.json','DS2_ACCESS_AUDIT.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required}
    candidates=list(csv.DictReader((root/'DATASET_CANDIDATES.csv').open(encoding='utf-8-sig',newline='')))
    fields=('dataset','failure_targets','subjects','sessions','gestures','channels','sampling_rate',
      'force_conditions','postures','wearing_conditions','speed_conditions','IMU','raw_data','size',
      'license','official_source','download_method','score','decision','reason')
    if len(candidates)!=14 or len({r['dataset'] for r in candidates})!=14:raise ValueError('Mandatory candidate coverage changed')
    if any(not r.get(k) for r in candidates for k in fields):raise ValueError('Required candidate field missing')
    for row in candidates:
        points=[int(row[k]) for k in row if k.startswith('score_review_') and k not in ('score_review_total','score_review_basis','score_review_status')]
        if len(points)!=8 or any(v<0 or v>5 for v in points) or sum(points)!=int(row['score_review_total']):raise ValueError('Invalid reviewed vector')
    manifest=json.loads((root/'DATASET_MANIFEST.json').read_text(encoding='utf-8'))
    ds2_access=json.loads((root/'DS2_ACCESS_AUDIT.json').read_text(encoding='utf-8'))
    if (ds2_access['page_status']!='accessible_without_sign_in'
            or ds2_access['download_attempt']['archive_downloaded']
            or ds2_access['historical_identity']!='unproven'):
        raise ValueError('DS2 access boundary changed')
    archives=[]
    for dataset in manifest['datasets']:
        if dataset['status']!='downloaded_verified':continue
        path=Path(dataset['path']); size=path.stat().st_size
        if size!=dataset['size'] or not dataset.get('sha256'):raise ValueError('Archive size/hash record missing or changed')
        for key in ('size','md5','sha256'):
            if f'expected_{key}' in dataset and str(dataset[f'expected_{key}']).lower()!=str(dataset[key]).lower():
                raise ValueError('Recorded official digest/size mismatch')
        archives.append({'id':dataset['id'],'current_file_size':size,'recorded_sha256':dataset['sha256'],
                         'fresh_archive_digest_computed':False})
    sanity=json.loads((root/'SANITY_AUDIT.json').read_text(encoding='utf-8'))
    expected={'libemg_force':(8,1000),'epn612':(8,200),'semg_manus':(8,200),
              'electrode_shift':(8,200),'unibo_inail':(4,500),'emg_fmg':(8,2000)}
    selection={
        'libemg_force':(np.arange(1,11),r'S(\d+)/.*_(20P|80P)_',{'20P','80P'}),
        'epn612':(np.arange(1,307),r'/user(\d+)/',{'fist','open'}),
        'semg_manus':([*range(3,17),18],r'/u_(\d+)/.*recording_(slow|fast)_',{'slow','fast'}),
        'electrode_shift':(np.arange(21),r'/subject(\d+)/(training|trial_1)/',{'training','trial_1'}),
        'unibo_inail':(np.arange(1,8),r'user_(\d+)_day_1_posture_(1|2)\.mat',{'1','2'}),
        'emg_fmg':(np.arange(1,28),r'Par (\d+)/Power/(0|1000)/',{'0','1000'}),
    }
    reports,plots=0,0
    subject_checks=[]
    for dataset in sanity['datasets']:
        path=Path(dataset['source_report'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=dataset['report_sha256']:raise ValueError('Sanity report changed')
        source=json.loads(path.read_text(encoding='utf-8'))
        if source['samples']!=dataset['samples'] or len(source['samples'])!=6:raise ValueError('Sample evidence changed')
        channels,rate=expected[dataset['id']]
        population,pattern,conditions=selection[dataset['id']]
        wanted=set(np.random.default_rng(20260915).choice(population,3,replace=False).tolist())
        observed={}
        for sample in source['samples']:
            if sample['channels']!=channels or sample['nan_or_inf_fraction']!=0 or sample['duration_seconds']<=0:
                raise ValueError('Native sanity signal contract failed')
            if abs(sample['samples']/sample['duration_seconds']-rate)>1e-5:raise ValueError('Reported native sample duration/rate mismatch')
            match=re.search(pattern,sample.get('path',sample.get('member','')))
            if match is None:raise ValueError('Native sample identity cannot be recovered')
            subject=int(match.group(1));condition=sample['gesture'] if dataset['id']=='epn612' else match.group(2)
            observed.setdefault(subject,[]).append(condition)
        if set(observed)!=wanted or any(len(values)!=2 or set(values)!=conditions for values in observed.values()):
            raise ValueError('Three deterministic random subjects by two conditions not reproduced')
        subject_checks.append({'dataset':dataset['id'],'subjects':sorted(wanted),'conditions':sorted(conditions),
            'scope':'same native population/seed as original sanity producer; not a full-cohort draw'})
        for plot in dataset['plots']:
            path=Path(plot['file']);raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=plot['sha256']:raise ValueError('Plot changed')
            captions=[e.text for e in ET.fromstring(raw).iter() if e.tag.endswith('text')]
            if len(captions)<4 or captions[0]!=plot['condition_caption']:raise ValueError('Condition caption changed')
            plots+=1
        reports+=1
    if reports!=6 or plots!=36 or len(archives)!=6:raise ValueError('Discovery sample coverage changed')
    total=sum(a['current_file_size'] for a in archives)
    result={'status':'checked_evidence_partial','completion_proven':False,'required_artifacts':hashes,
       'candidate_rows':len(candidates),'score_vectors_checked':len(candidates),'archives':archives,
       'archive_bytes':total,'archive_GB_decimal':total/1e9,'archive_GiB_binary':total/(1024**3),
       'source_sanity_reports_rehashed':reports,'captioned_plot_files_rehashed':plots,'sampled_native_recordings':36,
       'random_subject_condition_checks':subject_checks,
       'limitations':['fresh multi-GB archive digests not recomputed; recorded digests and current sizes only',
          'hash/caption verification does not establish visual or full-population signal quality',
          'reported durations agree with native rates; no independent hardware clock check',
          'publisher-linked DS2 metadata is accessible; ZIP requires Kaggle sign-in and historical identity/raw/old results remain missing',
          'DS2 publication and Kaggle page expose conflicting license labels; redistribution is not cleared',
          'retrospective scorecards do not prove original scoring/phase ordering',
          'secondary candidate paper/licensing/layout verification remains incomplete',
          'current total disk usage not scanned; DISK_USAGE.json remains explicitly dated historical snapshot']}
    (root/'DISCOVERY_DELIVERY_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','candidate_rows','score_vectors_checked','archive_bytes','source_sanity_reports_rehashed','captioned_plot_files_rehashed','completion_proven')}))


if __name__=='__main__':verify(Path('benchmarks/discovery'))
