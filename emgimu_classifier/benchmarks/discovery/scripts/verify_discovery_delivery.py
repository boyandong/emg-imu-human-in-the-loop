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
      'DATASET_MANIFEST.json','DS2_ACCESS_AUDIT.json','DS2_ARCHIVE_AUDIT.json',
      'DS2_NATIVE_MAT_AUDIT.json','DS2_TDMS_FIRST_METADATA_AUDIT.json',
      'DS2_TDMS_FIRST_METADATA.csv','DS2_TDMS_GROUP_AUDIT.json',
      'DS2_TDMS_GROUPS.csv','CORE_ARCHIVE_DIGEST_AUDIT.json')
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
    archive_digest=json.loads((root/'CORE_ARCHIVE_DIGEST_AUDIT.json').read_text(encoding='utf-8'))
    ds2_access=json.loads((root/'DS2_ACCESS_AUDIT.json').read_text(encoding='utf-8'))
    ds2_archive=json.loads((root/'DS2_ARCHIVE_AUDIT.json').read_text(encoding='utf-8'))
    ds2_native=json.loads((root/'DS2_NATIVE_MAT_AUDIT.json').read_text(encoding='utf-8'))
    tdms=json.loads((root/'DS2_TDMS_FIRST_METADATA_AUDIT.json').read_text(encoding='utf-8'))
    tdms_rows=list(csv.DictReader((root/'DS2_TDMS_FIRST_METADATA.csv').open(encoding='utf-8',newline='')))
    tdms_groups=json.loads((root/'DS2_TDMS_GROUP_AUDIT.json').read_text(encoding='utf-8'))
    group_rows=list(csv.DictReader((root/'DS2_TDMS_GROUPS.csv').open(encoding='utf-8',newline='')))
    if (ds2_access['page_status']!='accessible_without_sign_in'
            or ds2_access['download_attempt']['archive_downloaded']
            or not ds2_access['public_api_check']['archive_downloaded']
            or ds2_access['public_api_check']['files_listed']!=102
            or ds2_access['public_api_check']['sum_listed_file_bytes']!=1312583609
            or ds2_access['historical_identity']!='unproven'):
        raise ValueError('DS2 access boundary changed')
    ds2=next(dataset for dataset in manifest['datasets'] if dataset['id']=='ds2_force')
    ds2_path=Path(ds2['path'])
    if (ds2['status']!='downloaded_verified_candidate_historical_unproven'
            or ds2_path!=Path(ds2_archive['archive_path'])
            or ds2_path.stat().st_size!=ds2['size']
            or ds2['size']!=ds2_archive['archive_bytes']
            or ds2['sha256']!=ds2_archive['archive_sha256']
            or ds2['sha256']!=ds2_access['public_api_check']['archive_sha256']
            or ds2_archive['files']!=102
            or ds2_archive['uncompressed_bytes']!=1312583609
            or ds2_archive['zip_crc_status']!='all_members_ok'
            or ds2_archive['historical_identity']!='unproven'):
        raise ValueError('DS2 candidate archive evidence changed')
    ds2_digest=hashlib.sha256()
    with ds2_path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            ds2_digest.update(chunk)
    if ds2_digest.hexdigest()!=ds2['sha256']:
        raise ValueError('DS2 candidate archive bytes changed')
    if (ds2_native['raw_trials']!=2863 or ds2_native['raw_channels']!=3
            or ds2_native['samples_per_trial_per_channel']!=15000
            or ds2_native['gesture_window_label_count']!=332108
            or sum(ds2_native['gesture_window_label_distribution'].values())!=332108
            or len(ds2_native['sampled_raw_trials'])!=6
            or not ds2_native['raw_all_finite']):
        raise ValueError('DS2 native MAT audit changed')
    archived_tdms={item['name']:item for item in ds2_archive['members'] if item['name'].endswith('.tdms')}
    if (ds2.get('tdms_first_metadata_audit')!='benchmarks/discovery/DS2_TDMS_FIRST_METADATA_AUDIT.json'
            or tdms['tdms_members']!=97 or len(tdms_rows)!=97
            or tdms['subject_folders']!=[f'{subject:02d}' for subject in range(1,21)]
            or tdms['filename_movement_indices']!={'1':20,'2':20,'3':20,'4':20,'5':18}
            or tdms['missing_filename_movement_indices_by_subject']!={'01':['5'],'02':['5']}
            or tdms['combined_movement_files']!=['SEMG-04/SEMG-04_Mv4_Mv5.tdms']
            or {row['member'] for row in tdms_rows}!=set(archived_tdms)
            or any(not row['first_segment_file_name'] for row in tdms_rows)
            or any(row['member_crc32']!=archived_tdms[row['member']]['crc32']
                   or int(row['member_uncompressed_bytes'])!=archived_tdms[row['member']]['bytes']
                   for row in tdms_rows)):
        raise ValueError('DS2 TDMS first-metadata inventory inconsistent with verified archive')
    by_member={}
    for row in group_rows:
        by_member.setdefault(row['member'],[]).append(row)
    if (ds2.get('tdms_group_audit')!='benchmarks/discovery/DS2_TDMS_GROUP_AUDIT.json'
            or tdms_groups['status']!='all_tdms_group_metadata_parsed_no_aggregate_mat_or_force_join'
            or tdms_groups['parser']!='npTDMS 1.11.0'
            or tdms_groups['tdms_members']!=97 or tdms_groups['groups']!=3210
            or len(group_rows)!=3210 or set(by_member)!=set(archived_tdms)
            or tdms_groups['group_property_count_distribution']!={'0':3210}
            or tdms_groups['groups_with_exactly_15000_samples']!=1284
            or tdms_groups['groups_with_at_least_15000_samples']!=3085
            or not 1/1501 < tdms_groups['wf_increment_seconds_median'] < 1/1499
            or any(row['channel_count']!='3' or row['group_property_count']!='0'
                   or int(row['samples_per_channel'])<=0 for row in group_rows)
            or any([int(row['group_ordinal_zero_based']) for row in rows]!=list(range(len(rows)))
                   for rows in by_member.values())):
        raise ValueError('DS2 TDMS full group metadata audit inconsistent')
    if (manifest.get('core_archive_digest_audit')!='benchmarks/discovery/CORE_ARCHIVE_DIGEST_AUDIT.json'
            or archive_digest['status']!='all_six_core_archives_freshly_hashed_and_matched'
            or archive_digest['archive_count']!=6
            or len(archive_digest['archives'])!=6
            or archive_digest['bytes_hashed']!=sum(row['size'] for row in archive_digest['archives'])):
        raise ValueError('Core archive digest audit changed')
    digest_by_id={row['id']:row for row in archive_digest['archives']}
    if len(digest_by_id)!=6:
        raise ValueError('Core archive digest IDs are not unique')
    archives=[]
    for dataset in manifest['datasets']:
        if dataset['status']!='downloaded_verified':continue
        path=Path(dataset['path']); size=path.stat().st_size
        if size!=dataset['size'] or not dataset.get('sha256'):raise ValueError('Archive size/hash record missing or changed')
        verified=digest_by_id[dataset['id']]
        if (Path(verified['path'])!=path or verified['size']!=size
                or verified['mtime_ns_after']!=path.stat().st_mtime_ns
                or verified['sha256'].lower()!=dataset['sha256'].lower()
                or not verified['manifest_sha256_match']
                or (dataset.get('md5') and
                    (verified['md5'].lower()!=dataset['md5'].lower() or not verified['manifest_md5_match']))):
            raise ValueError('Core archive digest does not match current manifest/file metadata')
        for key in ('size','md5','sha256'):
            if f'expected_{key}' in dataset and str(dataset[f'expected_{key}']).lower()!=str(dataset[key]).lower():
                raise ValueError('Recorded official digest/size mismatch')
        archives.append({'id':dataset['id'],'current_file_size':size,'recorded_sha256':dataset['sha256'],
                         'fresh_archive_digest_computed':True,
                         'digest_checked_at_utc':archive_digest['checked_at_utc']})
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
       'ds2_candidate_archive':{'bytes':ds2['size'],'sha256':ds2['sha256'],
           'fresh_sha256_checked':True,'zip_members_crc_checked':ds2_archive['files'],
           'native_raw_mat_trials':ds2_native['raw_trials'],
           'native_gesture_window_labels':ds2_native['gesture_window_label_count'],
           'tdms_first_metadata_members':tdms['tdms_members'],
           'tdms_full_metadata_groups':tdms_groups['groups'],
           'tdms_per_trial_force_mapping':'unproven',
           'historical_identity':'unproven'},
       'archive_bytes':total,'archive_GB_decimal':total/1e9,'archive_GiB_binary':total/(1024**3),
       'core_archive_digest_audit':'CORE_ARCHIVE_DIGEST_AUDIT.json',
       'source_sanity_reports_rehashed':reports,'captioned_plot_files_rehashed':plots,'sampled_native_recordings':36,
       'random_subject_condition_checks':subject_checks,
       'limitations':['current verifier cross-checks the freshly recorded full digests against current sizes and mtimes; it does not rehash the six multi-GB archives on each invocation',
          'hash/caption verification does not establish visual or full-population signal quality',
          'reported durations agree with native rates; no independent hardware clock check',
          'publisher-linked DS2 v8 archive is verified; historical input identity and old-result reproduction remain unproven',
          'TDMS file names give subject/movement provenance clues; all 3210 groups lack group properties and do not join to 2863 aggregate MAT arrays or force levels',
          'DS2 publication and Kaggle page expose conflicting license labels; redistribution is not cleared',
          'retrospective scorecards do not prove original scoring/phase ordering',
          'secondary candidate paper/licensing/layout verification remains incomplete',
          'current total disk usage not scanned; DISK_USAGE.json remains explicitly dated historical snapshot']}
    (root/'DISCOVERY_DELIVERY_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','candidate_rows','score_vectors_checked','archive_bytes','source_sanity_reports_rehashed','captioned_plot_files_rehashed','completion_proven')}))


if __name__=='__main__':verify(Path('benchmarks/discovery'))
