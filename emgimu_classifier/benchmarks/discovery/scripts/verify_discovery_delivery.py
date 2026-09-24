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
      'DS2_NATIVE_MAT_AUDIT.json','DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json',
      'DS2_MAT_TRIAL_WINDOW_JOIN.csv','DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json',
      'DS2_TDMS_RAW_EXACT_JOIN.csv',
      'public_ds2_subject_gesture/REPORT.md',
      'public_ds2_subject_gesture/RESULTS.json',
      'public_ds2_subject_gesture/TRIAL_PREDICTIONS.csv',
      'public_ds2_subject_gesture/CONDITIONAL_INCREMENTAL.csv',
      'public_ds2_subject_gesture/ERROR_COMPLEMENTARITY.csv',
      'public_ds2_subject_gesture/VERIFICATION.json',
      'scripts/public_ds2_subject_gesture_incremental.py',
      'scripts/verify_public_ds2_subject_gesture.py',
      'DS2_TDMS_FIRST_METADATA_AUDIT.json',
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
    ds2_join=json.loads((root/'DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json').read_text(encoding='utf-8'))
    join_rows=list(csv.DictReader((root/'DS2_MAT_TRIAL_WINDOW_JOIN.csv').open(encoding='utf-8',newline='')))
    ds2_exact=json.loads((root/'DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json').read_text(encoding='utf-8'))
    exact_rows=list(csv.DictReader((root/'DS2_TDMS_RAW_EXACT_JOIN.csv').open(encoding='utf-8',newline='')))
    ds2_study=json.loads((root/'public_ds2_subject_gesture/RESULTS.json').read_text(encoding='utf-8'))
    ds2_study_verification=json.loads((root/'public_ds2_subject_gesture/VERIFICATION.json').read_text(encoding='utf-8'))
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
    if (ds2_join['status']!='public_ds2_raw_to_mav_window_order_verified_gesture_labels_partial'
            or ds2_join['source_sha256']['raw_mat_sha256']!=ds2_native['raw_mat_sha256']
            or ds2_join['source_sha256']['gesture_label_mat_sha256']!=ds2_native['gesture_label_mat_sha256']
            or ds2_join['native_audit_sha256']!=hashes['DS2_NATIVE_MAT_AUDIT.json']
            or ds2_join['trial_join_csv_sha256']!=hashes['DS2_MAT_TRIAL_WINDOW_JOIN.csv']
            or ds2_join['raw_trials']!=2863 or ds2_join['mav_window_rows']!=332108
            or ds2_join['mav_values_compared']!=996324
            or ds2_join['mav_values_over_1e_minus_10']!=0
            or ds2_join['global_max_mav_abs_error']!=0
            or ds2_join['uniform_gesture_label_trials']!=2862
            or len(ds2_join['mixed_gesture_label_trials'])!=1
            or ds2_join['mixed_gesture_label_trials'][0]['raw_trial_index_zero_based']!=209
            or len(join_rows)!=2863
            or [int(row['raw_trial_index_zero_based']) for row in join_rows]!=list(range(2863))
            or any(int(row['first_window_index_zero_based'])!=116*i
                   or int(row['last_window_index_zero_based'])!=116*i+115
                   or int(row['mav_values_over_tolerance'])!=0
                   for i,row in enumerate(join_rows))
            or sum(row['uniform_gesture_label']=='True' for row in join_rows)!=2862
            or join_rows[209]['uniform_gesture_label']!='False'):
        raise ValueError('DS2 raw-to-MAV window join audit inconsistent')
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
    group_index={(row['member'],int(row['group_ordinal_zero_based'])):row
                 for row in group_rows}
    if (ds2.get('raw_mav_window_join_audit')!='benchmarks/discovery/DS2_MAT_TRIAL_WINDOW_JOIN_AUDIT.json'
            or ds2.get('tdms_exact_waveform_join_audit')!='benchmarks/discovery/DS2_TDMS_RAW_EXACT_JOIN_AUDIT.json'
            or ds2.get('verified_tdms_subject_trials')!=2833
            or ds2.get('unmatched_tdms_subject_trials')!=30
            or ds2_exact['status']!='partial_exact_join'
            or ds2_exact['source_archive_sha256']!=ds2['sha256']
            or ds2_exact['source_raw_mat_sha256']!=ds2_native['raw_mat_sha256']
            or ds2_exact['source_trial_window_join_sha256']!=hashes['DS2_MAT_TRIAL_WINDOW_JOIN.csv']
            or ds2_exact['join_csv_sha256']!=hashes['DS2_TDMS_RAW_EXACT_JOIN.csv']
            or ds2_exact['tdms_members']!=97
            or ds2_exact['eligible_groups_at_least_15000_samples']!=3085
            or ds2_exact['raw_trials']!=2863
            or ds2_exact['exact_signal_matches']!=2833
            or ds2_exact['uniquely_matched_raw_trials']!=2833
            or ds2_exact['unmatched_raw_trial_indices_zero_based']!=list(range(389,419))
            or ds2_exact['multiply_matched_raw_trial_indices_zero_based']
            or ds2_exact['prefix_fingerprint_collision_keys']!=0
            or ds2_exact['unmatched_trials_searched_at_nonzero_tdms_group_offsets']!=30
            or ds2_exact['nonzero_offset_exact_subsequence_matches']
            or sum(ds2_exact['unique_subject_trial_counts'].values())!=2833
            or len(exact_rows)!=2833
            or len({int(row['raw_trial_index_zero_based']) for row in exact_rows})!=2833
            or any(int(row['raw_trial_index_zero_based']) in range(389,419)
                   or int(row['matched_samples_per_channel'])!=15000
                   or not row['tdms_member'].startswith(f"SEMG-{int(row['subject_folder']):02d}/")
                   or row['tdms_member'] not in archived_tdms
                   or (row['tdms_member'],int(row['tdms_group_ordinal_zero_based'])) not in group_index
                   or int(row['tdms_samples_per_channel'])!=int(group_index[
                       (row['tdms_member'],int(row['tdms_group_ordinal_zero_based']))]['samples_per_channel'])
                   or row['gesture_code_if_uniform']!=join_rows[
                       int(row['raw_trial_index_zero_based'])]['gesture_label_if_uniform']
                   for row in exact_rows)):
        raise ValueError('DS2 exact raw-MAT to TDMS subject join inconsistent')
    if (ds2.get('subject_held_out_gesture_study')!='benchmarks/discovery/public_ds2_subject_gesture/RESULTS.json'
            or ds2_study['status']!='public_v8_gesture_only_subject_held_out_exploratory'
            or ds2_study['eligible_trials']!=2832
            or ds2_study['source_sha256']['raw_mat']!=ds2_native['raw_mat_sha256']
            or ds2_study['source_sha256']['tdms_join_csv']!=hashes['DS2_TDMS_RAW_EXACT_JOIN.csv']
            or ds2_study['source_sha256']['window_join_csv']!=hashes['DS2_MAT_TRIAL_WINDOW_JOIN.csv']
            or ds2_study['source_sha256']['runner']!=hashes['scripts/public_ds2_subject_gesture_incremental.py']
            or ds2_study['trial_predictions_sha256']!=hashes['public_ds2_subject_gesture/TRIAL_PREDICTIONS.csv']
            or ds2_study['conditional_incremental_sha256']!=hashes['public_ds2_subject_gesture/CONDITIONAL_INCREMENTAL.csv']
            or ds2_study['error_complementarity_sha256']!=hashes['public_ds2_subject_gesture/ERROR_COMPLEMENTARITY.csv']
            or ds2_study['scores']['validation']['F0']['trials']!=587
            or ds2_study['scores']['final']['F0']['trials']!=558
            or ds2_study_verification['status']!='all_held_out_trial_scores_recomputed'
            or ds2_study_verification['result_sha256']!=hashes['public_ds2_subject_gesture/RESULTS.json']
            or ds2_study_verification['prediction_rows']!=5725
            or ds2_study_verification['arm_count']!=5):
        raise ValueError('DS2 public gesture-only held-out study inconsistent')
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
           'raw_to_mav_join_verified_trials':ds2_join['raw_trials'],
           'uniform_gesture_label_trials':ds2_join['uniform_gesture_label_trials'],
           'mixed_gesture_label_trial_zero_based':209,
           'exact_raw_mat_to_tdms_subject_trials':ds2_exact['uniquely_matched_raw_trials'],
           'unmatched_raw_mat_trials':len(ds2_exact['unmatched_raw_trial_indices_zero_based']),
           'gesture_only_subject_held_out_trial_counts':{
               split:ds2_study['scores'][split]['F0']['trials']
               for split in ('validation','final')},
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
          'TDMS metadata alone cannot join the 3210 groups to MAT trials; exact raw waveforms verify 2833 subject-folder joins, leaving 30 unmatched and no force labels',
          'public DS2 raw MAT to MAV window order is numerically verified; one mixed gesture-code block is ambiguous, 30 subjects are unmatched, and force/historical identity remain unproven',
          'public DS2 gesture-only subject-held-out family increments are new candidate-v8 results, not historical force-run reproduction or own-device eight-channel evidence',
          'DS2 publication and Kaggle page expose conflicting license labels; redistribution is not cleared',
          'retrospective scorecards do not prove original scoring/phase ordering',
          'secondary candidate paper/licensing/layout verification remains incomplete',
          'current total disk usage not scanned; DISK_USAGE.json remains explicitly dated historical snapshot']}
    (root/'DISCOVERY_DELIVERY_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','candidate_rows','score_vectors_checked','archive_bytes','source_sanity_reports_rehashed','captioned_plot_files_rehashed','completion_proven')}))


if __name__=='__main__':verify(Path('benchmarks/discovery'))
