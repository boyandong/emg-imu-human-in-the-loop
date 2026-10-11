"""Offline publication review from explicitly retrieved primary sources.

Facts and inspected figures below are manual review, not assertions proven by
tests. Source PDFs/XML stay local; this exports citations, digests and scope.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'benchmarks/discovery'

REVIEWS={
 'hyser':dict(
  source_names=['hyser_author_v1.pdf','hyser_equipment.pdf'],
  reviewed_sections=['II.B Data Acquisition','II.C Experimental Paradigm','equipment_info'],
  visually_reviewed_pdf_pages={'hyser_author_v1.pdf':[2,3],'hyser_equipment.pdf':[1]},
  paper_review='Official v1 author manuscript acquisition/protocol and Figure1 reviewed; official v2 equipment sheet reviewed. This is not a reproduction of published benchmarks.',
  facts=dict(array_count=4,electrodes_per_array=64,array_shape=[8,8],
   interelectrode_distance_mm=10,emg_sample_rate_hz=2048,force_sample_rate_hz=100,
   force_sensors=5,adc_bits=16,interday_interval_range_days=[3,25],
   source_array_concatenation=[1,2,3,4],synchronization='Task-onset trigger sent to EMG and force acquisition systems.'),
  layout='Four64-channel8x8 arrays, two per flexor/extensor side; array1-4 signals concatenated. Figure1 gives within-array numbering, not an eight-channel circumferential ring.',
  boundaries=['Author manuscript belongs to dataset v1; equipment sheet is v2. This does not prove all v1/v2 raw files are identical.',
   'Finger-force subtasks and PR gestures remain distinct. No new raw population download, header mapping, IMU calibration or performance reproduction.']),
 'great':dict(
  source_names=['great_fulltext.xml'],
  reviewed_sections=['Methods/Subjects','Acquisition apparatus','Acquisition protocol','Data Records','Technical Validation','Usage Notes'],
  visually_reviewed_pdf_pages={},
  paper_review='Official Europe PMC JATS full text retrieved; acquisition, records, validation protocol and usage sections reviewed. Author-hosted PDF returned403; figures were not visually reviewed.',
  facts=dict(subjects=8,days=2,sessions_per_day=2,emg_sample_rate_hz=2000,emg_channels=16,
   day1_positions=[2,4,5,6,8],day2_positions=[1,3,5,7,9],shared_position=5,
   repetitions_per_gesture_position=5,hold_seconds=5,rest_seconds=3,
   sensors_removed_between_same_day_sessions=False,recording_starts_after_gesture_formed=True,
   recording_start_after_audio_cue_ms=250),
  layout='Four Quattro sensors yield16 EMG channels in two staggered rows of8; first row starts at extensor carpi ulnaris.',
  boundaries=['Day and noncentral position sets are confounded. Five positions are acquired per day, not nine per day.',
   'Only five-second holds are stored per trial; cue timing is not physiological onset/offset truth. No new native data, six-axis IMU, stored-column wiring or benchmark reproduction.']),
 'electrode_replacement_secondary':dict(
  source_names=['electrode_preprint_v2.pdf'],
  reviewed_sections=['2.1 Subjects','2.2 Measurement setup','2.3 Measurement protocol','2.4 Processing','4 Limitations'],
  visually_reviewed_pdf_pages={'electrode_preprint_v2.pdf':[5]},
  paper_review='Author arXiv2005.02105v2 acquisition, protocol, processing and limitations reviewed; PDFpage5/Figures1-2 visually checked. Paper links the Zenodo4039550 dataset.',
  facts=dict(subjects=10,emg_channels=8,emg_sample_rate_hz=1000,adc_bits=16,amplifier_gain=1000,
   montage='bipolar',dominant_arm=True,interelectrode_longitudinal_distance_cm=2,
   proximal_translation_cm=1,medial_circumferential_displacement_cm=2,
   electrodes_removed_and_replaced_for_all_positions=True,all_positions_single_session=True,
   requested_active_seconds=5,requested_rest_seconds=3),
  layout='Eight uniformly circumferential bipolar pairs; reference over elbow bone. Figure2 shows the shift, but does not label raw columns CH1-CH8 around the arm.',
  boundaries=['A2cm circumferential displacement is not a2-degree rotation. Re-donning is within one recording session, not across days.',
   'Requested5+3second timing does not reconstruct retained/discarded repetitions or sample-level action labels. Circumferential channel order and differential sign remain unattested.']),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_metadata_census(path):
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        names=[e.filename for e in entries if re.search(r'(license|readme|copyright|terms)',e.filename,re.I)]
        # Includes CRC/size as well as names, but is explicitly not a raw-byte digest.
        inventory=[(e.filename,e.file_size,e.CRC) for e in entries]
    return dict(entries=len(entries),metadata_like_names=names,
        central_directory_inventory_sha256=hashlib.sha256(json.dumps(inventory,separators=(',',':')).encode()).hexdigest(),
        raw_archive_bytes_rehashed=False,data_license_proven=False,
        scope='Current local ZIP central-directory inspection only. No metadata-like names does not establish license absence elsewhere or grant redistribution.')


def build(source_dir, archive_path):
    source_dir=Path(source_dir)
    fetched=[]
    for name in ('retrieval.json','retrieval_text.json'):
        fetched.extend(json.loads((source_dir/name).read_text(encoding='utf-8')))
    by_name={r['name']:r for r in fetched}
    required=set(n for row in REVIEWS.values() for n in row['source_names'])|{'db6_page.html','reactemg_readme.md'}
    sources={}
    for name in sorted(required):
        row=by_name[name]
        if row['status']!='downloaded' or sha(source_dir/name)!=row['sha256']:
            raise ValueError('Retrieved primary source changed: '+name)
        sources[name]=row
    review=copy.deepcopy(REVIEWS)
    for row in review.values():
        row['visual_review_sha256']={}
        for name,pages in row['visually_reviewed_pdf_pages'].items():
            for page in pages:
                image=source_dir/(name+f'.page{page}.png')
                row['visual_review_sha256'][image.name]=sha(image)
        row['source_sha256']={name:sources[name]['sha256'] for name in row['source_names']}
        row['physical_eight_channel_layout_verified']=False
        row['six_axis_imu_verified']=False
        row['new_native_experiment']=False
    previous=OUTPUT/'SECONDARY_PRIMARY_REVIEW_V1.json'
    discovery=OUTPUT/'CURRENT_DISCOVERY_STATE.json'
    census=archive_metadata_census(archive_path)
    result=dict(schema='secondary_primary_review_v2',review_date='2026-10-11',
        sources=sources,datasets=review,roam_archive_census=census,
        license_rechecks={
         'ninapro_db6':dict(reviewed_url=sources['db6_page.html']['url'],formal_data_license_verified=None,
            scope='Official DB6 page was reread; it supplies access/citation instructions without an explicit formal data-license identifier. No claim about all external records.'),
         'roam_emg':dict(reviewed_url=sources['reactemg_readme.md']['url'],project_license_reported='MIT',
            separate_data_archive_license_verified=None,
            scope='README states project MIT; local data ZIP has no metadata-like filenames. Neither is a separate data-archive rights determination.')},
        failed_retrievals=[r for r in fetched if r['status']!='downloaded'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (
            previous,discovery,Path(__file__),ROOT/'tests/test_secondary_primary_review_v2.py')},
        raw_sources_redistributed=False,native_experiment_rerun=False,completion_proven=False,
        scope='Manual selected full-text/figure review of three primary publications plus two scoped license rechecks. '
              'Remote retrieval digests and local render digests bind observed sources; raw publications are not in Git. '
              'Metadata tests do not prove paper claims, native efficacy, licensing rights or all-source/full-goal completion.')
    overlay=json.loads(discovery.read_text(encoding='utf-8'))
    overlay['schema']='current_discovery_state_v2'
    overlay['previous_state']='CURRENT_DISCOVERY_STATE.json'
    overlay['secondary_primary_review']='SECONDARY_PRIMARY_REVIEW_V2.json'
    for row in overlay['datasets']:
        if row['id'] in review:
            row['current_primary_review_v2']=review[row['id']]
    overlay['remaining']=[
        'Independent multi-user/day eight-channel recordings and physical device validation.',
        'Some secondary full-paper/figure, native file-to-layout and archive inspection remain scoped; three publication reviews are extended in SECONDARY_PRIMARY_REVIEW_V2.json.',
        'DB6 formal data terms, separately stated ROAM archive terms and reported DS2 license conflict remain unresolved. No licensing determination.',
        'No calibrated anatomical IMU frame, own-device electrode wiring, physical fault truth or fresh prospective efficacy is established.',
        'This overlay does not rehash multi-GB raw archives or authenticate physical clocks.']
    return result,overlay


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_dir',type=Path);p.add_argument('archive',type=Path)
    args=p.parse_args();result,overlay=build(args.source_dir,args.archive)
    for name,data in [('SECONDARY_PRIMARY_REVIEW_V2.json',result),('CURRENT_DISCOVERY_STATE_V2.json',overlay)]:
        (OUTPUT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(reviewed_publications=len(result['datasets']),sources=len(result['sources']),
                         archive_entries=result['roam_archive_census']['entries'],completion_proven=False)))
