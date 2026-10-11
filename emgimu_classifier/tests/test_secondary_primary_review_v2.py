import hashlib
import json
from pathlib import Path
import zipfile

from benchmarks.discovery.scripts.review_secondary_sources_v2 import archive_metadata_census

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'benchmarks/discovery'


def read():return json.loads((BASE/'SECONDARY_PRIMARY_REVIEW_V2.json').read_text(encoding='utf-8'))


def test_review_binds_current_manual_review_sources_without_native_completion():
    review=read()
    for rel,h in review['source_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    assert len(review['datasets'])==3 and len(review['sources'])==6
    for name,row in review['datasets'].items():
        assert set(row['source_sha256'])==set(row['source_names'])
        for source,h in row['source_sha256'].items():assert review['sources'][source]['sha256']==h
        assert not any(row[k] for k in ['physical_eight_channel_layout_verified','six_axis_imu_verified','new_native_experiment'])
    assert not any(review[k] for k in ['completion_proven','native_experiment_rerun','raw_sources_redistributed'])


def test_hyser_geometry_and_great_day_position_confounds_are_kept_distinct():
    rows=read()['datasets'];h=rows['hyser'];g=rows['great']
    assert h['facts']['array_count']*h['facts']['electrodes_per_array']==256
    assert h['facts']['array_shape']==[8,8]
    assert h['facts']['source_array_concatenation']==[1,2,3,4]
    assert h['visually_reviewed_pdf_pages']['hyser_author_v1.pdf']==[2,3]
    assert set(g['facts']['day1_positions'])&set(g['facts']['day2_positions'])=={5}
    assert g['facts']['recording_starts_after_gesture_formed']
    assert not g['facts']['sensors_removed_between_same_day_sessions']
    assert not g['visually_reviewed_pdf_pages']  #403 author PDF never becomes a visual review.


def test_replacement_protocol_does_not_imply_days_channel_order_or_retained_boundaries():
    row=read()['datasets']['electrode_replacement_secondary'];f=row['facts']
    assert f['montage']=='bipolar' and f['emg_channels']==8
    assert f['proximal_translation_cm']==1 and f['medial_circumferential_displacement_cm']==2
    assert f['all_positions_single_session'] and f['electrodes_removed_and_replaced_for_all_positions']
    assert 'raw columns CH1-CH8' in row['layout']
    assert any('retained/discarded' in b for b in row['boundaries'])


def test_code_and_database_access_do_not_silently_grant_data_archive_rights():
    review=read();licenses=review['license_rechecks']
    assert licenses['ninapro_db6']['formal_data_license_verified'] is None
    assert licenses['roam_emg']['separate_data_archive_license_verified'] is None
    assert licenses['roam_emg']['project_license_reported']=='MIT'
    census=review['roam_archive_census']
    assert census['entries']==187515 and not census['metadata_like_names']
    assert not census['raw_archive_bytes_rehashed'] and not census['data_license_proven']


def test_archive_census_observes_metadata_names_and_binds_crc_not_license(tmp_path):
    p=tmp_path/'sample.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('records/data.csv','1,2\n');z.writestr('docs/LICENSE.txt','example terms')
    a=archive_metadata_census(p)
    assert a['metadata_like_names']==['docs/LICENSE.txt']
    assert not a['data_license_proven']
    with zipfile.ZipFile(p,'a') as z:z.writestr('other.csv','3,4\n')
    assert a['central_directory_inventory_sha256']!=archive_metadata_census(p)['central_directory_inventory_sha256']


def test_v2_discovery_overlay_links_current_review_and_keeps_full_objective_open():
    state=json.loads((BASE/'CURRENT_DISCOVERY_STATE_V2.json').read_text(encoding='utf-8'))
    assert state['secondary_primary_review']=='SECONDARY_PRIMARY_REVIEW_V2.json'
    assert not state['completion_proven']
    assert {d['id'] for d in state['datasets'] if 'current_primary_review_v2' in d}==set(read()['datasets'])
    assert len(state['remaining'])==5
    index_path=ROOT/'feature_bank/delivery/INDEX.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    for key in ['secondary_primary_review','current_discovery_state']:
        row=index[key];path=(index_path.parent/row['path']).resolve()
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    current=json.loads((ROOT/'feature_bank/CURRENT_REQUIREMENT_REVIEW_V2.json').read_text(encoding='utf-8'))
    assert current['secondary_primary_review']['extended_publication_reviews']==sorted(read()['datasets'])
    assert current['discovery_remaining']==state['remaining']
