import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/discovery'

def test_full_native_subset_census_keeps_physical_evidence_boundary():
    audit=json.loads((HERE/'GRAB_CHANNEL_CENSUS_V1.json').read_text(encoding='utf8'))
    for path,digest in audit['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    rows=audit['records']
    assert audit['record_count']==len(rows)==672
    assert len({r['stem'] for r in rows})==672
    assert Counter(r['session'] for r in rows)=={1:224,2:224,3:224}
    assert audit['independent_decoded_values']==sum(r['independent_checked_samples_per_channel']*len(r['selected_channels']) for r in rows)==21504
    for row in rows:
        assert (row['sample_rate_hz'],row['samples'],row['stored_channels'])==(2048,10240,32)
        assert row['selected_channels']==[f'F{i}' for i in range(1,9)]
        assert row['selected_zero_based_columns']==list(range(8))
        assert row['max_decode_error_mv']<=1e-12
        for suffix,key in (('hea','header_sha256'),('dat','signal_sha256')):
            path=Path(audit['data_root'])/row['folder']/(row['stem']+'.'+suffix)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row[key]
    metadata=audit['official_metadata_review']
    assert metadata['unused_one_based_columns']==[17,24,25,32]
    assert metadata['direction']==metadata['our_device_wiring']=='unverified'
    assert audit['all_recordings_header_and_signal_checksum_verified']
    assert not audit['six_axis_imu_proven'] and not audit['physical_layout_for_our_device_proven']
    review=json.loads((HERE/'SECONDARY_PRIMARY_REVIEW_V1.json').read_text(encoding='utf8'))
    entry=next(r for r in review['datasets'] if r['id']=='grabmyo')
    assert entry['native_channel_audit']=='GRAB_CHANNEL_CENSUS_V1.json'
    assert entry['stored_column_mapping_verified'] and not entry['physical_eight_channel_layout_verified']
