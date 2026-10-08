"""Read-only census of official GRAB subset channel contracts and independent raw decoding."""
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmarks.grabmyo_crossday import run as grab

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'benchmarks/discovery'
DATA=Path('D:/emg-imu-benchmarks/data/raw/grabmyo_crossday_subset_v1')
NAMES=tuple([f'F{i}' for i in range(1,17)]+['U1']+[f'W{i}' for i in range(1,7)]+['U2','U3']+[f'W{i}' for i in range(7,13)]+['U4'])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    manifest=DATA/'SHA256SUMS.txt'
    recorded=json.loads((HERE/'DATASET_MANIFEST.json').read_text())
    expected=next(r for r in recorded['datasets'] if r['id']=='grabmyo')['official_checksum_manifest_sha256']
    if sha(manifest)!=expected:raise ValueError('Official checksum manifest changed')
    checksums={name.removeprefix('*'):digest for digest,name in (line.split(maxsplit=1) for line in manifest.read_text().splitlines())}
    rows=[]
    for record in grab.records():
        header=DATA/grab.relative_file(record,'hea'); signal=DATA/grab.relative_file(record,'dat')
        for path in (header,signal):
            if sha(path)!=checksums[path.relative_to(DATA).as_posix()]:raise ValueError('Native file checksum failed')
        lines=header.read_text().splitlines()
        identity,channels,rate,samples=lines[0].split()[:4]
        names=tuple(line.split()[-1] for line in lines[1:33])
        if names!=NAMES or (int(channels),int(rate),int(samples))!=(32,2048,10240):
            raise ValueError('Native header contract differs')
        raw=signal.read_bytes()
        if len(raw)!=int(channels)*int(samples)*2:raise ValueError('Signal byte length differs')
        actual=grab.read_record(DATA,record).reshape(10240,8)
        max_error=0.
        for sample in (0,1,5120,10239):
            for channel in range(8):
                fields=lines[channel+1].split()
                gain,tail=fields[2].split('('); baseline,unit=tail.split(')/')
                if unit!='mV' or fields[1]!='16' or float(gain)<=0:raise ValueError('Invalid channel scale')
                offset=(sample*32+channel)*2
                digital=int.from_bytes(raw[offset:offset+2],byteorder='little',signed=True)
                physical=(digital-int(baseline))/float(gain)
                error=abs(physical-actual[sample,channel]);max_error=max(max_error,error)
                if error>1e-12:raise ValueError('Independent byte decoding differs from loader')
        rows.append({**record,'header_sha256':sha(header),'signal_sha256':sha(signal),
                     'sample_rate_hz':int(rate),'samples':int(samples),'stored_channels':int(channels),
                     'selected_channels':list(names[:8]),'selected_zero_based_columns':list(range(8)),
                     'independent_checked_samples_per_channel':4,'max_decode_error_mv':max_error})
    evidence=[Path(__file__),ROOT/'benchmarks/grabmyo_crossday/run.py',ROOT/'benchmarks/grabmyo_crossday/PROTOCOL.json']
    payload={'schema':'grab_channel_census_v1','data_root':str(DATA),'source_version':'1.1.0',
             'official_checksum_manifest_sha256':expected,
             'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
             'official_metadata_url':'https://physionet.org/content/grabmyo/1.1.0/',
             'official_metadata_review':{'forearm_ring1_one_based_columns':list(range(1,9)),
                  'forearm_ring2_one_based_columns':list(range(9,17)),
                  'unused_one_based_columns':[17,24,25,32],
                  'first_electrode_reference':'Centerline of elbow crease in each ring',
                  'direction':'unverified','our_device_wiring':'unverified',
                  'channel_interpretation':'Stored F1-F8 ring channels. No derived differential F1-F9 pairs or monopolar/bipolar equivalence asserted.'},
             'records':rows,'record_count':len(rows),'independent_decoded_values':len(rows)*4*8,
             'all_recordings_header_and_signal_checksum_verified':True,
             'six_axis_imu_proven':False,'physical_layout_for_our_device_proven':False,
             'scope':'Census of existing frozen subset only; digital columns, units and sample rate verified. No new acquisition, physical direction, live transfer or full 43-user dataset claim.'}
    (HERE/'GRAB_CHANNEL_CENSUS_V1.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    print(f'GRAB native metadata/bytes verified: {len(rows)} records, {len(rows)*32} independently decoded values',flush=True)

if __name__=='__main__':run()
