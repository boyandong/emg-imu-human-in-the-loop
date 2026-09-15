"""Acquire original secondary archive; no derived trial labels or model training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests

FILES={
    'EMG dataset.7z':(315604904,'c577384488e7fd0d7f7916485c87d72b','EMG%20dataset.7z'),
    'README.txt':(773,'d22fe9bdc536cee8f4695d4bfe54301c','README.txt'),
}


def digest(path):
    sha=hashlib.sha256(); md5=hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):
            sha.update(chunk);md5.update(chunk)
    return sha.hexdigest(),md5.hexdigest()


def acquire(root):
    root=root.resolve();workspace=Path.cwd().parents[1].resolve()
    if not root.is_relative_to(workspace):raise ValueError('Acquisition target must stay inside current workspace')
    root.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.trust_env=False
    records=[]
    for name,(size,official_md5,encoded) in FILES.items():
        target=root/name;partial=root/(name+'.part')
        url=f'https://zenodo.org/api/records/4039550/files/{encoded}/content'
        if not target.exists():
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>size:raise ValueError('Partial download exceeds official size')
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with session.get(url,headers=headers,stream=True,timeout=(15,30)) as response:
                response.raise_for_status()
                if offset and response.status_code!=206:offset=0
                if response.status_code==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected resumed response range')
                progress=time.monotonic()
                with partial.open('ab' if offset else 'wb') as handle:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:
                            handle.write(chunk);offset+=len(chunk)
                            if offset>size:raise ValueError('Response exceeds official file size')
                        if time.monotonic()-progress>=10:
                            print(f'{name}: {offset}/{size} bytes ({offset/size:.1%})',flush=True);progress=time.monotonic()
            if partial.stat().st_size!=size:raise ValueError('Incomplete response retained for resume')
            sha,md5=digest(partial)
            if md5!=official_md5:raise ValueError('Official archive MD5 mismatch; partial retained')
            partial.replace(target)
        sha,md5=digest(target)
        if target.stat().st_size!=size or md5!=official_md5:raise ValueError('Existing immutable acquisition differs from official file')
        records.append({'file':str(target),'source':url,'size':size,'sha256':sha,'md5':md5,'official_md5':official_md5})
        print(f'{name}: size and official MD5 verified',flush=True)
    audit={'dataset':'electrode_replacement','source':'https://zenodo.org/records/4039550',
        'release':'10.5281/zenodo.4039550','license':'CC-BY-4.0','status':'acquired_pending_native_sanity',
        'files':records,'acquired_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'original data archive and README only; videos not downloaded; no extraction or classifier training',
        'capability_boundary':'recordings contain repeated movements; native interval/trial annotations not established'}
    (root/'acquisition_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'status':audit['status'],'files':len(records),'archive_bytes':FILES['EMG dataset.7z'][0]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();acquire(args.root)
