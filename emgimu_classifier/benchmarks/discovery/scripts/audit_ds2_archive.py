"""Verify the publisher-linked DS2 v8 ZIP without extracting raw recordings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


EXPECTED_ARCHIVE_BYTES = 1_123_505_003
EXPECTED_FILE_BYTES = 1_312_583_609
EXPECTED_FILES = 102


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(archive: Path, output: Path) -> dict:
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError('DS2 v8 archive size differs from official download length')
    with zipfile.ZipFile(archive) as bundle:
        files = [member for member in bundle.infolist() if not member.is_dir()]
        names = [member.filename for member in files]
        if len(files) != EXPECTED_FILES or len(set(names)) != len(names):
            raise ValueError('DS2 v8 file count or names differ from public API listing')
        for name in names:
            parts = PurePosixPath(name.replace('\\', '/'))
            if parts.is_absolute() or '..' in parts.parts or ':' in parts.parts[0]:
                raise ValueError(f'unsafe archive path: {name}')
        total = sum(member.file_size for member in files)
        if total != EXPECTED_FILE_BYTES:
            raise ValueError('DS2 v8 uncompressed bytes differ from public API listing')
        types = Counter(Path(member.filename).suffix.lower() for member in files)
        if types != {'.mat': 5, '.tdms': 97}:
            raise ValueError(f'DS2 v8 file types differ from public API listing: {types}')
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ValueError(f'ZIP CRC failed: {corrupt}')
        members = [{'name': member.filename, 'bytes': member.file_size,
                    'crc32': f'{member.CRC:08x}'} for member in files]
    result = {
        'status': 'complete_public_v8_archive_crc_verified_historical_identity_unproven',
        'source_ref': 'cinthyazuniga/ds2-emg-signals-three-force-type',
        'dataset_version': 8,
        'archive_path': str(archive.resolve()),
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': archive_sha256(archive),
        'files': len(files),
        'uncompressed_bytes': total,
        'file_types': dict(types),
        'zip_crc_status': 'all_members_ok',
        'members': members,
        'historical_identity': 'unproven',
        'boundary': ('Public Kaggle v8 archive is complete and its ZIP members pass CRC. '
                     'No old B0/X1-H/X2 archive hash, code or result manifest proves '
                     'that this release was the historical experiment input.'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'archive_bytes',
          'archive_sha256', 'files', 'uncompressed_bytes', 'zip_crc_status')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.archive, args.output)
