"""Read only public publisher metadata; never download signal archives."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parents[1]
SOURCES = {
    'electrode_replacement_secondary': 'https://zenodo.org/api/records/4039550',
    'ninapro_db5': 'https://zenodo.org/api/records/1000116',
    'great': 'https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.8sf7m0czv',
}


def fetch():
    observations = []
    for identity, url in SOURCES.items():
        record = {'dataset': identity, 'url': url,
                  'observed_at_utc': datetime.now(timezone.utc).isoformat()}
        try:
            response = requests.get(url, timeout=20)
            record.update({'http_status': response.status_code,
                           'response_sha256': hashlib.sha256(response.content).hexdigest()})
            response.raise_for_status()
            payload = response.json()
            metadata = payload.get('metadata', payload)
            record['fields'] = {key: metadata[key] for key in
                                ('title', 'license', 'rights', 'identifier', 'doi') if key in metadata}
            record['publisher_record_id'] = payload.get('id')
            record['status'] = 'publisher_metadata_observed'
        except (requests.RequestException, ValueError) as exc:
            record['status'] = 'unresolved'
            record['error_type'] = type(exc).__name__
        observations.append(record)
    result = {'schema': 'secondary_license_metadata_v1', 'observations': observations,
              'scope': 'Selected public API metadata fields with original response digest. No signal archives, account credentials or legal-use determination.',
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / 'SECONDARY_LICENSE_METADATA_V1.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps([{key: row.get(key) for key in ('dataset', 'status', 'http_status', 'fields')}
                      for row in observations]))
    return result


if __name__ == '__main__':
    fetch()
