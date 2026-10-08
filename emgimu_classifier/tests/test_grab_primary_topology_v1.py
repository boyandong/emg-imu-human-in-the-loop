import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from benchmarks.discovery.scripts.grab_primary_topology_v1 import review, EXPECTED_SHA

ROOT = Path(__file__).resolve().parents[1]


def test_bound_native_paper_sections_and_monopolar_montage():
    artifact = ROOT/'benchmarks/discovery/GRAB_PRIMARY_TOPOLOGY_V1.json'
    result = json.loads(artifact.read_text(encoding='utf8'))
    assert result['generator_sha256'] == hashlib.sha256((ROOT/'benchmarks/discovery/scripts/grab_primary_topology_v1.py').read_bytes()).hexdigest()
    assert result['source_sha256'] == EXPECTED_SHA
    assert result['recorded_monopolar_electrodes'] == 2*8+2*6 == 28
    assert result['emg_sample_rate_hz'] == 2048 and result['wfdb_physical_unit'].startswith('mV')
    assert result['paper_preprocessing'] == {'bandpass_hz':[10,500], 'butterworth_order':4, 'notch_hz':60}
    assert not any(result[k] for k in ('own_device_layout_verified','clockwise_direction_verified',
                                      'bipolar_pair_sign_verified','figure_visually_reviewed','default_promoted','completion_proven'))
    path = ROOT/'tmp/grabmyo_primary_fulltext.xml'
    if not path.exists(): pytest.skip('Downloaded XML unavailable; metadata checks above do not prove native-source review')
    raw = path.read_bytes(); assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    tree = ET.fromstring(raw)
    for ident, digest in result['reviewed_section_text_sha256'].items():
        section = tree.find(f'.//sec[@id="{ident}"]')
        normalized = ' '.join(''.join(section.itertext()).split())
        assert hashlib.sha256(normalized.encode('utf8')).hexdigest() == digest
    assert review(raw)['stored_signal_montage'] == result['stored_signal_montage']
    changed = raw.replace(b'monopolar sEMG channels as columns', b'bipolar sEMG channels as columns')
    assert changed != raw
    with pytest.raises(ValueError, match='statement changed'): review(changed)
    changed = raw.replace(b'10.1038/s41597-022-01836-y', b'10.1038/different-paper')
    with pytest.raises(ValueError, match='paper identity'): review(changed)
