"""Bound selected-section GRAB paper review, not device-layout authentication."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9712490/fullTextXML'
DOI = '10.1038/s41597-022-01836-y'
EXPECTED_SHA = '0251e25f08716a500b053cbb376813ddae895ded9c60f41b2d27313e6db439d4'


def text(element): return ' '.join(''.join(element.itertext()).split())


def review(raw):
    root = ET.fromstring(raw)
    ids = {a.attrib['pub-id-type']: a.text for a in root.findall('.//article-id')}
    if ids.get('doi') != DOI or ids.get('pmcid') != 'PMC9712490':
        raise ValueError('Different paper identity')
    ids_to_review = ['Sec3', 'Sec4', 'Sec5', 'Sec6', 'Sec8', 'Sec9', 'Sec17', 'Sec22']
    sections = {}
    for ident in ids_to_review:
        elements = root.findall(f'.//sec[@id="{ident}"]')
        if len(elements) != 1: raise ValueError('Missing or ambiguous reviewed section')
        sections[ident] = text(elements[0])
    # These fixed source checks bind the manually reviewed facts to the native
    # paragraph content; they do not infer current hardware from a public paper.
    required = {'Sec4': ['gain of the device was set to 500', 'sampling rate was set to 2048 Hz',
                          'two rings, each consisting of eight electrodes', 'maintained at 2 cm',
                          'two rings, each consisting of six electrodes', '28 monopolar sEMG electrodes',
                          'center-line of the elbow crease', 'no marks were left'],
                'Sec5': ['normal force level', 'self-defined force levels', 'seven runs', '119 contractions'],
                'Sec6': ['10 Hz and 500 Hz', 'fourth-order Butterworth', 'notch filter of 60 Hz'],
                'Sec8': ['scaling factor', 'physical units (in mVs)'],
                'Sec17': ['monopolar sEMG channels as columns', 'forming bipolar pairs between them']}
    for ident, phrases in required.items():
        if any(phrase not in sections[ident] for phrase in phrases):
            raise ValueError(f'Reviewed acquisition statement changed: {ident}')
    return {'schema': 'grab_primary_topology_v1', 'dataset': 'grabmyo', 'doi': DOI,
            'source_url': SOURCE_URL, 'source_format': 'JATS XML', 'http_status': 200,
            'source_bytes': len(raw), 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'reviewed_section_text_sha256': {k: hashlib.sha256(v.encode('utf8')).hexdigest() for k,v in sections.items()},
            'reviewed_sections': ids_to_review, 'figure_visually_reviewed': False,
            'emg_sample_rate_hz': 2048, 'amplifier_gain': 500,
            'recorded_monopolar_electrodes': 28, 'forearm_rings': 2, 'forearm_channels_per_ring': 8,
            'wrist_rings': 2, 'wrist_channels_per_ring': 6, 'inter_ring_center_distance_cm': 2,
            'first_electrode_anatomical_anchor': 'Each ring starts on the center-line of the elbow crease.',
            'stored_signal_montage': 'Monopolar columns; bipolar pairs may be constructed separately using channel identity.',
            'paper_preprocessing': {'bandpass_hz': [10,500], 'butterworth_order': 4, 'notch_hz': 60},
            'wfdb_physical_unit': 'mV; conversion requires native header scaling',
            'force_instruction': 'Ordinary effort after self-defined soft/medium/hard practice; not a recorded mechanical-force label.',
            'own_device_layout_verified': False, 'clockwise_direction_verified': False,
            'bipolar_pair_sign_verified': False, 'default_promoted': False, 'completion_proven': False,
            'scope': 'Selected text sections reviewed from official open full text; figure, supplementary geometry and performance reproduction not completed. Ring-column identity is bound separately by the 672-record native census. Public monopolar rings do not authenticate our device wiring or imply an eight-bipolar-channel match.'}


def build(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA: raise ValueError('Downloaded source changed; review a new version explicitly')
    result = review(raw)
    result['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'GRAB_PRIMARY_TOPOLOGY_V1.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('Saved selected-section acquisition review; no signal conversion or device claim')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('xml',type=Path); args = parser.parse_args(); build(args.xml)
