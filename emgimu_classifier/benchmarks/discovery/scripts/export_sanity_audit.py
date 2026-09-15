"""Export small, hashed evidence for the current six-dataset sanity sample."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

DATASETS = {'libemg_force':8, 'epn612':8, 'semg_manus':8,
            'electrode_shift':8, 'unibo_inail':4, 'emg_fmg':8}


def export(root: Path, output: Path) -> None:
    datasets = []
    for name, channels in DATASETS.items():
        path = root/f'{name}_sanity.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        if len(payload['samples']) != 6:
            raise ValueError(f'{name}: expected six sampled recordings')
        plots = []
        for sample in payload['samples']:
            if sample['channels'] != channels or sample['nan_or_inf_fraction'] != 0:
                raise ValueError(f'{name}: invalid channel count or nonfinite signal')
            svg = root/sample['plot']; data = svg.read_bytes(); tree = ET.fromstring(data)
            captions = [e.text for e in tree.iter() if e.tag.endswith('text')]
            if len(captions) < 4 or not captions[0]:
                raise ValueError(f'{svg}: missing condition caption')
            plots.append({'file':str(svg),'sha256':hashlib.sha256(data).hexdigest(),
                          'condition_caption':captions[0]})
        datasets.append({'id':name, 'source_report':str(path),
            'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'sample_strategy':payload.get('sample_strategy'),
            'samples':payload['samples'], 'plots':plots})
    output.mkdir(parents=True, exist_ok=True)
    audit = {'seed':20260915, 'datasets_checked':6, 'sampled_recordings':36,
        'condition_captioned_plots_checked':36, 'datasets':datasets,
        'scope':'three random subjects and two prespecified conditions per dataset; not a full-population QC audit',
        'limitations':['rate comes from verified acquisition metadata, not an independent hardware clock measurement',
            'near observed extrema is a heuristic, not confirmed ADC clipping',
            'first-channel plots do not visually inspect every channel',
            'UniBo source files contain labelled intervals; one file is not one isolated gesture trial']}
    (output/'SANITY_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k in ('datasets_checked','sampled_recordings','condition_captioned_plots_checked')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); export(a.root,a.output)
