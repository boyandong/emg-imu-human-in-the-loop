"""Replay saved source/calibration/evaluation identities through the strict fusion API."""
import hashlib
import json
from pathlib import Path
import numpy as np
from emgimu.feature_bank.session_fusion_v2 import PROVIDERS, fuse_disjoint_session_predictions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run():
    source = HERE/'F8_ROUTER_MANUS_V1_RESULTS.json'
    parent = json.loads(source.read_text())
    blocks = []
    for b in parent['blocks']:
        kw = dict(probabilities=b['providers'], weights=b['weights'], source_trials=b['source_trials'],
            calibration_trials=b['calibration_trials'], evaluation_trials=b['evaluation_trials'],
            provider_names=PROVIDERS, provider_classes=[tuple(range(6))]*4, classes=tuple(range(6)))
        actual = fuse_disjoint_session_predictions(**kw)
        np.testing.assert_allclose(actual, b['probabilities']['F8'], rtol=0., atol=1e-12)
        rejected = []
        for name, change in [('source_evaluation_overlap', {'source_trials': [b['evaluation_trials'][0]]}),
            ('calibration_evaluation_overlap', {'calibration_trials': [b['evaluation_trials'][0]]}),
            ('class_axis_reversal', {'provider_classes': [tuple(reversed(range(6)))]+[tuple(range(6))]*3}),
            ('provider_axis_reversal', {'provider_names': tuple(reversed(PROVIDERS))})]:
            try:
                fuse_disjoint_session_predictions(**{**kw, **change})
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Unsafe fusion accepted: '+name)
        blocks.append({'phase': b['phase'], 'user': b['user'], 'shots': b['shots'],
                       'evaluation_trials': len(b['evaluation_trials']), 'rejected': rejected,
                       'max_probability_error': float(np.max(np.abs(actual-b['probabilities']['F8'])))})
    paths = [source, Path(__file__), ROOT/'src/emgimu/feature_bank/session_fusion_v2.py']
    result = {'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'blocks': blocks, 'scope': 'Identity guard replay of frozen MANUS F8 probabilities. Native six-class axes declared by original benchmark; no new training, score improvement or device validation.',
        'default_promoted': False, 'completion_proven': False}
    (HERE/'F8_CHECKED_FUSION_V2_RESULTS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(f'{len(blocks)} native blocks matched; {sum(len(b["rejected"]) for b in blocks)} invalid identity/axis calls rejected')


if __name__ == '__main__':
    run()
