import hashlib
import json
from pathlib import Path
import re
import pytest
from benchmarks.new_bank_v3.render_current_scientific_report_v1 import render, START, STOP

ROOT = Path(__file__).resolve().parents[1]


def test_required_report_current_answers_numerical_boundaries_and_local_links():
    source = ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json'
    data = json.loads(source.read_text(encoding='utf8'))
    report = (ROOT / 'feature_bank/REPORT.md').read_text(encoding='utf8')
    assert report.count(START) == report.count(STOP) == 1
    section = report[report.index(START):report.index(STOP)+len(STOP)]
    assert section == render(data, hashlib.sha256(source.read_bytes()).hexdigest())
    assert '### Answers to questions A–H, with remaining uncertainty' not in report
    assert section.count('| A:') == section.count('| H:') == 1
    for target in re.findall(r'\]\(([^)]+)\)', section):
        assert (ROOT/'feature_bank'/target).resolve().is_file()
    # Independent numerical interpretation: both budget increases reduce the
    # worst user's score, despite the better mean, and all native hold failures
    # remain visible rather than reporting successful windows as whole actions.
    assert '26/160' in section and '57/160' in section
    assert '0.5315' in section and '0.5026' in section
    assert '| N/A |' in section and 'not hardware latency' in section
    assert '### Calibration recovery and model-composition limits' in report
    assert 'F7_AFFINE_CORE_MATCHED_REPORT.md' in report
    assert '| open | 0.355 | 0.390 | 7 | 0 |' in section
    assert '| fist | 0.525 | 0.530 | 5 | 4 |' in section
    assert '194/1000 to 189/1000' in section
    assert '57/200 to 49/200' in section
    assert '| window_bank | 265 | 0.3902 | 1.9194 |' in section
    assert 'All four predeclared joined-bank guards fail; 0/10' in section
    assert '| 5 | reliability_bank | 30 | 0.5178 | 1.4720 |' in section
    assert '| 5 | F0 | 0 | 0.4969 | 1.3616 |' in section
    assert 'versus uniform. Positive means improvement; both loss changes are negative.' in section
    assert '| EMG_F0_F7_BANK_V1 | 5 | 30 | 24 | 147.900–149.720 | N/A |' in section
    assert '| EMG_CALIBRATED_FUSION_V1 | 5 | 30 | 24 | 149.155–149.530 | N/A |' in section
    assert '24-second physical onboarding protocol' in section
    assert '#### Raw quality masks and explicit Unknown decisions' in section
    assert 'Soft routing changes3 previously correct unmodified trials to wrong' in section
    assert '| dropout_ch3 | structural | 124 | 0.0000 | 0.0000 | N/A |' in section
    assert 'real' in section and 'normal-data false-rejection rate cannot be estimated' in section


def test_report_renderer_escapes_table_text_and_rejects_incomplete_question_sets():
    data = json.loads((ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json').read_text(encoding='utf8'))
    data['questions'][0]['answer'] = 'literal|separator\nand newline'
    rendered = render(data, 'fixture')
    assert 'literal&#124;separator and newline' in rendered
    data['questions'] = data['questions'][:-1]
    with pytest.raises(ValueError, match='Eight ordered questions'): render(data, 'fixture')
