"""Render current A-H evidence in the required REPORT without retraining."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json'
REPORT = ROOT / 'feature_bank/REPORT.md'
START = '<!-- CURRENT_SCIENTIFIC_ANSWERS_V1_START -->'
STOP = '<!-- CURRENT_SCIENTIFIC_ANSWERS_V1_STOP -->'


def safe(value): return str(value).replace('|', '&#124;').replace('\n', ' ')


def render(data, source_sha):
    if [q['question_id'] for q in data['questions']] != list('ABCDEFGH'):
        raise ValueError('Eight ordered questions required')
    lines = [START, '### Current answers to questions A–H (independent versioned evidence)', '',
             'These answers replace the earlier global A–H summary. Individual earlier',
             'experiments below retain their original cohort, method and budget boundaries.',
             'No new own-device efficacy or complete seven-axis robustness is claimed.', '',
             '[Machine-readable answers and measurements](CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json).',
             f'Source SHA-256: `{source_sha}`.', '',
             '| Question | Current evidence-based answer | Interpretation boundary |',
             '|---|---|---|']
    for q in data['questions']:
        links = ', '.join(f"[{Path(p).name}](../{p})" for p in q['evidence'])
        lines.append(f"| {q['question_id']}: {safe(q['question'])} | {safe(q['answer'])} {links} | {safe(q['boundary'])} |")
    lines += ['', '#### Quantitative evidence behind the current answers', '',
              'The following values retain their native units and denominators; they are',
              'not interchangeable measures of one global bank or device.', '',
              '| ROAM cohort (20 recordings each) | Raw / confirmed shared-sample macro-F1 | Successful hold events (raw / confirmed) | Maintenance switches (raw / confirmed) | Unknown samples (raw / confirmed) |',
              '|---|---:|---:|---:|---:|']
    questions = {q['question_id']: q for q in data['questions']}
    for cell in questions['A']['measurements']:
        n = cell['eligible_transitions_per_arm']
        lines.append(f"| {cell['phase']} | {cell['raw_shared_macro_f1']:.4f} / {cell['confirmed_shared_macro_f1']:.4f} | {round(cell['raw_hold_success']*n)}/{n} / {round(cell['confirmed_hold_success']*n)}/{n} | {cell['raw_switches']} / {cell['confirmed_switches']} | {cell['raw_unknown_samples']} / {cell['confirmed_unknown_samples']} |")
    lines += ['', 'Both policies use identical saved emissions. F1 uses shared known samples;',
              'full-record accuracy counts initialization unknown as wrong in the native',
              'artifact. Label confirmation adds no calibrated probabilities; its log loss,',
              'Brier and ECE remain unavailable. Cue-grid timing is not hardware latency.', '',
              '| EPN calibration | Method | Equal-user macro-F1 | Sample SD | Worst-user macro-F1 |',
              '|---|---|---:|---:|---:|']
    for cell in questions['E']['measurements']:
        lines.append(f"| {cell['shots_per_class']} trials/class | {cell['method']} | {cell['equal_user_mean']:.4f} | {cell['sample_std_ddof1']:.4f} | {cell['minimum']:.4f} |")
    lines += ['', 'Both methods use personal calibration; this comparison is not a no-anchor',
              'control. Sample SD uses ten users and nine degrees of freedom. Three users',
              'lose macro-F1 when Mahalanobis calibration increases from 10 to 20 shots.', '',
              '| EPN budget | Independent calibration trials | Extracted signal exposure | Full stored recording duration across users | Actual session wall time |',
              '|---|---:|---:|---:|---|']
    for cell in questions['H']['measurements']:
        lines.append(f"| {cell['shots_per_class']}/class, six classes | {cell['used_trials'][0]} | {cell['used_signal_seconds'][0]:g} s | {cell['full_recording_seconds_min']:.3f}–{cell['full_recording_seconds_max']:.3f} s | N/A |")
    primary = questions['B']['measurements']
    lines += ['', f"For the precommitted independent five-shot F7/Core comparison, log loss improves by {primary['delta_logloss']:.4f}, macro-F1 by {primary['delta_macro_f1']:.4f} and class-mean Brier by {primary['delta_brier']:.4f}. The loss gain beyond uniform softening is {primary['delta_logloss_over_uniform']:.4f}; {primary['user_logloss_wins']}/10 users improve log loss. This is fixed-composition predictive evidence, not a mutual-information estimate.", '',
              'The current MANUS routing cells and negative default-extension checks are',
              'retained in the linked machine-readable answers. Native extension-guard loss',
              'deltas use candidate minus base, while routing loss improvements use base',
              'minus alternative; their sign conventions are explicitly separate.', STOP]
    return '\n'.join(lines)


def update():
    data = json.loads(SOURCE.read_text(encoding='utf8'))
    for path, digest in data['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed evidence: {path}')
    section = render(data, hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    raw = REPORT.read_bytes(); eol = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode('utf8').replace('\r\n', '\n')
    if START in text or STOP in text:
        if text.count(START) != 1 or text.count(STOP) != 1: raise ValueError('Ambiguous current section')
        start = text.index(START); stop = text.index(STOP) + len(STOP)
    else:
        heading = '### Answers to questions A–H, with remaining uncertainty'
        if text.count(heading) != 1: raise ValueError('Missing or ambiguous old A-H section')
        start = text.index(heading)
        stop = text.index('### Calibration recovery and model-composition limits', start)
        section += '\n\n'
    revised = text[:start] + section + text[stop:]
    REPORT.write_bytes(revised.replace('\n', eol).encode('utf8'))
    print('Updated required REPORT A-H section from bound evidence; original experiments preserved')


if __name__ == '__main__': update()
