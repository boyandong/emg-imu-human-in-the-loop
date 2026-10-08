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
              'minus alternative; their sign conventions are explicitly separate.']
    emg = questions['B']['emg_only_bank']
    lines += ['', '#### Separate EMG-only F0 + F7 bank and provider removals', '',
              'This precommitted EPN42–51 experiment uses source-fitted 48-coordinate F0',
              'and calibrated affine-SPD F7. It has no IMU feature input. Its two-provider',
              'removals are distinct from the earlier whole-bank ablations and the older',
              'Core containing reference IMU. All budgets use the same 1,200 held-out trials;',
              '30 candidate calibration trials per user are reserved even at zero shots.', '',
              '| F7 trials/class | Actual F7 calibration trials/user | Source-only F0 macro-F1 | F0 + F7 macro-F1 | F0 + F7 log loss | Source-only / combined worst-user F1 |',
              '|---|---:|---:|---:|---:|---|']
    for cell in emg['cells']:
        full = f"{cell['F0_F7_macro_f1']:.4f}" if cell['F0_F7_macro_f1'] is not None else 'N/A'
        loss = f"{cell['F0_F7_log_loss']:.4f}" if cell['F0_F7_log_loss'] is not None else 'N/A'
        worst = f"{cell['F0_F7_minimum_user_macro_f1']:.4f}" if cell['F0_F7_minimum_user_macro_f1'] is not None else 'N/A'
        lines.append(f"| {cell['shots_per_class']} | {cell['actual_F7_target_calibration_trials']} | {cell['F0_macro_f1']:.4f} | {full} | {loss} | {cell['F0_minimum_user_macro_f1']:.4f} / {worst} |")
    guard = emg['primary_five_shot']
    lines += ['', f"The five-shot combination improves pooled log loss by {guard['delta_logloss']:.5f} versus source-only F0 and {guard['delta_logloss_over_uniform']:.5f} versus uniform softening. Only {guard['user_logloss_wins']}/10 users improve log loss, so the predeclared seven-user conjunction fails. No default is promoted.", '',
              'Source-only F0 and its uniform control use zero target calibration trials.',
              'The budget labels describe the paired scenario; they do not assign the',
              'F7 calibration cost to F0. F7 at zero shots remains unavailable. The',
              'dataset provides native cue-aligned windows, not online segmentation.',
              '[Paired provider removals](delivery/new_bank_v3/ablation_full_bank.csv)',
              'and [saved source parameters and trial readouts](../benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_RESULTS.json)',
              'retain these boundaries. This is not the complete document-wide bank,',
              'strict calibrated F6, an all-612-user result or own-device efficacy.', STOP]
    variation_lines = ['', '| Same EPN42–51 trials | Equal-user macro-F1 | Sample SD | Worst-user macro-F1 |',
                       '|---|---:|---:|---:|']
    for row in questions['E']['emg_only_anchor_vs_no_target_anchor']:
        variation_lines.append(f"| {row['method']} | {row['equal_user_mean']:.4f} | {row['sample_std_ddof1']:.4f} | {row['minimum']:.4f} |")
    variation_lines += ['', 'These are matched source-only versus five-shot-anchor outcomes on a',
                        'separate cohort from the Mahalanobis/Euclidean comparison above;',
                        'their user variation is not pooled across experiments.']
    diagnostics = data['emg_only_class_diagnostics']
    native_cells = {c['arm']:c for c in diagnostics['cells'] if c['shots_per_class']==5 and c['user']=='ALL'}
    variation_lines += ['', '#### Native class-level calibration changes', '',
                        '| Native EPN class | F0 recall | F0 + F7 recall | Errors corrected | Correct trials harmed |',
                        '|---|---:|---:|---:|---:|']
    for item in diagnostics['paired_class_changes']:
        if item['shots_per_class']!=5 or item['user']!='ALL': continue
        label=item['class_label']
        variation_lines.append(f"| {item['native_gesture']} | {native_cells['F0']['recall'][label]:.3f} | {native_cells['F0_F7']['recall'][label]:.3f} | {item['corrected']} | {item['harmed']} |")
    a,b=native_cells['F0'],native_cells['F0_F7']
    variation_lines += ['', f"Active trials misclassified as rest change from {a['active_predicted_rest_count']}/{a['active_trials']} to {b['active_predicted_rest_count']}/{b['active_trials']}; rest trials misclassified as active change from {a['rest_predicted_active_count']}/{a['support'][0]} to {b['rest_predicted_active_count']}/{b['support'][0]}.", '',
                        'All six native classes have 200 held-out trials. These modest pooled',
                        'recall gains retain substantial missed gestures and can hide individual',
                        'user harms. They do not explain the current hardware user\'s failures.',
                        '[Class confusions and active/rest rates](../benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json)',
                        'and [paired class correction/harm table](../benchmarks/new_bank_v3/EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.csv)',
                        'are descriptive readouts of immutable predictions; no retraining or default promotion.']
    lines[-1:-1] = variation_lines
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
