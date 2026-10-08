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
    window_bank = data['emg_window_bank']
    variation_lines += ['', '#### Six declared EMG window groups: fixed combinations and removals', '',
                        'A separate precommitted EPN52–61 experiment fits 13 source-only models',
                        'on users1–15 and retains all 1,500 held-out native trials. It joins F0,',
                        'F1, centered F2a/F2c, F3b CES, F4a/b/c and local F5 window features.',
                        'The joined bank has 265 coordinates and zero target calibration.',
                        'The missing subfamilies, personal/session/context providers and strict',
                        'F6 remain outside this bank; this is not the document-wide F0–F9 bank.', '',
                        '| Fixed composition | Coordinates | Macro-F1 | Log loss | Worst-user macro-F1 |',
                        '|---|---:|---:|---:|---:|']
    for c in window_bank['cells']:
        variation_lines.append(f"| {c['arm']} | {c['dimension']} | {c['macro_f1']:.4f} | {c['log_loss']:.4f} | {c['minimum_user_macro_f1']:.4f} |")
    primary=window_bank['primary']
    variation_lines += ['', f"All four predeclared joined-bank guards fail; {primary['user_logloss_wins']}/10 users improve log loss. Joined-bank macro-F1 change is {primary['delta_macro_f1']:+.4f}; log-loss improvement is {primary['delta_logloss']:+.4f}, where positive means better. No composition is selected or promoted using these target results.", '',
                        'Every group removal independently refits the source classifier. This',
                        'differs from removing a probability provider in the F0/F7 experiment.',
                        'Direct F0-to-F0-plus-group comparisons are explicitly labelled source-refit',
                        'concatenated increments; other paired comparisons remain alternatives.',
                        '[Native models and readouts](../benchmarks/new_bank_v3/EMG_WINDOW_BANK_V1_RESULTS.json)',
                        'and [all declared group removals](delivery/new_bank_v3/ablation_full_bank.csv)',
                        'preserve the negative results and the restricted scope.']
    availability=data['available_provider_fusion']
    variation_lines += ['', '#### Missing-provider calibration and inference interface', '',
                        'A separate opt-in interface supports arbitrary source-selected named',
                        'providers. Zero-shot uses the source population policy. With labelled',
                        'calibration, missing calibration providers are skipped and the available',
                        'population is renormalized before document-exact reliability shrinkage.',
                        'At prediction time, missing providers are skipped and frozen remaining',
                        'weights are renormalized. No prediction-time refit or target-label input',
                        'is accepted; every provider declares its actual trial and class axes.', '',
                        f"The imported frozen MANUS policy exactly reproduces {availability['native_full_cases']} native fusion blocks. {availability['native_missing_provider_cases']} one-provider omissions match independent weighted arithmetic, and {availability['rejected_invalid_native_calls']} leakage/axis violations are rejected.", '',
                        'Independent synthetic arithmetic verifies zero-shot and calibration-only',
                        'weights, repeated-window trial mass, six named providers and immutable',
                        'inference. Imported mode does not claim this interface fitted the original',
                        'weights. Omissions test software behavior; they do not demonstrate sensor',
                        'fault efficacy or validate the full document bank on actual hardware.',
                        '[Bound software acceptance](AVAILABLE_BANK_FUSION_ACCEPTANCE_V1.json)',
                        'retains that scope. No live deployment default is changed.']
    fusion=data['emg_calibrated_fusion']
    variation_lines += ['', '#### Source OOF-calibrated six-provider late fusion', '',
                        'This separate precommitted EPN62–71 experiment refits all representations,',
                        'scalers and six independent classifiers inside each of three source-user',
                        'OOF folds. Source OOF probabilities fit the six probability temperatures;',
                        'their temperature-fitting losses are not held-out performance. Final',
                        'providers are fitted on source users1–15. The new target cohort uses',
                        'nested0/1/2/5 calibration and identical1,200 held-out native trials.', '',
                        '| Shots/class | Method | Actual calibration trials/user | Macro-F1 | Log loss | Worst-user macro-F1 |',
                        '|---|---|---:|---:|---:|---:|']
    for c in fusion['cells']:
        variation_lines.append(f"| {c['shots_per_class']} | {c['arm']} | {c['actual_target_calibration_trials_per_user']} | {c['macro_f1']:.4f} | {c['log_loss']:.4f} | {c['minimum_user_macro_f1']:.4f} |")
    primary=fusion['primary_five_shot']
    variation_lines += ['', f"The five-shot reliability bank changes macro-F1 by {primary['delta_macro_f1']:+.4f}, while log-loss improvement is {primary['delta_logloss']:+.4f} versus source-calibrated F0 and {primary['delta_logloss_over_uniform']:+.4f} versus uniform. Positive means improvement; both loss changes are negative. {primary['user_logloss_wins']}/10 users improve log loss, so the primary conjunction fails.", '',
                        'Population weights are predeclared uniform; n0=12 and reliability',
                        'temperature1 are fixed before target reading. Only fusion weights adapt',
                        'from calibration. Source-provider classifiers and source probability',
                        'temperatures remain immutable; no personal prototypes or normalization',
                        'are updated. The controls use zero target calibration trials even in',
                        'nonzero-shot scenarios. Zero-shot reliability equals uniform fusion.',
                        'All six provider removals renormalize the same frozen reliability weights;',
                        'they do not refit classifiers or recompute reliability. One-shot within-',
                        'class variation is zero and uses the documented epsilon denominator;',
                        'this numerical definition is not an efficacy guarantee.', '',
                        '[Native OOF probabilities, calibration weights and readouts](../benchmarks/new_bank_v3/EMG_CALIBRATED_FUSION_V1_RESULTS.json)',
                        'and [frozen-weight provider removals](delivery/new_bank_v3/ablation_full_bank.csv)',
                        'retain all results. No target-selected subset or default promotion occurs.',
                        'These six EMG window providers do not validate the complete F0–F9 bank,',
                        'anatomical F6, personal F7, session F8, physical F9 or current hardware.']
    portable=data['portable_emg_bank']
    variation_lines += ['', '#### Portable source-fitted EMG model package', '',
                        f"The opt-in checkpoint contains all fitted feature transforms, scalers, classifiers and source probability temperatures. Unlabelled pure-EMG inference reproduces {portable['native_provider_cases']} users, {portable['native_fusion_cases']} frozen fusion cases and {portable['native_omission_cases']} provider omissions to absolute probability error below1e-12.", '',
                        'The explicit input contract is eight EMG channels at200Hz, with40 samples',
                        'per window, native trial IDs and distinct contiguous chronological window',
                        'offsets starting at zero. Rows may be permuted when offsets are retained.',
                        'Window features are averaged per trial in their original chronological',
                        'order. Inference accepts no evaluation labels, IMU or posture. A separate',
                        'labelled calibration call returns a user-and-bank-bound frozen profile;',
                        'source models remain unchanged. Missing providers are omitted before',
                        'feature extraction and remaining profile weights are renormalized.', '',
                        'Loading this package for inference does not require the source archive',
                        'or retraining. The compiler recovered only final source fits and checked',
                        'their numeric parameters against the unchanged native experiment.',
                        'This is cued-trial inference, not an autonomous streaming detector or',
                        'a250Hz device adapter. The failed primary guard remains unchanged;',
                        'neither GUI defaults nor hardware efficacy are promoted.', '',
                        '[Portable model package](models/epn_emg_calibrated_bank_v1.pkl)',
                        'and [native replay and source fingerprints](FROZEN_EMG_BANK_ACCEPTANCE_V1.json)',
                        'make the source-fitted delivery independently loadable.', '',
                        'The offline command accepts an NPZ containing exactly `emg` (windows',
                        'by40 samples by8 channels), scalar `sample_rate_hz=200`, string',
                        '`trial_ids` per window and integer `window_offsets` per trial.',
                        'Calibration labels are a separate JSON object mapping trial IDs to',
                        'native class indices0–5. Prediction rejects labels. The native class',
                        'axis is noGesture, fist, waveIn, waveOut, open, pinch. Output paths',
                        'must be new files; existing results are preserved.', '',
                        'From the classifier directory, use its installed Python environment',
                        'with `src` on PYTHONPATH. Zero-shot prediction:', '',
                        '```powershell',
                        '$env:PYTHONPATH="src"',
                        'python -m emgimu.feature_bank.frozen_emg_bank_cli_v1 predict --package feature_bank/models/epn_emg_calibrated_bank_v1.pkl --acceptance feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json --windows evaluation.npz --user my_user --output prediction.json',
                        '```', '',
                        'Use the same command with action `calibrate`, `--windows calibration.npz`,',
                        '`--labels calibration_labels.json` and `--output my_user.pkl` to save',
                        'a profile. Optionally pass `--evaluation-trials evaluation_trial_ids.json`',
                        'to enforce reserved evaluation exclusion. Add `--profile my_user.pkl`',
                        'to subsequent prediction calls for the same user and bank. The command',
                        'verifies checkpoint SHA-256 before loading; it never fits source models.',
                        'A separate-process test runs without the native training archive.']
    variation_lines += ['', '#### Target calibration burden: extracted signal versus stored recording time', '',
                        'Native archive accounting binds 600 reserved trial durations and 690',
                        'method-specific cost rows for the two experiments above. Every source-',
                        'only classifier and uniform control uses zero target calibration trials,',
                        'even in a paired nonzero-shot scenario. All candidate calibration trials',
                        'remain excluded from evaluation; reservation is not calibration usage.', '',
                        '| Experiment | Shots/class | Used native trials/user | Used signal seconds/user | Complete stored recording seconds/user | Physical elapsed session time |',
                        '|---|---:|---:|---:|---|---|']
    for c in data['emg_native_calibration_cost']:
        if c['feature_bank'] not in ('F0_F7','reliability_bank') or c['shots_per_class']==0:continue
        variation_lines.append(f"| {c['run_id']} | {c['shots_per_class']} | {c['actual_calibration_trials_per_user']} | {c['used_signal_seconds_mean_per_user']:g} | {c['full_recording_seconds_min_per_user']:.3f}–{c['full_recording_seconds_max_per_user']:.3f} | N/A |")
    variation_lines += ['', 'Five-shot means 30 distinct native trials across all six gestures. The',
                        'classifiers use 24 seconds of extracted windows, while complete stored',
                        'EMG records sum to about148–150 seconds. This does not establish a',
                        '24-second physical onboarding protocol. Hardware setup, prompts, rest',
                        'and wall time are unmeasured. These datasets do not establish whether',
                        'the calibration must repeat each session or cover controlled target',
                        'force/posture; those product requirements remain N/A.', '',
                        '![Separate-cohort calibration benefit and burden](../benchmarks/new_bank_v3/EMG_CALIBRATION_COST_CURVE_V1.png)', '',
                        '[Native cost ledger](../benchmarks/new_bank_v3/EMG_CALIBRATION_BURDEN_V1.json)',
                        'and [figure data and vector export](../benchmarks/new_bank_v3/EMG_CALIBRATION_COST_CURVE_V1.json)',
                        'retain the distinction between extracted exposure, stored recordings',
                        'and unknown physical time. Zero-shot F7 is unavailable. Curves are',
                        'separate by cohort; neither longer calibration nor a few-second physical',
                        'calibration guarantee follows from these mixed outcomes.']
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
