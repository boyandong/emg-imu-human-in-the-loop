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
    song=data['song_f0_runtime']
    variation_lines += ['', '#### Song native8-channel250Hz frozen F0 runtime', '',
                        f"A separate package contains both unchanged Song F0 threshold arms, fitted only on {song['source_windows']} S01/S02 formal source windows. Original source family/model fingerprints matched before packaging; all {song['prediction_rows']} saved paired S03/S04 trial probabilities replay with absolute error below1e-12. No target fitting or target-selected arm occurs.", '',
                        'This package uses50-sample windows at250Hz and the native class axis',
                        'neutral, index_pinch, fist, open_hand. It averages window probabilities',
                        'within each explicit trial; the200Hz EPN package instead averages features.',
                        'The protocols are deliberately distinct and neither resamples the other.', '',
                        'The source-fixed continuous-session filter is fourth-order40Hz high-pass',
                        'followed by50Hz and100Hz Q30 notches, with zero state at session start.',
                        'A separate filter instance carries state across received chunks; independent',
                        'arbitrary-chunk tests exactly match one-pass causal filtering. Resetting',
                        'before each window is incompatible. Explicit50-sample recording start',
                        'indices, trial IDs and contiguous window offsets select inference windows;',
                        'evaluation labels and IMU are not accepted or required.', '',
                        '[Song250Hz model package](models/song_f0_250hz_v1.pkl)',
                        'and [native replay acceptance](SONG_F0_RUNTIME_ACCEPTANCE_V1.json)',
                        'are independently loadable without source data or training. After loading',
                        'the package, `predict_recording(raw, window_starts, trial_ids,',
                        'window_offsets=offsets, sample_rate_hz=250)` filters a complete recording',
                        'and returns both explicitly named arms. `new_filter()` provides a separate',
                        'stateful chunk filter; `predict_windows` requires its declared matching',
                        'preprocessing identity for already filtered windows.', '',
                        'This recovers existing models, not a new live-accuracy result. Song is',
                        'one person on one day; S01–S03 readiness failures and prior S04 inspection',
                        'remain. Cued stable trials do not validate autonomous online recognition',
                        'or new electrode placement. No GUI or model default is changed, and the',
                        'historical Song Brier archive retains its original class-sum convention.']
    song_stream=data['song_f0_stream']
    variation_lines += ['', '#### Full-recording250Hz Song stream and label confirmation', '',
                        'Both fixed source models now run on complete S03/S04 raw recordings.',
                        'They retain causal filter memory, emit trailing50-sample probabilities',
                        'every10 samples and optionally require two consecutive class decisions.',
                        'No cue, trial boundary, evaluation label or IMU enters inference.',
                        'Sample gaps or duplicates require an explicit reset of filter, window',
                        'and confirmation state. Confirmation begins UNKNOWN; before the first',
                        'complete window the stream emits nothing. No model fitting occurs.', '',
                        '| Session | Source threshold | Label policy | Trial-balanced window macro-F1 | Entire stable interval correct | Within-stable switches |',
                        '|---|---|---|---:|---:|---:|']
    for cell in song_stream['cells']:
        for policy in ('raw','confirmed'):
            m=cell[policy]
            variation_lines.append(f"| {cell['session']} | {cell['arm']} | {policy} | {m['trial_balanced_macro_f1']:.4f} | {m['whole_stable_hold_correct']}/{m['eligible_trials']} | {m['within_stable_trial_switches']} |")
    variation_lines += ['', 'S03 emits33610 and S04 emits29790 windows per arm. Only2238 and2283',
                        'windows wholly inside valid completed formal stable intervals are scored;',
                        'the remaining emissions are unscored, never inferred Neutral ground truth.',
                        'Scoring gives each native trial equal mass despite different window counts.',
                        'Independent native HDF5 interval checks confirm those exclusions.', '',
                        'Two-step confirmation reduces switching and raises whole-stable-hold counts',
                        'in these four cells, while window F1 declines in all four. This is a',
                        'stability/decision-delay tradeoff, not a universal accuracy improvement.',
                        'The first window is available after200ms of nominal sampling and each',
                        'additional confirmation waits at least one40ms hop; these are software',
                        'sample-grid timings, not measured wall-time or hardware reaction latency.', '',
                        '[Complete stream arrays](../benchmarks/song_real8/SONG_F0_STREAM_V1_EMISSIONS.npz)',
                        'and [bound native results](../benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json)',
                        'retain every probability, raw label and confirmed label. `SongF0StreamV1`',
                        'requires an explicitly selected existing source arm and recording ID;',
                        '`push(raw_chunk, first_sample_index=..., sample_rate_hz=250)` returns',
                        'causal emissions. Initial and explicitly reset states preserve UNKNOWN.',
                        'This new protocol is frozen separately from the prior three-window trial',
                        'average benchmark. Stable cue intervals do not prove physiological onset',
                        'timing or complete-action success. One user/day and readiness/inspection',
                        'limitations persist; no default model is promoted.']
    gui=data['song_gui_v2']
    variation_lines += ['', '#### Collection-page integration of the fixed Song models', '',
                        'The existing realtime recognition page discovers two bundled v2 JSON',
                        'models: pooled-source and rest-only thresholds. Select a model, click',
                        'Load, and start recognition after connecting. These models require',
                        'eight raw channels at250Hz; they do not resample other hardware rates',
                        'or use IMU. Model SHA-256 is checked before loading. No source recordings',
                        'or classifier/sklearn installation are required by the collection runtime.', '',
                        'Both original source parameter arrays are preserved exactly. The runtime',
                        'casts scaler parameters to float32 before its in-place standardization,',
                        'matching the frozen source pipeline; probabilities remain float64.',
                        'The fixed policy uses a200ms window,40ms hop and two consecutive argmax',
                        'decisions, initially UNKNOWN. The confidence control is disabled for',
                        'these bundles so the page cannot silently change the audited policy.', '',
                        f"All{sum(r['emissions'] for r in gui['records'])} emitted probabilities match the frozen native reference below1e-12,",
                        'and all confirmed labels match exactly. The application Python environment',
                        'also replays both native recordings. Offscreen Qt checks cover selection,',
                        'loading, recognition, pause, explicit packet loss, index discontinuities',
                        'and disconnect. Resets clear the displayed gesture and require a fresh',
                        'complete window. Legacy v1 bundles retain their original policy.', '',
                        '[GUI acceptance and bundle hashes](SONG_GUI_V2_ACCEPTANCE.json)',
                        'bind the exporter, runtime, worker, page and collection tests.',
                        'This proves software integration and retrospective equivalence, not live',
                        'device accuracy or recovery after electrode reattachment. Both source',
                        'arms remain user-selectable experimental candidates; neither is promoted.']
    personal=data['song_personal_session'];cells={(c['shots'],c['arm']):c for c in personal['cells']}
    variation_lines += ['', '#### Persisted personal and current-session workflow on native Song recordings', '',
                        'A separate source-frozen experiment fits six EMG window providers on S01/S02',
                        'and calibrates their probability temperatures using recorded-source-session OOF',
                        'predictions. Population weights and the reliability parameters are selected',
                        'using source recordings only. S03 supplies20 distinct long-term personal trials.',
                        'S04 supplies nested0/4/8/20 current-session trials. All budgets use the same124',
                        'remaining S04 evaluation trials; all20 reserved S04 trials are excluded even at0.', '',
                        '| Current shots/class | S03 personal trials | S04 current trials | F0 macro-F1 / logloss | Personal macro-F1 / logloss | Session macro-F1 / logloss |',
                        '|---:|---:|---:|---:|---:|---:|']
    for shots in (0,1,2,5):
        values=[f"{cells[shots,a]['macro_f1']:.4f} / {cells[shots,a]['log_loss']:.4f}" for a in ('F0','personal','session')]
        variation_lines.append(f"| {shots} | 20 | {4*shots} | "+' | '.join(values)+' |')
    variation_lines += ['', 'F0, population and uniform source controls use zero target calibration trials.',
                        'Personal uses20 S03 trials; session/removal arms use20 S03 plus the stated S04',
                        'budget. Thus0 current shots is not zero total onboarding. One-shot improves',
                        'loss relative to the personal profile here, while two/five-shot worsen it.',
                        'The six-provider source population also improves F1 versus F0 but worsens',
                        'logloss. These mixed outcomes do not authorize choosing the best target budget.', '',
                        '| Removed provider | Delta logloss at0 shots | At1 shot | At2 shots | At5 shots |',
                        '|---|---:|---:|---:|---:|']
    for group in ('F0','F1','F2ac','F3b','F4abc','F5window'):
        deltas=[cells[s,'session_minus_'+group]['log_loss']-cells[s,'session']['log_loss'] for s in (0,1,2,5)]
        variation_lines.append('| '+group+' | '+' | '.join(f'{v:+.4f}' for v in deltas)+' |')
    variation_lines += ['', 'Positive delta means the full fusion has lower loss than that frozen-weight',
                        'removal. Remaining weights are renormalized without refitting or retuning.',
                        'Individual family results, pair errors, all24 removals and signed comparisons',
                        'are included in the current canonical tables.', '',
                        'The workflow keeps separate long-term/current rest centers, additive-epsilon',
                        'Q95 scales and activation ranges; it exports per-family F7 long/local/blended',
                        'coordinates, calibration-only F8 descriptors and source-referenced F9 observations.',
                        'Prototype blending uses only long/current calibration trial counts. User/session',
                        'identities, channel names, preprocessing identity and source bank are bound to',
                        'checksum-verified persistent profiles. Prediction accepts no labels, rejects',
                        'source/personal/current calibration trial overlap and leaves source/long-term',
                        'state unchanged. All5456 trial predictions and saved context arrays reverify;',
                        'session package reloads preserve probabilities exactly.', '',
                        'The offline CLI `python -m emgimu.feature_bank.personal_session_cli_v1` supports',
                        '`enroll`, `session` and `predict`; it requires an explicit preprocessed-window NPZ,',
                        'observed channel names, preprocessing identity, user and recording/session ID.',
                        'Calibration labels are a separate trial-ID JSON. A separate-process test needs',
                        'only source checkpoint, acceptance, windows and profile files, not raw recordings.', '',
                        '[Native workflow results](../benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_RESULTS.json)',
                        'and [package/configuration acceptance](SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json)',
                        'retain44 cells, source-only policy selection and every calibration reservation.',
                        'This experiment evaluates the offline workflow. A separate GUI integration is described below. Normalized views and',
                        'F7/F8/F9 context are not silently inserted into raw-trained classifiers. No',
                        'quality rejection, learned anchor/context classifier, complete DTW bout, native',
                        'anatomical F6 or full document-wide F0-F9 fusion is proved. All four recordings',
                        'are from one user/day with readiness and prior-inspection limits; recording IDs',
                        'do not establish separate days or physical electrode reattachment.']
    lifecycle=data['song_personal_gui']
    variation_lines += ['', '#### Guided personal/session calibration in the collection page', '',
                        'The opt-in Song250 personal/session card invokes the frozen classifier in a',
                        'persistent background Python process, preserving the existing Qt environment.',
                        'Enter the actual user, a unique recording/session ID and observed CH1-CH8',
                        'column order. Load the backend, optionally apply matching ZIP profiles,',
                        'then enroll a person or calibrate a new session. Each guided trial has one',
                        'second to settle and one second of held capture; the user explicitly starts',
                        'the next action. A recording gap cancels pending calibration. Shots count',
                        'native guided trials, not overlapping windows.', '',
                        'Saving creates the profile ZIP plus separate `.windows.npz`, `.labels.json`',
                        'and `.capture.json` companions, without overwriting existing files. Capture',
                        'metadata distinguishes nominal sampled time from begin-to-save wall time.',
                        'New-session profiles require their original personal profile and a different',
                        'recording ID. Calibration changes reliability weights; source classifiers',
                        'and long-term profiles remain fixed. Zero-personal source-population',
                        'prediction is available, as is unlabeled preprocessed-window replay.', '',
                        f"Both population and five-shot session paths verify {sum(r['windows'] for r in lifecycle['records'])} emitted windows",
                        'over the complete S04 recording against independently filtered, source-frozen',
                        'provider probabilities. Errors are below1e-12 and chronological two-confirmation',
                        'labels agree exactly. This recording includes calibration intervals and was',
                        'previously inspected: the check proves numerical parity, not independent',
                        'classification accuracy. Qt tests use the actual classifier subprocess to',
                        'exercise enrollment, persistence, new-session calibration, live input, replay,',
                        'profile mismatch rejection, disconnect and switching back to ordinary models.', '',
                        '[GUI lifecycle provenance](SONG_PERSONAL_GUI_V1_ACCEPTANCE.json)',
                        'retains experimental status. Physical usability, end-to-end latency, new',
                        'electrode placement, quality rejection and full F0-F9 integration remain open.']
    matched=data['song_matched_normalization']
    variation_lines += ['', '#### Matched source training for personal normalization', '',
                        'This separate precommitted experiment fits paired raw and document-normalized',
                        'versions of all six window providers. Both branches reserve40 source trials',
                        '(five/class in each source recording) and fit every representation, scaler',
                        'and classifier on the same245 remaining source trials. Source normalization',
                        'uses only its recording calibration; leave-source-recording-out models refit',
                        'all learned representations. Source OOF temperatures and reliability policy',
                        'use the declared0/1/2/5 scenarios. Reusing source validation for probability',
                        'calibration and policy selection is not nested unbiased source performance.', '',
                        'The same20 S03 personal trials,20 reserved S04 trials and124 evaluation trials',
                        'as the previous lifecycle study are retained. At0 current shots normalization',
                        'uses the long-term profile; at1/2/5 it uses only the current calibration subset.',
                        'Rest median and active Q95+1e-10 are applied exactly once before feature',
                        'extraction, in the same domain as source training. Evaluation never updates',
                        'the source model, probability temperatures, long-term profile or normalizer.', '',
                        '| Domain | Current shots/class | Macro F1 | Logloss | Fist recall | Pinch recall | Open recall |',
                        '|---|---:|---:|---:|---:|---:|---:|']
    for cell in matched['cells']:
        if cell['arm']=='session':
            variation_lines.append(f"| {cell['mode']} | {cell['shots']} | {cell['macro_f1']:.4f} | {cell['log_loss']:.4f} | {cell['recall']['fist']:.4f} | {cell['recall']['index_pinch']:.4f} | {cell['recall']['open_hand']:.4f} |")
    variation_lines += ['', f"The fixed five-shot primary guards are `{json.dumps(matched['primary_guards'],sort_keys=True)}`.",
                        'The normalized candidate fails all four: loss/Brier increase, overall F1',
                        'decreases and fist recall drops, despite improved pinch recall in this cell.',
                        'No target-selected normalization budget or GUI default is promoted. The',
                        'current GUI fixed source model is a different source-fit protocol; do not',
                        'substitute these paired-source numbers for its existing acceptance results.', '',
                        'All80 cells, including48 provider removals, and9920 trial predictions are',
                        'retained. Independent no-fit replay verifies raw-calibration medians/quantiles,',
                        'source normalization reservations, every saved probability and package',
                        'roundtrip. The portable CLI `python -m emgimu.feature_bank.matched_normalization_cli_v1`',
                        'supports enrollment, new-session calibration and unlabeled prediction from',
                        'explicit source packages and profiles without the recording archive. A',
                        'normalized source package rejects a missing personal profile and cannot be',
                        'interchanged with the paired raw package.', '',
                        'Even population-weight or single-F0 normalized inference consumes the',
                        'normalization calibration:20 long-term plus0/4/8/20 current trials. Separate',
                        'source-only raw controls consume none. Normalized inputs do not establish',
                        'raw-ADC quality or F8 routing. Full document F0-F9, native device efficacy,',
                        'cross-day/re-donning and normalized-candidate GUI integration remain open.', '',
                        '[Matched normalization results](../benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_RESULTS.json)',
                        'and [independent normalization/package acceptance](SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json)',
                        'bind this developmental comparison to its frozen implementation.']
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
