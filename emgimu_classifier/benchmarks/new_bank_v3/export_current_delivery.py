"""Add current V3 native diagnostics to canonical tables without rewriting prior releases."""
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, log_loss
from emgimu.feature_bank.screening import expected_calibration_error

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'feature_bank/delivery'
OUT = BASE/'new_bank_v3'
METRICS = ('macro_f1', 'accuracy', 'log_loss', 'brier', 'ece')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, prediction, probability, classes):
    y = np.asarray(y); prediction = np.asarray(prediction)
    result = {'macro_f1': float(f1_score(y, prediction, labels=np.arange(classes), average='macro', zero_division=0)),
              'accuracy': float(np.mean(y == prediction)), 'log_loss': 'N/A', 'brier': 'N/A', 'ece': 'N/A'}
    if probability is not None:
        p = np.asarray(probability)
        result.update(log_loss=float(log_loss(y, p, labels=np.arange(classes))),
                      brier=float(np.mean((p-np.eye(classes)[y])**2)),
                      ece=expected_calibration_error(y, p, np.ones(len(y))))
    per_class = {}
    for c in range(classes):
        support = int(np.sum(y == c)); predicted = int(np.sum(prediction == c))
        tp = int(np.sum((y == c) & (prediction == c)))
        per_class[str(c)] = {'support':support,'predicted':predicted,'true_positives':tp,
            'false_positives':predicted-tp,'false_negatives':support-tp,
            'precision':tp/predicted if predicted else None,
            'recall':tp/support if support else None,
            'f1':2*tp/(support+predicted) if support else None}
    result['class_metrics_json'] = json.dumps(per_class,sort_keys=True)
    return result


def write(name, rows, fields):
    if not rows:
        raise AssertionError('Empty current V3 delivery table: '+name)
    with (OUT/name).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    family = []; incremental = []; errors = []; curve = []; eligibility = []; boundaries = []; ablations = []
    def read(name):
        path = HERE/name; sources[path.relative_to(ROOT).as_posix()] = sha(path)
        value = json.loads(path.read_text(encoding='utf8'))
        if value.get('default_promoted', False):
            raise AssertionError('Exporter does not authorize default promotion')
        return value
    def add_group(run, source, dataset, subject, domain, shots, condition, arms, y, classes, notes, concatenated_pairs=(), comparison_pairs=None):
        context = {'run_id': run, 'source_artifact': source, 'source_sha256': sources[source],
                   'dataset': dataset, 'subject': str(subject), 'session/domain': domain,
                   'condition': condition, 'calibration_budget': shots,
                   'evaluation_trials': len(y),
                   'metadata_notes_json': json.dumps(notes, sort_keys=True)}
        statistics = {}
        for name, (prediction, probability) in arms.items():
            s = score(y, prediction, probability, classes); statistics[name] = s
            family.append({**context, 'feature_family': name, 'calibration_budget': shots, **s})
            curve.append({**context, 'feature_bank': name, 'shots_per_class': shots, 'method': run,
                          'supported': True, 'macro_f1': s['macro_f1'], 'log_loss': s['log_loss']})
        for a, b in (combinations(arms, 2) if comparison_pairs is None else comparison_pairs):
            pa = np.asarray(arms[a][0]); pb = np.asarray(arms[b][0]); truth = np.asarray(y)
            ea = pa != truth; eb = pb != truth
            corr = float(np.corrcoef(ea.astype(float), eb.astype(float))[0, 1]) if np.std(ea) > 0 and np.std(eb) > 0 else 'N/A'
            errors.append({**context, 'family_a': a, 'family_b': b, 'error_correlation': corr,
                           'correlation_status': 'undefined_constant_error_vector' if corr == 'N/A' else 'defined',
                           'disagreement_rate': float(np.mean(pa != pb)),
                           'a_correct_b_wrong': int(np.sum(~ea & eb)), 'a_wrong_b_correct': int(np.sum(ea & ~eb)),
                           'a_correct_b_wrong_probability':float(np.mean(~ea & eb)),
                           'a_wrong_b_correct_probability':float(np.mean(ea & ~eb))})
            delta = {}
            for metric, output in [('log_loss', 'delta_logloss'), ('macro_f1', 'delta_macro_f1'), ('brier', 'delta_brier')]:
                x, z = statistics[a][metric], statistics[b][metric]
                delta[output] = ((z-x if metric == 'macro_f1' else x-z)
                                 if x != 'N/A' and z != 'N/A' else 'N/A')
            incremental.append({**context, 'core_bank': a, 'added_family': b,
                                'comparison_kind': ('source_refit_concatenated_group_increment' if (a,b) in concatenated_pairs else 'paired_alternative_not_concatenated_increment'), **delta})
    for name, run_id in [('F8_ROUTER_MANUS_V1_RESULTS.json', 'f8_router_manus_v1'),
                         ('F8_CALIBRATED_MANUS_V2_RESULTS.json', 'f8_calibrated_manus_v2')]:
        f8 = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
        for phase in ('validation', 'descriptive_final'):
            for shots in (1, 2):
                selected = [b for b in f8['blocks'] if b['phase'] == phase and b['shots'] == shots]
                for user in ['ALL']+sorted({b['user'] for b in selected}):
                    group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
                    y = np.concatenate([b['labels'] for b in group])
                    arms = {}
                    for arm in ('TD24', 'uniform', 'F8'):
                        p = np.concatenate([b['probabilities'][arm] for b in group]); arms[arm] = (p.argmax(axis=1), p)
                    add_group(run_id, source, 'sEMG-MANUS', user, phase, shots, 'frozen_evaluation_trials',
                              arms, y, 6, {'classes': 6, 'ontology': 'six native finger-flexion gestures',
                              'scope': f8['scope'], 'target_calibration_disjoint': True,
                              'source_oof_temperature_calibrated': run_id.endswith('v2'), 'TD24_is_document_F0': False})
    name = 'MAHALANOBIS_EPN_BUDGET_V1_RESULTS.json'; md = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    for block in md['blocks']:
        eligibility.append({'dataset': 'EPN612', 'subject': block['user'], 'feature_family': block['family'],
                            'shots_per_class': block['shots'], 'dimension': block['dimension'],
                            'required_independent_trials': block['required_trials_per_class'],
                            'eligible': block['mahalanobis_eligible'],
                            'reason': 'eligible' if block['mahalanobis_eligible'] else block['ineligible_reason'],
                            'source_artifact': source, 'source_sha256': sources[source]})
    for feature in ('pattern', 'TD24', 'SPD'):
        for shots in (1, 2, 5, 10, 20):
            selected = [b for b in md['blocks'] if b['family'] == feature and b['shots'] == shots]
            for user in ['ALL']+sorted({b['user'] for b in selected}):
                group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
                y = np.concatenate([b['labels'] for b in group]); arms = {}
                for arm in ('euclidean', 'mahalanobis'):
                    if arm == 'mahalanobis' and not all(b['mahalanobis_eligible'] for b in group):
                        continue
                    p = np.concatenate([b[f'{arm}_probability'] for b in group]); arms[feature+'_'+arm] = (p.argmax(axis=1), p)
                add_group('mahalanobis_epn_budget_v1', source, 'EPN612', user, 'descriptive_users22_31', shots,
                          'fixed_evaluation_trials_across_budgets', arms, y, 6,
                          {'classes': 6, 'ontology': 'EPN612 native six classes', 'dimension': group[0]['dimension'],
                           'scope': md['scope'], 'independent_trial_guard': True})
    name = 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json'; holdout = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    for block in holdout['blocks']:
        eligibility.append({'dataset': 'EPN612', 'subject': block['user'], 'feature_family': 'pattern',
                            'shots_per_class': block['shots'], 'dimension': 8,
                            'required_independent_trials': 10, 'eligible': True, 'reason': 'eligible',
                            'source_artifact': source, 'source_sha256': sources[source]})
    for shots in (10, 20):
        selected = [b for b in holdout['blocks'] if b['shots'] == shots]
        for user in ['ALL'] + sorted({b['user'] for b in selected}):
            group = selected if user == 'ALL' else [b for b in selected if b['user'] == user]
            y = np.concatenate([b['labels'] for b in group]); arms = {}
            for arm in ('euclidean', 'mahalanobis'):
                p = np.concatenate([b['probabilities'][arm] for b in group])
                arms['pattern_' + arm] = (p.argmax(axis=1), p)
            add_group('mahalanobis_epn_holdout_v2', source, 'EPN612', user, 'precommitted_users32_41', shots,
                      'fixed_evaluation_trials_across_budgets', arms, y, 6,
                      {'classes': 6, 'ontology': 'EPN612 native six classes', 'dimension': 8,
                       'scope': holdout['scope'], 'independent_trial_guard': True,
                       'native_score_brier_normalization': 'mean across trials and classes',
                       'canonical_export_brier_normalization': 'mean across trials and classes'})
    name = 'DETECTED_G5_UNIBO_V1_RESULTS.json'; g5 = read(name); source = (HERE/name).relative_to(ROOT).as_posix()
    supported = [e for e in g5['events'] if e['reference_label'] is not None]
    for boundary in ('detected', 'matched_oracle'):
        for user in ['ALL']+sorted({e['user'] for e in supported}):
            group = supported if user == 'ALL' else [e for e in supported if e['user'] == user]
            y = np.array([e['reference_label'] for e in group]); oracle = boundary == 'matched_oracle'
            p = np.array([e['g5_oracle_probability' if oracle else 'g5_probability'] for e in group])
            dtw = np.array([e['oracle_prediction' if oracle else 'prediction'] for e in group])
            # Keep prediction-only DTW explicitly probability-unavailable.
            arms = {'DTW': (dtw, None), 'G5': (p.argmax(axis=1), p)}
            add_group('detected_g5_unibo_v1', source, 'UniBo4', user, 'Day6_continuous', 0,
                      boundary+'_supported_matched_active', arms, y, 4,
                      {'classes': 4, 'no_rest_ground_truth': True, 'scope': g5['scope'],
                       'DTW_probability': 'unavailable; distances are not calibrated probabilities',
                       'calibration_budget': 'zero target-Day6 shots; source Days1-5 model/calibration retained'})
    for filename in ('AUTONOMOUS_UNIBO_V1_RESULTS.json', 'AUTONOMOUS_CONTINUOUS_UNIBO_V1_RESULTS.json'):
        result = read(filename); path = (HERE/filename).relative_to(ROOT).as_posix()
        t = result['totals']
        boundaries.append({'run_id': filename.removesuffix('_RESULTS.json'), 'dataset': 'UniBo4',
                           'references': t['references'], 'detections': t['detections'], 'matched': t['matched'],
                           'precision': t['precision'], 'recall': t['recall'],
                           'onset_mae_s': t['onset_mae_s'] if t['onset_mae_s'] is not None else 'N/A',
                           'offset_mae_s': t['offset_mae_s'] if t['offset_mae_s'] is not None else 'N/A',
                           'source_artifact': path, 'source_sha256': sources[path], 'scope': result.get('scope', '')})
    bank_name = 'EMG_F0_F7_BANK_V1_RESULTS.json'; bank = read(bank_name)
    bank_source = (HERE/bank_name).relative_to(ROOT).as_posix()
    for shots in (0,1,2,5):
        for user in ['ALL']+[b['user'] for b in bank['blocks']]:
            blocks = bank['blocks'] if user == 'ALL' else [b for b in bank['blocks'] if b['user'] == user]
            y = np.concatenate([b['labels'] for b in blocks])
            base = np.concatenate([b['F0_probabilities'] for b in blocks])
            probabilities = {'F0':base}
            if shots:
                anchor = np.concatenate([next(c['F7_probabilities'] for c in b['calibrations'] if c['shots']==shots) for b in blocks])
                probabilities.update(F7=anchor,F0_F7=(base+anchor)/2,F0_uniform=(base+1/6)/2)
            notes = {'classes':6,'scope':bank['scope'],'trial_mean_feature_dimension':48,
                     'source_only_F0':True,'no_IMU_features':True,'reserved_trials_per_user':30,
                     'actual_target_calibration_trials_per_user':{'F0':0,'F0_uniform':0,'F7':6*shots,'F0_F7':6*shots},
                     'calibration_budget_is_scenario_not_F0_used_shots':True,
                     'fixed_evaluation_trial_set_across_budgets':True,
                     'ablation_is_two_provider_probability_removal_not_whole_document_bank':True}
            arms = {name:(q.argmax(1),q) for name,q in probabilities.items()}
            add_group('emg_f0_f7_bank_v1',bank_source,'EPN612',user,'new_cohort_42_51',shots,
                      'fixed_reserved5_per_class',arms,y,6,notes)
            if shots:
                full = score(y,probabilities['F0_F7'].argmax(1),probabilities['F0_F7'],6)
                for removed,remaining in [('F0','F7'),('F7','F0')]:
                    alternative = score(y,probabilities[remaining].argmax(1),probabilities[remaining],6)
                    ablations.append({'run_id':'emg_f0_f7_bank_v1','source_artifact':bank_source,
                        'source_sha256':sources[bank_source],'dataset':'EPN612','subject':str(user),
                        'session/domain':'new_cohort_42_51','condition':'fixed_reserved5_per_class',
                        'calibration_budget':shots,'evaluation_trials':len(y),'metadata_notes_json':json.dumps(notes,sort_keys=True),
                        'full_bank':'F0_F7','removed_provider':removed,'remaining_bank':remaining,
                        'full_target_calibration_trials_per_user':6*shots,
                        'remaining_target_calibration_trials_per_user':6*shots if remaining=='F7' else 0,
                        **{'full_'+k:full[k] for k in METRICS},**{'remaining_'+k:alternative[k] for k in METRICS},
                        'delta_logloss':alternative['log_loss']-full['log_loss'],
                        'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                        'delta_brier':alternative['brier']-full['brier']})
    window_name = 'EMG_WINDOW_BANK_V1_RESULTS.json'; window_bank = read(window_name)
    window_source = (HERE/window_name).relative_to(ROOT).as_posix()
    concatenated_pairs = [('F0','F0_plus_'+g) for g in window_bank['group_feature_names'] if g!='F0']
    for user in ['ALL']+[b['user'] for b in window_bank['blocks']]:
        blocks = window_bank['blocks'] if user=='ALL' else [b for b in window_bank['blocks'] if b['user']==user]
        y = np.concatenate([b['labels'] for b in blocks])
        probabilities = {arm:np.concatenate([b['probabilities'][arm] for b in blocks]) for arm in window_bank['source_models']}
        arms = {arm:(q.argmax(1),q) for arm,q in probabilities.items()}
        notes = {'no_IMU_features':True,'actual_target_calibration_trials_per_user':0,
                 'source_refit_for_each_declared_composition':True,
                 'ablation_is_six_declared_window_groups_not_whole_document_bank':True,
                 'scope':window_bank['scope']}
        add_group('emg_window_bank_v1',window_source,'EPN612',user,'new_users52-61',0,
                  'native_cue_aligned_trials',arms,y,6,notes,concatenated_pairs=concatenated_pairs)
        full = score(y,*arms['window_bank'],6)
        for removed in window_bank['group_feature_names']:
            remaining = 'window_bank_minus_'+removed
            alternative = score(y,*arms[remaining],6)
            ablations.append({'run_id':'emg_window_bank_v1','source_artifact':window_source,
                'source_sha256':sources[window_source],'dataset':'EPN612','subject':str(user),
                'session/domain':'new_users52-61','condition':'native_cue_aligned_trials','calibration_budget':0,
                'evaluation_trials':len(y),'metadata_notes_json':json.dumps(notes,sort_keys=True),
                'full_bank':'window_bank','removed_provider':removed,'remaining_bank':remaining,
                'full_target_calibration_trials_per_user':0,'remaining_target_calibration_trials_per_user':0,
                **{'full_'+k:full[k] for k in METRICS},**{'remaining_'+k:alternative[k] for k in METRICS},
                'delta_logloss':alternative['log_loss']-full['log_loss'],
                'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                'delta_brier':alternative['brier']-full['brier']})
    fusion_name='EMG_CALIBRATED_FUSION_V1_RESULTS.json'; fusion=read(fusion_name)
    fusion_source=(HERE/fusion_name).relative_to(ROOT).as_posix()
    provider_names=list(fusion['source_models'])
    comparisons=list(combinations(provider_names,2))+[('F0','uniform_bank'),('F0','reliability_bank'),('uniform_bank','reliability_bank')]
    comparisons += [('reliability_bank','reliability_minus_'+name) for name in provider_names]
    for shots in (0,1,2,5):
        for user in ['ALL']+[b['user'] for b in fusion['blocks']]:
            blocks=fusion['blocks'] if user=='ALL' else [b for b in fusion['blocks'] if b['user']==user]
            entries=[next(c for c in b['calibrations'] if c['shots']==shots) for b in blocks]
            y=np.concatenate([b['labels'] for b in blocks])
            probabilities={arm:np.concatenate([c['probabilities'][arm] for c in entries]) for arm in entries[0]['probabilities']}
            arms={arm:(q.argmax(1),q) for arm,q in probabilities.items()}
            costs={arm:6*shots if arm.startswith('reliability') else 0 for arm in arms}
            notes={'no_IMU_features':True,'source_user_OOF_probability_temperatures':True,
                'actual_target_calibration_trials_per_user':costs,
                'calibration_updates_only_weights_not_provider_models':True,
                'ablation_is_six_provider_frozen_weight_removal_not_whole_document_bank':True,
                'scope':fusion['scope']}
            add_group('emg_calibrated_fusion_v1',fusion_source,'EPN612',user,'new_users62-71',shots,
                'native_cue_aligned_trials',arms,y,6,notes,comparison_pairs=comparisons)
            full=score(y,*arms['reliability_bank'],6)
            for removed in provider_names:
                remaining='reliability_minus_'+removed; alternative=score(y,*arms[remaining],6)
                ablations.append({'run_id':'emg_calibrated_fusion_v1','source_artifact':fusion_source,
                    'source_sha256':sources[fusion_source],'dataset':'EPN612','subject':str(user),
                    'session/domain':'new_users62-71','condition':'native_cue_aligned_trials','calibration_budget':shots,
                    'evaluation_trials':len(y),'metadata_notes_json':json.dumps(notes,sort_keys=True),
                    'full_bank':'reliability_bank','removed_provider':removed,'remaining_bank':remaining,
                    'full_target_calibration_trials_per_user':6*shots,'remaining_target_calibration_trials_per_user':6*shots,
                    **{'full_'+k:full[k] for k in METRICS},**{'remaining_'+k:alternative[k] for k in METRICS},
                    'delta_logloss':alternative['log_loss']-full['log_loss'],
                    'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                    'delta_brier':alternative['brier']-full['brier']})
    song_path=ROOT/'benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_RESULTS.json'
    song_source=song_path.relative_to(ROOT).as_posix();sources[song_source]=sha(song_path)
    song=json.loads(song_path.read_text(encoding='utf8'))
    provider_path=ROOT/'feature_bank/SONG_PERSONAL_SESSION_PROVIDER_READOUTS_V1.npz'
    sources[provider_path.relative_to(ROOT).as_posix()]=sha(provider_path)
    prediction_path=ROOT/song['prediction_path'];sources[prediction_path.relative_to(ROOT).as_posix()]=sha(prediction_path)
    with prediction_path.open(encoding='utf8',newline='') as stream:song_rows=list(csv.DictReader(stream))
    with np.load(provider_path,allow_pickle=False) as native:
        classes=native['class_names'].tolist();y=np.array([classes.index(c) for c in native['labels']])
        groups=list(song['source_dimensions'])
        comparisons=list(combinations(groups,2))+[('F0',a) for a in ('population','uniform','personal','session')]
        comparisons += [('personal','session')]+[('session','session_minus_'+g) for g in groups]
        for shots in (0,1,2,5):
            probabilities={g:native[g] for g in groups}
            for cell in song['cells']:
                if cell['shots']!=shots:continue
                rows=[r for r in song_rows if int(r['shots'])==shots and r['arm']==cell['arm']]
                if [r['trial_id'] for r in rows]!=native['trial_ids'].tolist():raise ValueError('Song canonical trial axis differs')
                probabilities[cell['arm']]=np.array([[float(r['p_'+c]) for c in classes] for r in rows])
            arms={k:(q.argmax(1),q) for k,q in probabilities.items()}
            costs={k:(20+4*shots if k.startswith('session') else 20 if k=='personal' else 0) for k in arms}
            notes={'class_names':classes,'same_person_day':True,'long_term_session':'S03','current_session':'S04',
                'actual_total_target_calibration_trials':costs,'long_term_personal_trials':20,
                'current_session_trials':4*shots,'source_only_recorded_session_OOF_probability_calibration':True,
                'provider_removal_is_frozen_weight_renormalization':True,
                'F7_F8_F9_are_separate_context_not_fused_classifiers':True,'scope':song['scope']}
            add_group('song_personal_session_v1',song_source,'Song_real8','Song','S04_recording',shots,
                'same_person_day_cued_stable_native_trials',arms,y,4,notes,comparison_pairs=comparisons)
            full=score(y,*arms['session'],4)
            for removed in groups:
                remaining='session_minus_'+removed;alternative=score(y,*arms[remaining],4)
                ablations.append({'run_id':'song_personal_session_v1','source_artifact':song_source,
                    'source_sha256':sources[song_source],'dataset':'Song_real8','subject':'Song',
                    'session/domain':'S04_recording','condition':'same_person_day_cued_stable_native_trials',
                    'calibration_budget':shots,'evaluation_trials':len(y),'metadata_notes_json':json.dumps(notes,sort_keys=True),
                    'full_bank':'session','removed_provider':removed,'remaining_bank':remaining,
                    'full_target_calibration_trials_per_user':20+4*shots,'remaining_target_calibration_trials_per_user':20+4*shots,
                    **{'full_'+k:full[k] for k in METRICS},**{'remaining_'+k:alternative[k] for k in METRICS},
                    'delta_logloss':alternative['log_loss']-full['log_loss'],
                    'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                    'delta_brier':alternative['brier']-full['brier']})
    matched_path=ROOT/'benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_RESULTS.json'
    matched_source=matched_path.relative_to(ROOT).as_posix();sources[matched_source]=sha(matched_path)
    matched=json.loads(matched_path.read_text(encoding='utf8'))
    matched_folder=matched_path.parent/'song_matched_normalization_v1'
    for path in (matched_folder/'predictions.csv',matched_folder/'readouts.npz'):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    with (matched_folder/'predictions.csv').open(encoding='utf8',newline='') as stream:matched_rows=list(csv.DictReader(stream))
    with np.load(matched_folder/'readouts.npz',allow_pickle=False) as native:
        classes=native['class_names'].tolist();y=np.array([classes.index(c) for c in native['labels']])
        groups=list(matched['source']['raw']['dimensions'])
        comparisons=[]
        for mode in ('raw','normalized'):
            comparisons += [(mode+'_'+a,mode+'_'+b) for a,b in list(combinations(groups,2))+
                [('F0',a) for a in ('population','personal','session')]+[('personal','session')]+
                [('session','session_minus_'+g) for g in groups]]
        comparisons += [('raw_'+arm,'normalized_'+arm) for arm in ('F0','population','personal','session')]
        for shots in (0,1,2,5):
            probabilities={mode+'_'+g:native[f'{mode}_{shots}_{g}'] for mode in ('raw','normalized') for g in groups}
            for cell in matched['cells']:
                if cell['shots']!=shots:continue
                rows=[r for r in matched_rows if r['mode']==cell['mode'] and int(r['shots'])==shots and r['arm']==cell['arm']]
                if [r['trial_id'] for r in rows]!=native['trial_ids'].tolist():raise ValueError('Matched normalization trial axis differs')
                probabilities[cell['mode']+'_'+cell['arm']]=np.array([[float(r['p_'+c]) for c in classes] for r in rows])
            arms={name:(q.argmax(1),q) for name,q in probabilities.items()}
            costs={}
            for name in arms:
                mode,arm=name.split('_',1)
                costs[name]=(20+4*shots if (mode=='normalized' and arm!='personal') or arm.startswith('session') else
                             20 if arm=='personal' else 0)
            notes={'class_names':classes,'actual_total_target_calibration_trials':costs,
                'source_model_fit_trials_common_to_both_domains':245,'source_normalization_reserved_trials':40,
                'long_term_personal_trials':20,'current_session_trials':4*shots,'same_person_day':True,
                'source_representations_scalers_classifiers_and_probability_temperatures_refitted_per_domain':True,
                'normalized_models_require_calibration_even_for_population_or_F0_prediction':True,
                'provider_removal_is_frozen_weight_renormalization':True,'scope':matched['scope']}
            add_group('song_matched_normalization_v1',matched_source,'Song_real8','Song','S04_matched_input_domains',shots,
                'same_person_day_cued_stable_native_trials',arms,y,4,notes,comparison_pairs=comparisons)
            for mode in ('raw','normalized'):
                full_name=mode+'_session';full=score(y,*arms[full_name],4)
                for removed in groups:
                    remaining=mode+'_session_minus_'+removed;alternative=score(y,*arms[remaining],4)
                    ablations.append({'run_id':'song_matched_normalization_v1','source_artifact':matched_source,
                        'source_sha256':sources[matched_source],'dataset':'Song_real8','subject':'Song',
                        'session/domain':'S04_matched_input_domains','condition':'same_person_day_cued_stable_native_trials',
                        'calibration_budget':shots,'evaluation_trials':len(y),'metadata_notes_json':json.dumps(notes,sort_keys=True),
                        'full_bank':full_name,'removed_provider':removed,'remaining_bank':remaining,
                        'full_target_calibration_trials_per_user':20+4*shots,'remaining_target_calibration_trials_per_user':20+4*shots,
                        **{'full_'+k:full[k] for k in METRICS},**{'remaining_'+k:alternative[k] for k in METRICS},
                        'delta_logloss':alternative['log_loss']-full['log_loss'],
                        'delta_macro_f1':full['macro_f1']-alternative['macro_f1'],
                        'delta_brier':alternative['brier']-full['brier']})
    burden_name = 'MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json'
    burden = read(burden_name)
    burden_source = (HERE/burden_name).relative_to(ROOT).as_posix()
    burden_rows = [{**row, 'run_id':'MAHALANOBIS_EPN_HOLDOUT_V2',
                    'feature_bank':'euclidean_and_mahalanobis_shared_calibration',
                    'source_artifact': burden_source, 'source_sha256': sources[burden_source]}
                   for row in burden['records']]
    current_burden_name='EMG_CALIBRATION_BURDEN_V1.json';current_burden=read(current_burden_name)
    current_burden_source=(HERE/current_burden_name).relative_to(ROOT).as_posix()
    burden_rows += [{**row,'source_artifact':current_burden_source,'source_sha256':sources[current_burden_source]} for row in current_burden['records']]
    burden_fields=list(dict.fromkeys(key for row in burden_rows for key in row))
    burden_rows=[{field:row.get(field,'N/A') for field in burden_fields} for row in burden_rows]
    continuous_name = 'ROAM_CAUSAL_WINDOW_V1_RESULTS.json'
    continuous = read(continuous_name)
    continuous_source = (HERE/continuous_name).relative_to(ROOT).as_posix()
    emissions_path = HERE/'ROAM_CAUSAL_WINDOW_V1_EMISSIONS.csv'
    sources[emissions_path.relative_to(ROOT).as_posix()] = sha(emissions_path)
    emitted = {}
    with emissions_path.open(encoding='utf8', newline='') as stream:
        for row in csv.DictReader(stream):
            emitted.setdefault(row['native_file'], []).append(row)
    continuous_rows = []; transition_rows = []
    for record in continuous['records']:
        truth = np.empty(record['samples'], dtype=int)
        for a,b,label in record['truth_rle']:
            truth[a:b] = label
        events = emitted[record['native_file']]
        probability = np.asarray([[float(e[f'p_{c}']) for c in range(3)] for e in events])
        dense = np.repeat(probability, 10, axis=0)[:record['samples']-39]
        metrics = score(truth[39:], dense.argmax(1), dense, 3)
        context = {'run_id': 'roam_causal_window_v1', 'source_artifact': continuous_source,
                   'source_sha256': sources[continuous_source], 'dataset': 'ROAM_EMG',
                   'subject': record['user'], 'session/domain': record['phase'],
                   'condition': record['posture'], 'native_file': record['native_file']}
        continuous_rows.append({**context, 'feature_family': 'F0v2_window48', 'calibration_budget': 0,
             'evaluation_unit': 'nominal_sample', 'native_recordings': 1,
             'scored_samples': record['samples']-39, 'unknown_warmup_samples': 39, **metrics,
             'metadata_notes_json': json.dumps({'classes':3, 'rate_hz':200, 'window_samples':40,
                 'hop_samples':10, 'no_future_samples':True, 'source_only_rest_thresholds':True,
                 'brier_normalization':'mean across scored nominal samples and classes',
                 'scope':continuous['scope'], 'sample_count_is_not_calibration_trial_count':True}, sort_keys=True)})
        diagnostic = record['transition_hold']
        transition_rows.append({**context, 'annotated_transitions': diagnostic['annotated_transitions'],
              'eligible_transitions': diagnostic['eligible_transitions'],
              'correct_transitions': diagnostic['correct_transitions'],
              'transition_hold_accuracy': diagnostic['transition_hold_accuracy'],
              'maintenance_switches': diagnostic['maintenance_switches'],
              'reaction_half_buffer_samples':100, 'nominal_sample_rate_hz':200,
              'scope':diagnostic['scope']})
    control_name = 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json'
    control = read(control_name)
    control_source = (HERE/control_name).relative_to(ROOT).as_posix()
    control_rows = []
    for record in control['records']:
        for arm, info in record['arms'].items():
            diagnostic = info['transition_hold']
            control_rows.append({'run_id':'roam_debounce_control_v1',
                'source_artifact':control_source, 'source_sha256':sources[control_source],
                'dataset':'ROAM_EMG', 'subject':record['user'], 'session/domain':record['phase'],
                'condition':record['posture'], 'native_file':record['native_file'], 'label_policy':arm,
                'evaluation_unit':'nominal_sample', 'samples':record['samples'],
                'shared_known_samples':record['shared_known_samples'],
                'unknown_samples':info['unknown_samples'], **info['shared_known_scores'],
                'full_record_accuracy_unknown_wrong':info['full_record_accuracy_unknown_wrong'],
                'eligible_transitions':diagnostic['eligible_transitions'],
                'correct_transitions':diagnostic['correct_transitions'],
                'maintenance_switches':diagnostic['maintenance_switches'],
                'log_loss':'N/A', 'brier':'N/A', 'ece':'N/A', 'scope':control['scope']})
    quality_path=ROOT/'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'
    quality_source=quality_path.relative_to(ROOT).as_posix();sources[quality_source]=sha(quality_path)
    quality=json.loads(quality_path.read_text(encoding='utf8'))
    quality_csv=quality_path.parent/'song_raw_quality_v1/predictions.csv'
    sources[quality_csv.relative_to(ROOT).as_posix()]=sha(quality_csv)
    quality_rows=[]
    for cell in quality['cells']:
        metrics=cell['gate_evaluation']['trial_balanced_metrics']
        quality_rows.append(dict(run_id='song_raw_quality_v1',source_artifact=quality_source,source_sha256=sources[quality_source],
            dataset='Song_real8',subject='Song',scenario=cell['scenario'],mode=cell['mode'],evaluation_trials=cell['trials'],
            long_term_calibration_trials=20,current_calibration_trials=20,rejected=cell['rejected'],coverage=cell['coverage'],
            accuracy_unknown_wrong=cell['accuracy_unknown_wrong'],accepted_accuracy=cell['accepted_accuracy'] if cell['accepted_accuracy'] is not None else 'N/A',
            fault_annotation=cell['fault_annotation'],fault_recall=metrics['fault_recall'] if metrics['fault_recall'] is not None else 'N/A',
            normal_false_rejection_rate='N/A',physical_validation_proven=False,default_promoted=False,
            scope='Raw pre-software-highpass synthetic faults; fallback probabilities are not accepted decisions; physical normal/fault truth unavailable.'))
    context_fields = ['run_id', 'source_artifact', 'source_sha256', 'dataset', 'subject', 'session/domain', 'condition', 'calibration_budget', 'evaluation_trials', 'metadata_notes_json']
    tables = {'ablation_full_bank.csv': (ablations, list(ablations[0])),
              'quality_gate.csv': (quality_rows, list(quality_rows[0])),
              'label_stability_control.csv': (control_rows, list(control_rows[0])),
              'continuous_recognition.csv': (continuous_rows, list(continuous_rows[0])),
              'transition_hold.csv': (transition_rows, list(transition_rows[0])),
              'feature_family_results.csv': (family, context_fields+['feature_family']+list(METRICS)+['class_metrics_json']),
              'conditional_incremental.csv': (incremental, context_fields+['core_bank', 'added_family', 'comparison_kind', 'delta_logloss', 'delta_macro_f1', 'delta_brier']),
              'error_complementarity.csv': (errors, context_fields+['family_a', 'family_b', 'error_correlation', 'correlation_status', 'disagreement_rate', 'a_correct_b_wrong', 'a_wrong_b_correct', 'a_correct_b_wrong_probability', 'a_wrong_b_correct_probability']),
              'calibration_curve.csv': (curve, context_fields+['feature_bank', 'shots_per_class', 'method', 'supported', 'macro_f1', 'log_loss']),
              'calibration_burden.csv': (burden_rows, list(burden_rows[0])),
              'budget_eligibility.csv': (eligibility, list(eligibility[0])), 'boundary_detection.csv': (boundaries, list(boundaries[0]))}
    for filename, (rows, fields) in tables.items():
        write(filename, rows, fields)
    manifest = {'schema': 'current_v3_canonical_delivery_v1', 'generator_sha256': sha(Path(__file__)),
                'source_sha256': sources,
                'brier_normalization': 'mean across trials and classes; not class-summed; compare only matched class ontologies',
                'calibration_burden_scope': 'Extracted window exposure differs from complete recording duration and physical session wall time; hardware times remain N/A.',
                'continuous_brier_normalization': 'mean across scored nominal samples and classes',
                'continuous_evaluation_scope': 'Separate nominal-sample records: source thresholds only, target warmup explicitly unknown, sample counts never treated as independent calibration trials. Transition-hold metric is not exact ReactEMG reproduction or hardware latency.',
                'class_metrics_scope': 'Per-class precision/recall/F1 with explicit support and prediction counts. Undefined precision or absent-ground-truth recall/F1 are null; no true-neutral claim on matched active-only UniBo trials.',
                'error_probability_scope': 'Paired correctness probabilities use the explicit shared evaluation-trial denominator. Undefined correlations are explained, never filled with zero.',
                'delta_convention': {'delta_logloss': 'base minus alternative; positive means improvement',
                                     'delta_brier': 'base minus alternative; positive means improvement',
                                     'delta_macro_f1': 'alternative minus base; positive means improvement'},
                'tables': {n: {'rows': len(rows), 'sha256': sha(OUT/n)} for n, (rows, _) in tables.items()},
                'boundary': 'Paired alternatives are labelled explicitly; actual source-refit concatenated group increments have a distinct comparison_kind. DTW has no probability metrics. Budget failures are in eligibility, not fake performance rows. Conditions/datasets/class ontologies cannot be pooled indiscriminately.',
                'default_promoted': False, 'completion_proven': False}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    index = {'schema': 'versioned_feature_bank_delivery_index_v1',
             'tables': {n: [n, 'new_bank_v3/'+n] for n in tables if (BASE/n).exists()},
             'additional_tables': ['new_bank_v3/boundary_detection.csv', 'new_bank_v3/budget_eligibility.csv', 'new_bank_v3/calibration_burden.csv', 'new_bank_v3/continuous_recognition.csv', 'new_bank_v3/transition_hold.csv', 'new_bank_v3/label_stability_control.csv', 'new_bank_v3/quality_gate.csv'],
             'full_bank_ablation': 'ablation_full_bank.csv', 'current_manifest': 'new_bank_v3/MANIFEST.json',
             'current_ablation_scope': 'new_bank_v3/ablation_full_bank.csv contains restricted F0/F7 removals, source-only six-window-group removals and OOF-calibrated six-provider frozen-weight removals; none is the document-wide bank. The base table remains unchanged.',
             'base_provenance': 'PROVENANCE_AUDIT.json',
             'base_table_sha256': {f.name: sha(f) for f in sorted(BASE.glob('*.csv'))},
             'v3_result_inventory_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(HERE.rglob('*.json'))
                                            if f.name.endswith('_RESULTS.json') or f.name == 'results.json'},
             'current_scientific_conclusions': {'path':'../CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json',
                 'sha256':sha(ROOT/'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json')},
             'current_regression_snapshot': {'path':'../CLASSIFIER_REGRESSION_20261008.json',
                 'sha256':sha(ROOT/'feature_bank/CLASSIFIER_REGRESSION_20261008.json')},
             'official_unibo_adapter_acceptance': {'path':'../OFFICIAL_UNIBO_ADAPTER_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/OFFICIAL_UNIBO_ADAPTER_ACCEPTANCE_V1.json')},
             'portable_emg_bank_acceptance': {'path':'../FROZEN_EMG_BANK_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json')},
             'portable_emg_bank': {'path':'../models/epn_emg_calibrated_bank_v1.pkl',
                 'sha256':sha(ROOT/'feature_bank/models/epn_emg_calibrated_bank_v1.pkl')},
             'song_f0_runtime_acceptance': {'path':'../SONG_F0_RUNTIME_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json')},
             'song_f0_runtime': {'path':'../models/song_f0_250hz_v1.pkl',
                 'sha256':sha(ROOT/'feature_bank/models/song_f0_250hz_v1.pkl')},
             'song_f0_stream_replay': {'path':'../../benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json',
                 'sha256':sha(ROOT/'benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json')},
             'song_gui_v2_acceptance': {'path':'../SONG_GUI_V2_ACCEPTANCE.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_GUI_V2_ACCEPTANCE.json')},
             'song_personal_session_acceptance': {'path':'../SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json')},
             'song_personal_gui_acceptance': {'path':'../SONG_PERSONAL_GUI_V1_ACCEPTANCE.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_PERSONAL_GUI_V1_ACCEPTANCE.json')},
             'song_matched_normalization_acceptance': {'path':'../SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json')},
             'song_raw_quality_acceptance': {'path':'../SONG_RAW_QUALITY_ACCEPTANCE_V1.json',
                 'sha256':sha(ROOT/'feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json')},
             'completion_proven': False}
    (BASE/'INDEX.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf8')
    print(json.dumps({n: len(rows) for n, (rows, _) in tables.items()}), flush=True)
    return manifest


if __name__ == '__main__':
    export()
