"""Copy inspectable small results; retain run IDs and never copy raw signals."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


RUNS = (
    'feature_bank_wearing_session_validation_20260916',
    'feature_bank_wearing_session_final_20260916',
    'feature_bank_unibo_sequence_incremental_validation_20260916',
    'feature_bank_unibo_sequence_incremental_final_20260916',
    'feature_bank_unibo_sequence_temporal_validation_20260916',
    'feature_bank_force_core_temperature_validation_20260916',
    'feature_bank_force_core_temperature_final_20260916',
    'feature_bank_manus_core_temperature_validation_20260916',
    'feature_bank_manus_core_temperature_final_20260916',
    'feature_bank_force_core_probability_source_20260916',
    'feature_bank_manus_core_probability_source_20260916',
    'feature_bank_manus_concat_core_calibration_validation_20260916',
    'feature_bank_manus_concat_core_calibration_final_20260916',
    'feature_bank_manus_concat_core_source_20260916',
    'feature_bank_manus_concat_core_validation_20260916',
    'feature_bank_manus_concat_core_final_20260916',
    'feature_bank_force_concat_core_calibration_validation_20260915',
    'feature_bank_force_concat_core_calibration_final_20260915',
    'feature_bank_force_concat_core_source_20260915',
    'feature_bank_force_concat_core_validation_20260915',
    'feature_bank_force_concat_core_final_20260915',
    'feature_bank_unibo_temporal_complementarity_validation_20260915',
    'feature_bank_force_quality_validation_20260915',
    'feature_bank_force_quality_final_20260915',
    'feature_bank_epn_core_incremental_validation_20260915',
    'feature_bank_epn_core_incremental_final_20260915',
    'feature_bank_epn_core_incremental_20260915',
    'feature_bank_epn_relative_spectrum_source_20260915',
    'feature_bank_epn_relative_spectrum_validation_20260915',
    'feature_bank_epn_relative_spectrum_final_20260915',
    'feature_bank_manus_session_blend_validation_20260915_v2',
    'feature_bank_manus_session_blend_final_20260915_v2',
    'feature_bank_manus_selected_validation_20260915',
    'feature_bank_manus_selected_final_20260915',
    'feature_bank_epn_selected_validation_20260915',
    'feature_bank_epn_selected_final_20260915',
    'feature_bank_manus_reliability_selection_20260915',
    'feature_bank_epn_reliability_selection_20260915',
    'feature_bank_manus_calibrated_quality_validation_20260915',
    'feature_bank_manus_calibrated_quality_final_20260915',
    'feature_bank_unibo_validated_reuse_validation_20260915',
    'feature_bank_wearing_quality_roles_validation_20260915',
    'feature_bank_wearing_quality_roles_final_20260915',
    'feature_bank_load_position_quality_roles_validation_20260915',
    'feature_bank_load_position_quality_roles_final_20260915',
    'feature_bank_unibo_quality_roles_validation_20260915',
    'feature_bank_unibo_quality_roles_final_20260915',
    'feature_bank_manus_quality_roles_validation_20260915',
    'feature_bank_manus_quality_roles_final_20260915',
    'feature_bank_load_position_full_validation_20260915',
    'feature_bank_load_position_full_final_20260915',
    'feature_bank_unibo_full_source_20260915',
    'feature_bank_unibo_full_fusion_validation_20260915',
    'feature_bank_unibo_full_fusion_final_20260915',
    'feature_bank_wearing_full_fusion_validation_20260915',
    'feature_bank_wearing_full_fusion_final_20260915',
    'feature_bank_manus_nested_oof_20260915',
    'feature_bank_manus_probability_validation_20260915',
    'feature_bank_manus_probability_final_20260915',
    'feature_bank_force_condition_validation_20260915',
    'feature_bank_force_condition_final_20260915',
    'feature_bank_epn_probability_validation_20260915',
    'feature_bank_epn_probability_final_20260915',
    'feature_bank_epn_nested_oof_20260915',
    'feature_bank_force_probability_validation_20260915',
    'feature_bank_force_probability_final_20260915',
    'feature_bank_force_nested_oof_20260915',
    'feature_bank_force_full_fusion_validation_20260915',
    'feature_bank_force_full_fusion_final_20260915',
    'feature_bank_force_audited_validation_20260915',
    'feature_bank_force_selection_audited_20260915',
    'feature_bank_force_product_validation_20260915',
    'feature_bank_force_product_final_20260915',
    'feature_bank_epn612_trial_validation_20260915',
    'feature_bank_epn612_trial_calibration_validation_20260915',
    'feature_bank_epn612_trial_finetune_validation_20260915',
    'feature_bank_epn612_trial_finetune_final_20260915',
    'feature_bank_manus_validation_20260915',
    'feature_bank_manus_final_20260915',
    'feature_bank_manus_calibration_validation_20260915',
    'feature_bank_manus_calibration_final_20260915',
    'feature_bank_electrode_shift_audited_validation_20260915',
    'feature_bank_electrode_shift_audited_final_20260915',
    'feature_bank_quality_validation_20260915',
    'feature_bank_unibo_audited_validation_20260915',
    'feature_bank_unibo_audited_final_20260915',
    'feature_bank_emg_fmg_audited_validation_20260915',
    'feature_bank_emg_fmg_audited_final_20260915',
    'feature_bank_fusion_validation_20260915',
    'feature_bank_fusion_final_20260915',
    'feature_bank_session_context_validation_20260915',
    'feature_bank_session_context_final_20260915',
    'feature_bank_normalization_validation_20260915',
    'feature_bank_normalization_final_20260915',
    'feature_bank_template_validation_20260915_v2',
    'feature_bank_template_final_20260915',
    'feature_bank_epn_selection_20260915',
    'feature_bank_force_zero_validation_20260915',
    'feature_bank_force_zero_final_20260915',
    'feature_bank_force_final_reference_20260915',
    'feature_bank_full_fusion_validation_20260915',
    'feature_bank_full_fusion_final_20260915',
    'feature_bank_manus_full_fusion_validation_20260915_v2',
    'feature_bank_manus_full_fusion_final_20260915_v2',
    'feature_bank_manus_calibration_diagnostics_validation_20260915',
    'feature_bank_manus_calibration_diagnostics_final_20260915',
    'feature_bank_epn_calibration_diagnostics_validation_20260915_v2',
    'feature_bank_epn_calibration_diagnostics_final_20260915_v2',
)
ARTIFACTS = ('feature_family_results.csv', 'conditional_incremental.csv', 'interaction_results.csv',
             'error_complementarity.csv', 'calibration_curve.csv', 'ablation_full_bank.csv',
             'per_family_calibration_gain.csv', 'cross_session_family_diagnostics.csv',
             'cross_user_family_diagnostics.csv', 'cross_day_posture_diagnostics.csv',
             'wearing_family_diagnostics.csv', 'load_position_family_diagnostics.csv',
             'anchor_variation_diagnostics.csv', 'force_worst_condition.csv')


def run_directory(root,run,local_root=None):
    candidates=[base/run for base in (root,local_root) if base is not None and (base/run).is_dir()]
    if len(candidates)>1:raise ValueError(f'Ambiguous source run: {run}')
    return candidates[0] if candidates else root/run


def consolidate(root: Path, output: Path, local_root=None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name in ARTIFACTS:
        rows = []
        for run in RUNS:
            directory=run_directory(root,run,local_root)
            source = directory / name
            if not source.exists():
                continue
            data = source.read_bytes()
            provenance.append({'run_id': run, 'artifact': name,
                               'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            if directory.parent!=root:
                provenance[-1]['source_root']=str(directory.parent.resolve())
            with source.open(encoding='utf-8-sig', newline='') as handle:
                rows.extend({'run_id': run, **row} for row in csv.DictReader(handle))
        if not rows:
            raise ValueError(f'No evidence for {name}')
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifests = output / 'manifests'
    manifests.mkdir(exist_ok=True)
    for run in RUNS:
        for source in run_directory(root,run,local_root).glob('*.json'):
            (manifests / f'{run}__{source.name}').write_bytes(source.read_bytes())
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    source=root/'feature_bank_force_audited_validation_20260915/family_diagnostics.csv'
    (output/'family_diagnostics.csv').write_bytes(source.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--local-root',type=Path)
    args = parser.parse_args()
    consolidate(args.root, args.output,args.local_root)
