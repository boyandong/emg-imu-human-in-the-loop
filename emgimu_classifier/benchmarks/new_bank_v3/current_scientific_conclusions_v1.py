"""Current, versioned answers to the goal's eight scientific questions.

This synthesizes immutable experiments; no target tuning, refitting or device
claim is introduced. Historical reports remain independently identifiable.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'feature_bank/CURRENT_SCIENTIFIC_CONCLUSIONS_V1.json'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    sources = {}
    def read(name):
        path = HERE / name; sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf8'))
    stream = read('ROAM_CAUSAL_WINDOW_V1_RESULTS.json')
    debounce = read('ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json')
    fresh = read('F7_AFFINE_FRESH/results.json')
    guard = read('PUBLIC_DEFAULT_EXTENSION_AUDIT.json')
    epn = read('MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json')
    users = read('MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json')
    burden = read('MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json')
    raw_router = read('F8_ROUTER_MANUS_V1_RESULTS.json')
    calibrated_router = read('F8_CALIBRATED_MANUS_V2_RESULTS.json')
    def evidence(*names): return ['benchmarks/new_bank_v3/' + n for n in names]
    stream_cells = []
    for phase, arms in debounce['summaries'].items():
        raw, confirmed = arms['raw'], arms['confirmed']
        stream_cells.append({'phase': phase, 'recordings': raw['recordings'],
            'raw_shared_macro_f1': raw['equal_recording_shared_macro_f1'],
            'confirmed_shared_macro_f1': confirmed['equal_recording_shared_macro_f1'],
            'delta_shared_macro_f1': confirmed['equal_recording_shared_macro_f1'] - raw['equal_recording_shared_macro_f1'],
            'raw_hold_success': raw['transition_hold_accuracy'], 'confirmed_hold_success': confirmed['transition_hold_accuracy'],
            'eligible_transitions_per_arm': raw['eligible_transitions'],
            'raw_switches': raw['maintenance_switches'], 'confirmed_switches': confirmed['maintenance_switches'],
            'switch_reduction_fraction': 1 - confirmed['maintenance_switches'] / raw['maintenance_switches'],
            'raw_unknown_samples': raw['unknown_samples'], 'confirmed_unknown_samples': confirmed['unknown_samples']})
    routing = []
    for cell, values in calibrated_router['scores'].items():
        old = raw_router['scores'][cell]
        routing.append({'cell': cell,
            'calibrated_F8_vs_calibrated_uniform_delta_log_loss': values['uniform']['log_loss'] - values['F8']['log_loss'],
            'calibrated_F8_vs_calibrated_uniform_delta_macro_f1': values['F8']['macro_f1'] - values['uniform']['macro_f1'],
            'calibrated_F8_vs_raw_F8_delta_log_loss': old['F8']['log_loss'] - values['F8']['log_loss'],
            'calibrated_F8_vs_raw_F8_delta_macro_f1': values['F8']['macro_f1'] - old['F8']['macro_f1']})
    variation = [row for row in users['aggregate_rows'] if row['metric'] == 'macro_f1']
    costs = []
    for shots in sorted({row['shots_per_class'] for row in burden['records']}):
        rows = [r for r in burden['records'] if r['shots_per_class'] == shots]
        costs.append({'shots_per_class': shots, 'users': len(rows), 'classes': 6,
            'used_trials': sorted({r['used_calibration_trials'] for r in rows}),
            'used_signal_seconds': sorted({r['used_signal_seconds'] for r in rows}),
            'full_recording_seconds_min': min(r['used_trials_full_recording_seconds'] for r in rows),
            'full_recording_seconds_max': max(r['used_trials_full_recording_seconds'] for r in rows),
            'device_wall_time_seconds': None})
    answers = [
        {'question_id': 'A', 'question': '缺少新信息，还是已有信息组织不好？',
         'answer': '已有信息的组织会影响结果：固定因果确认降低跳动，却略降F1，且完整动作保持成功率仍低。当前证据不能判断真实8通道系统的物理信息是否不足。',
         'measurements': stream_cells,
         'boundary': '保持指标与样本F1是不同分母；确认规则没有引入新传感器信息。公开三类Myo数据不能代表用户设备或捏合。',
         'evidence': evidence('ROAM_CAUSAL_WINDOW_V1_RESULTS.json', 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json')},
        {'question_id': 'B', 'question': '哪些family提供条件增量信息？',
         'answer': '新版F7在预先冻结的独立EPN人群上，相对于同一Core及均匀软化对照取得预测增益。该证据支持这个固定组合，不等于估计了条件互信息，也不等于所有family都有效。',
         'measurements': fresh['primary_five_shot'],
         'boundary': '五shot、固定Core与0.5混合、同一留出试次；不得与不同试次的独立模型分数相减。七个较新公共默认扩展仍未通过各自验证门槛。',
         'evidence': evidence('F7_AFFINE_FRESH/results.json', 'PUBLIC_DEFAULT_EXTENSION_AUDIT.json')},
        {'question_id': 'C', 'question': '哪些family只在特定条件下有价值？',
         'answer': '当前收益依赖数据轴、预算与组合。七项新版默认扩展检查没有确立新的通用默认；F8的各预算和阶段结果应逐格报告，不能把局部收益写成全局不变性。',
         'measurements': {'default_extension_checks': guard['candidate_checks'], 'session_routing_cells': routing,
             'default_extension_delta_convention': 'F1 and log_loss both candidate minus base; positive loss means harm.',
             'routing_delta_convention': 'F1 alternative minus base; log_loss base minus alternative; positive means improvement.'},
         'boundary': '不同数据集、动作本体及已检查阶段不能合并为同一个统计检验；局部混合收益不是直接证明某个新增特征的信息量。',
         'evidence': evidence('PUBLIC_DEFAULT_EXTENSION_AUDIT.json', 'F8_ROUTER_MANUS_V1_RESULTS.json', 'F8_CALIBRATED_MANUS_V2_RESULTS.json')},
        {'question_id': 'D', 'question': '哪些family只有个人校准后才有明显价值？',
         'answer': 'F7和低维Mahalanobis在明确个人校准预算下取得收益，但这里没有同预算、同表示的未校准对照，不能由两种已校准方法的比较推断“只有校准才有效”。',
         'measurements': {'F7_five_shot': fresh['primary_five_shot'], 'EPN_calibrated_methods': epn['scores']},
         'boundary': '独立试次才计为shot；10/20shot低维协方差结论不能推广到被拒绝的高维、小样本设置或零校准部署。',
         'evidence': evidence('F7_AFFINE_FRESH/results.json', 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json')},
        {'question_id': 'E', 'question': 'Personal Anchor是否降低跨用户变化？',
         'answer': '新版EPN低维Mahalanobis相对Euclidean改善均值及最差用户，并降低这组用户的F1标准差。两者都使用个人校准，不能据此宣称相对无Anchor必然降低跨用户变化。',
         'measurements': variation,
         'boundary': '等用户均值、样本标准差与 pooled F1分开。人群仅32–41，未证明全部用户或当前设备。增加预算仍有个体退步。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json')},
        {'question_id': 'F', 'question': 'Session Signature能帮助跨天或重新佩戴吗？',
         'answer': 'MANUS新版Session路由存在混合结果；源用户温度校准后的各格改善与退步都保留，不能确立可靠的跨天或重贴默认。',
         'measurements': routing,
         'boundary': '已检查的MANUS会话是描述性证据；TD24不等于Rest拟合F0。物理重贴和本设备会话恢复未验证。',
         'evidence': evidence('F8_ROUTER_MANUS_V1_RESULTS.json', 'F8_CALIBRATED_MANUS_V2_RESULTS.json')},
        {'question_id': 'G', 'question': '是否改善最差场景R_min，而不只是均值？',
         'answer': '低维Mahalanobis改善相同预算下这组EPN最差用户，但增加到20shot仍可能损害最差用户；不能将单轴最差用户指标冒充完整七轴R_min。连续识别的样本F1也不能代替动作保持成功率。',
         'measurements': {'EPN_user_macro_f1': variation,
             'budget_comparisons': [r for r in users['paired_comparisons'] if r['kind'] == 'budget_at_fixed_method'],
             'continuous_cells': stream_cells},
         'boundary': '跨force/wearing/day/posture及真实质量等所有轴的统一改善尚未证明。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json', 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json')},
        {'question_id': 'H', 'question': '新用户或新session需要多少校准？',
         'answer': '该EPN有效低维实验使用每类10或20个独立试次，共60或120试次。提取信号曝光48或96秒，完整保存的录制更长；这些不是实际提示、休息、准备和设备总耗时。',
         'measurements': costs,
         'boundary': '覆盖六类动作；未证明跨力、姿态、重新佩戴或重复session的必要预算，也未证明几秒校准或实机收益。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json', 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json')},
    ]
    for answer in answers:
        if not set(answer['evidence']) <= set(sources): raise ValueError('Unbound evidence')
    historical = ROOT / 'feature_bank/REPORT.md'
    sources[historical.relative_to(ROOT).as_posix()] = sha(historical)
    verification_tests = ['tests/test_roam_causal_window_v1_delivery.py',
                          'tests/test_roam_debounce_control_v1_delivery.py',
                          'tests/test_f7_affine_fresh_delivery.py',
                          'tests/test_current_scientific_conclusions_v1.py',
                          'tests/test_epn_holdout_user_robustness_v2.py',
                          'tests/test_mahalanobis_epn_holdout_v2.py',
                          'tests/test_f8_calibrated_manus_v2_delivery.py']
    for name in verification_tests: sources[name] = sha(ROOT / name)
    result = {'schema': 'current_scientific_conclusions_v1', 'generator_sha256': sha(Path(__file__)),
              'requirement_document_sha256': '4da8b372c8f07936c1156935114849f85c0d83957c1ecacc6a0b6462bdf3b1f0',
              'requirement_lines': [1013, 1049], 'verification_tests': verification_tests,
              'source_sha256': sources, 'questions': answers,
              'interpretation_scope': 'Current independent versioned evidence, not resurrection of unavailable historical experiments. Prior REPORT conclusions are historical snapshots; these answers distinguish evidence changes and present limits.',
              'nominal_continuous_samples': sum(r['samples'] for r in stream['records']),
              'default_promoted': False, 'own_device_efficacy_proven': False,
              'full_seven_axis_robustness_proven': False, 'completion_proven': False}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Saved eight evidence-bound current scientific answers; no refitting')
    return result


if __name__ == '__main__': build()
