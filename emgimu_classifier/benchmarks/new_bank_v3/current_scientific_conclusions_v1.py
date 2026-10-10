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
    availability_path=ROOT/'feature_bank/AVAILABLE_BANK_FUSION_ACCEPTANCE_V1.json'
    sources[availability_path.relative_to(ROOT).as_posix()]=sha(availability_path)
    availability=json.loads(availability_path.read_text(encoding='utf8'))
    quality_fusion_path=ROOT/'feature_bank/AVAILABLE_BANK_QUALITY_ACCEPTANCE_V2.json'
    sources[quality_fusion_path.relative_to(ROOT).as_posix()]=sha(quality_fusion_path)
    quality_fusion=json.loads(quality_fusion_path.read_text(encoding='utf8'))
    portable_path=ROOT/'feature_bank/FROZEN_EMG_BANK_ACCEPTANCE_V1.json'
    sources[portable_path.relative_to(ROOT).as_posix()]=sha(portable_path)
    portable=json.loads(portable_path.read_text(encoding='utf8'))
    sources['src/emgimu/feature_bank/frozen_emg_bank_cli_v1.py']=sha(ROOT/'src/emgimu/feature_bank/frozen_emg_bank_cli_v1.py')
    song_runtime_path=ROOT/'feature_bank/SONG_F0_RUNTIME_ACCEPTANCE_V1.json'
    sources[song_runtime_path.relative_to(ROOT).as_posix()]=sha(song_runtime_path)
    song_runtime=json.loads(song_runtime_path.read_text(encoding='utf8'))
    song_stream_path=ROOT/'benchmarks/song_real8/SONG_F0_STREAM_V1_RESULTS.json'
    sources[song_stream_path.relative_to(ROOT).as_posix()]=sha(song_stream_path)
    song_stream=json.loads(song_stream_path.read_text(encoding='utf8'))
    song_gui_path=ROOT/'feature_bank/SONG_GUI_V2_ACCEPTANCE.json'
    sources[song_gui_path.relative_to(ROOT).as_posix()]=sha(song_gui_path)
    song_gui=json.loads(song_gui_path.read_text(encoding='utf8'))
    song_personal_path=ROOT/'benchmarks/song_real8/SONG_PERSONAL_SESSION_V1_RESULTS.json'
    sources[song_personal_path.relative_to(ROOT).as_posix()]=sha(song_personal_path)
    song_personal=json.loads(song_personal_path.read_text(encoding='utf8'))
    song_acceptance_path=ROOT/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'
    sources[song_acceptance_path.relative_to(ROOT).as_posix()]=sha(song_acceptance_path)
    song_acceptance=json.loads(song_acceptance_path.read_text(encoding='utf8'))
    song_lifecycle_path=ROOT/'feature_bank/SONG_PERSONAL_GUI_V1_ACCEPTANCE.json'
    sources[song_lifecycle_path.relative_to(ROOT).as_posix()]=sha(song_lifecycle_path)
    song_lifecycle=json.loads(song_lifecycle_path.read_text(encoding='utf8'))
    matched_path=ROOT/'benchmarks/song_real8/SONG_MATCHED_NORMALIZATION_V1_RESULTS.json'
    sources[matched_path.relative_to(ROOT).as_posix()]=sha(matched_path)
    matched=json.loads(matched_path.read_text(encoding='utf8'))
    matched_acceptance_path=ROOT/'feature_bank/SONG_MATCHED_NORMALIZATION_ACCEPTANCE_V1.json'
    sources[matched_acceptance_path.relative_to(ROOT).as_posix()]=sha(matched_acceptance_path)
    matched_acceptance=json.loads(matched_acceptance_path.read_text(encoding='utf8'))
    quality_path=ROOT/'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'
    sources[quality_path.relative_to(ROOT).as_posix()]=sha(quality_path)
    quality=json.loads(quality_path.read_text(encoding='utf8'))
    quality_acceptance_path=ROOT/'feature_bank/SONG_RAW_QUALITY_ACCEPTANCE_V1.json'
    sources[quality_acceptance_path.relative_to(ROOT).as_posix()]=sha(quality_acceptance_path)
    quality_acceptance=json.loads(quality_acceptance_path.read_text(encoding='utf8'))
    integrated_path=ROOT/'benchmarks/song_real8/SONG_INTEGRATED_DECISION_V1_RESULTS.json'
    sources[integrated_path.relative_to(ROOT).as_posix()]=sha(integrated_path)
    integrated=json.loads(integrated_path.read_text(encoding='utf8'))
    integrated_acceptance_path=ROOT/'feature_bank/SONG_INTEGRATED_DECISION_ACCEPTANCE_V1.json'
    sources[integrated_acceptance_path.relative_to(ROOT).as_posix()]=sha(integrated_acceptance_path)
    integrated_acceptance=json.loads(integrated_acceptance_path.read_text(encoding='utf8'))
    decision_gui_path=ROOT/'feature_bank/SONG_DECISION_GUI_V1_ACCEPTANCE.json'
    sources[decision_gui_path.relative_to(ROOT).as_posix()]=sha(decision_gui_path)
    decision_gui=json.loads(decision_gui_path.read_text(encoding='utf8'))
    extension_path=ROOT/'benchmarks/song_real8/SONG_EXTENDED_WINDOW_V1_RESULTS.json'
    extension=json.loads(extension_path.read_text(encoding='utf8'))
    extension_acceptance_path=ROOT/'feature_bank/SONG_EXTENDED_WINDOW_ACCEPTANCE_V1.json'
    extension_acceptance=json.loads(extension_acceptance_path.read_text(encoding='utf8'))
    extension_gui_path=ROOT/'feature_bank/SONG_EXTENDED_GUI_V1_ACCEPTANCE.json'
    extension_gui=json.loads(extension_gui_path.read_text(encoding='utf8'))
    for path in (extension_path,extension_acceptance_path,extension_gui_path):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ('benchmarks/song_real8/SONG_DECISION_GUI_V1_PROTOCOL.json',
                 'tests/test_personal_session_stream_v3.py', 'tests/test_decision_gui_v1_delivery.py'):
        sources[name]=sha(ROOT/name)
    temporal_gui_path=ROOT/'feature_bank/TEMPORAL_LIVE_GUI_V1_ACCEPTANCE.json'
    temporal_gui=json.loads(temporal_gui_path.read_text(encoding='utf8'))
    sources[temporal_gui_path.relative_to(ROOT).as_posix()]=sha(temporal_gui_path)
    temporal=read('PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json')
    temporal_acceptance_path=ROOT/'feature_bank/PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V1.json'
    sources[temporal_acceptance_path.relative_to(ROOT).as_posix()]=sha(temporal_acceptance_path)
    temporal_acceptance=json.loads(temporal_acceptance_path.read_text(encoding='utf8'))
    for name in ('tests/test_personal_temporal_unibo_v1_delivery.py',
                 'benchmarks/new_bank_v3/verify_personal_temporal_unibo_v1.py'):
        sources[name]=sha(ROOT/name)
    detected_temporal=read('DETECTED_PERSONAL_TEMPORAL_UNIBO_V3_RESULTS.json')
    detected_temporal_acceptance_path=ROOT/'feature_bank/DETECTED_PERSONAL_TEMPORAL_UNIBO_ACCEPTANCE_V3.json'
    sources[detected_temporal_acceptance_path.relative_to(ROOT).as_posix()]=sha(detected_temporal_acceptance_path)
    detected_temporal_acceptance=json.loads(detected_temporal_acceptance_path.read_text(encoding='utf8'))
    for name in ('benchmarks/new_bank_v3/verify_detected_personal_temporal_unibo_v3.py',
                 'tests/test_detected_personal_temporal_unibo_v3_delivery.py'):
        sources[name]=sha(ROOT/name)
    registered_rest=read('CALIBRATION_REST_CONTINUOUS_UNIBO_V1_RESULTS.json')
    registered_rest_acceptance_path=ROOT/'feature_bank/CALIBRATION_REST_CONTINUOUS_UNIBO_ACCEPTANCE_V1.json'
    registered_rest_acceptance=json.loads(registered_rest_acceptance_path.read_text(encoding='utf8'))
    sources[registered_rest_acceptance_path.relative_to(ROOT).as_posix()]=sha(registered_rest_acceptance_path)
    sources['benchmarks/new_bank_v3/verify_calibration_rest_continuous_unibo_v1.py']=sha(HERE/'verify_calibration_rest_continuous_unibo_v1.py')
    native_cost_curve=read('EMG_CALIBRATION_COST_CURVE_V1.json')
    calibrated_fusion=read('EMG_CALIBRATED_FUSION_V1_RESULTS.json')
    window_bank = read('EMG_WINDOW_BANK_V1_RESULTS.json')
    emg_bank = read('EMG_F0_F7_BANK_V1_RESULTS.json')
    class_diagnostics = read('EMG_F0_F7_BANK_V1_CLASS_DIAGNOSTICS.json')
    emg_cells = []
    for budget, arms in emg_bank['scores'].items():
        base = arms['F0']; full = arms.get('F0_F7')
        emg_cells.append({'shots_per_class':int(budget),
            'actual_F0_target_calibration_trials':0,'actual_F7_target_calibration_trials':6*int(budget),
            'F0_macro_f1':base['pooled']['macro_f1'],'F0_log_loss':base['pooled']['log_loss'],
            'F0_minimum_user_macro_f1':base['minimum_user_macro_f1'],
            'F0_F7_macro_f1':full['pooled']['macro_f1'] if full else None,
            'F0_F7_log_loss':full['pooled']['log_loss'] if full else None,
            'F0_F7_minimum_user_macro_f1':full['minimum_user_macro_f1'] if full else None})
    emg_user_variation = []
    for arm in ('F0','F0_F7'):
        values = [v['macro_f1'] for v in emg_bank['scores']['5'][arm]['per_user'].values()]
        mean = sum(values)/len(values)
        emg_user_variation.append({'method':arm,'F7_shots_per_class':5 if arm=='F0_F7' else 0,
            'users':len(values),'equal_user_mean':mean,
            'sample_std_ddof1':(sum((v-mean)**2 for v in values)/(len(values)-1))**.5,
            'minimum':min(values),'maximum':max(values)})
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
         'answer': '新版F7相对固定Core取得增益。新增纯EMG F0＋F7在另一预先冻结人群上有pooled收益，但只有6/10用户改善对数损失，未通过7/10门槛。两种Core与人群分开报告；不等于估计条件互信息或证明所有family有效。',
         'measurements': fresh['primary_five_shot'],
         'emg_only_bank':{'primary_five_shot':emg_bank['primary_five_shot'],'cells':emg_cells,
             'evaluation_trials':sum(len(b['evaluation_ids']) for b in emg_bank['blocks']),
             'reserved_calibration_trials_per_user':30,'scope':emg_bank['scope']},
         'boundary': '五shot、固定Core与0.5混合、同一留出试次；不得与不同试次的独立模型分数相减。七个较新公共默认扩展仍未通过各自验证门槛。',
         'evidence': evidence('F7_AFFINE_FRESH/results.json', 'PUBLIC_DEFAULT_EXTENSION_AUDIT.json', 'EMG_F0_F7_BANK_V1_RESULTS.json')},
        {'question_id': 'C', 'question': '哪些family只在特定条件下有价值？',
         'answer': '当前收益依赖数据轴、预算与组合。七项新版默认扩展检查没有确立新的通用默认；F8的各预算和阶段结果应逐格报告，不能把局部收益写成全局不变性。',
         'measurements': {'default_extension_checks': guard['candidate_checks'], 'session_routing_cells': routing,
             'default_extension_delta_convention': 'F1 and log_loss both candidate minus base; positive loss means harm.',
             'routing_delta_convention': 'F1 alternative minus base; log_loss base minus alternative; positive means improvement.'},
         'boundary': '不同数据集、动作本体及已检查阶段不能合并为同一个统计检验；局部混合收益不是直接证明某个新增特征的信息量。',
         'evidence': evidence('PUBLIC_DEFAULT_EXTENSION_AUDIT.json', 'F8_ROUTER_MANUS_V1_RESULTS.json', 'F8_CALIBRATED_MANUS_V2_RESULTS.json')},
        {'question_id': 'D', 'question': '哪些family只有个人校准后才有明显价值？',
         'answer': '两种已校准距离方法不能证明“只有校准才有效”。新增纯EMG实验在相同1200留出试次上比较source-only F0与1/2/5-shot F7组合，获得有限增益，但五shot主要门槛失败；其他family不能据此推断。',
         'measurements': {'F7_five_shot': fresh['primary_five_shot'], 'EPN_calibrated_methods': epn['scores']},
         'emg_only_same_trial_cells':emg_cells,
         'boundary': '独立试次才计为shot；10/20shot低维协方差结论不能推广到被拒绝的高维、小样本设置或零校准部署。',
         'evidence': evidence('F7_AFFINE_FRESH/results.json', 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json','EMG_F0_F7_BANK_V1_RESULTS.json')},
        {'question_id': 'E', 'question': 'Personal Anchor是否降低跨用户变化？',
         'answer': '新版EPN低维Mahalanobis相对Euclidean改善均值及最差用户，并降低这组用户的F1标准差。两者都使用个人校准，不能据此宣称相对无Anchor必然降低跨用户变化。',
         'measurements': variation,
         'emg_only_anchor_vs_no_target_anchor':emg_user_variation,
         'boundary': '等用户均值、样本标准差与 pooled F1分开。人群仅32–41，未证明全部用户或当前设备。增加预算仍有个体退步。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json','EMG_F0_F7_BANK_V1_RESULTS.json')},
        {'question_id': 'F', 'question': 'Session Signature能帮助跨天或重新佩戴吗？',
         'answer': 'MANUS新版Session路由存在混合结果；源用户温度校准后的各格改善与退步都保留，不能确立可靠的跨天或重贴默认。',
         'measurements': routing,
         'boundary': '已检查的MANUS会话是描述性证据；TD24不等于Rest拟合F0。物理重贴和本设备会话恢复未验证。',
         'evidence': evidence('F8_ROUTER_MANUS_V1_RESULTS.json', 'F8_CALIBRATED_MANUS_V2_RESULTS.json')},
        {'question_id': 'G', 'question': '是否改善最差场景R_min，而不只是均值？',
         'answer': '低维Mahalanobis改善相同预算下这组EPN最差用户，但增加到20shot仍可能损害最差用户；不能将单轴最差用户指标冒充完整七轴R_min。连续识别的样本F1也不能代替动作保持成功率。',
         'measurements': {'EPN_user_macro_f1': variation,
             'EMG_only_bank_single_axis_minima':emg_cells,
             'budget_comparisons': [r for r in users['paired_comparisons'] if r['kind'] == 'budget_at_fixed_method'],
             'continuous_cells': stream_cells},
         'boundary': '跨force/wearing/day/posture及真实质量等所有轴的统一改善尚未证明。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_USER_ROBUSTNESS.json', 'ROAM_DEBOUNCE_CONTROL_V1_RESULTS.json','EMG_F0_F7_BANK_V1_RESULTS.json')},
        {'question_id': 'H', 'question': '新用户或新session需要多少校准？',
         'answer': '该EPN有效低维实验使用每类10或20个独立试次，共60或120试次。提取信号曝光48或96秒，完整保存的录制更长；这些不是实际提示、休息、准备和设备总耗时。',
         'measurements': costs,
         'emg_only_F7_budget_trials_per_user':{'0':0,'1':6,'2':12,'5':30},
         'emg_only_budget_boundary':'Source-only F0 uses no target trials; 30 trials/user reserved at every budget for matched evaluation. New F7 physical timing and complete recording duration are not measured here; do not transfer the other EPN cohort timing or efficacy to this bank.',
         'boundary': '覆盖六类动作；未证明跨力、姿态、重新佩戴或重复session的必要预算，也未证明几秒校准或实机收益。',
         'evidence': evidence('MAHALANOBIS_EPN_HOLDOUT_V2_BURDEN.json', 'MAHALANOBIS_EPN_HOLDOUT_V2_RESULTS.json','EMG_F0_F7_BANK_V1_RESULTS.json')},
    ]
    for index in (5,7):
        answers[index]['native_Song_recording_session_cells']=[c for c in song_personal['cells'] if c['arm'] in ('F0','personal','session')]
        answers[index]['evidence'].append(song_personal_path.relative_to(ROOT).as_posix())
    answers[5]['answer'] += ' Song新版在同人单日不同录制中也呈混合结果：1-shot会话权重改善损失，2/5-shot反而退步；尚非跨天或真实重贴证明。'
    answers[7]['answer'] += ' Song新版另需20个S03长期试次，再加0/4/8/20个S04会话试次；0-shot当前会话不等于零总校准。'
    answers[5]['answer'] += ' 新版配对训练进一步检验了个人归一化：五试次/类时总体F1、概率损失及握拳召回退步，四项主要守门均未通过；不能默认启用。'
    for index in (5,7):answers[index]['evidence'].append(matched_path.relative_to(ROOT).as_posix())
    answers[7]['answer'] += ' 归一化分支即使使用人口权重或单F0，也需长期20试次及所用会话校准，不能记成零校准控制。'
    for index in (1,3,5,7):
        answers[index]['native_Song_CSP_extension_cells']=[c for c in extension['cells'] if c['arm'] in ('old_six_baseline','seven_reliability','old_six_F7_F8','seven_F7_F8')]
        answers[index]['evidence'].extend(path.relative_to(ROOT).as_posix() for path in
            (extension_path,extension_acceptance_path,extension_gui_path))
    answers[1]['answer'] += ' 独立新版在六组冻结模型外加入仅源数据训练的CSP；同124试次上五次/类校准时，F1从.9154升到.9422，损失从.3303降到.2675，四项预声明标准通过。仅为已检查的同人同日记录结果。'
    answers[5]['answer'] += ' 新七组F7/F8相对旧六组F7/F8也通过四项标准，但相对新七组基础方案，虽F1更高，损失/Brier更差。F4d仅为窗口/会话相对长期参考的背景坐标，不是分类器或疲劳测量。'
    answers[7]['answer'] += ' 新七组人口权重、均匀权重和单CSP控制实际目标校准成本为0；可靠性/个人原型等分支仍需长期20加当前0/4/8/20试次。新版界面提供明确选择，保留六组初始选项。'
    for index in (1,5,7):
        answers[index]['evidence'].extend(['benchmarks/new_bank_v3/PERSONAL_TEMPORAL_UNIBO_V1_RESULTS.json',
            temporal_acceptance_path.relative_to(ROOT).as_posix()])
    answers[1]['answer'] += ' 完整动作新版F5融合在UniBo Day6五次/类校准时F1从.7705升到.7763，但损失从.3906升到.4558，7人仅1人损失改善，主要标准失败。'
    answers[5]['answer'] += ' 新版完整动作长期/当次DTW和路径分支已完成84组合、924方案单元、187044逐试次概率核验；Day7/8仅描述性，不能证明实时八通道或自动边界收益。'
    answers[7]['answer'] += ' 新时序分支额外使用20个长期动作段和当前0/4/8/20个独立动作段；原G5还使用Days1-5个人训练历史，不能把基础分支写成零总注册成本。'
    for index in (5,7):answers[index]['evidence'].append(temporal_gui_path.relative_to(ROOT).as_posix())
    answers[5]['answer'] += ' 独立V5桌面入口已支持完整动作登记、保存重载、会话校准、手动与自动估计边界判断；197项软件测试通过。自动边界保留估计属性，断流/超时丢弃动作；这不是新的实机或公开数据准确率结论。'
    for answer in answers:
        if not set(answer['evidence']) <= set(sources): raise ValueError('Unbound evidence')
    # REPORT is a rendered consumer, not independent experiment evidence.
    # Excluding it here prevents a circular JSON/report hash dependency.
    renderer = ROOT / 'benchmarks/new_bank_v3/render_current_scientific_report_v1.py'
    sources[renderer.relative_to(ROOT).as_posix()] = sha(renderer)
    verification_tests = ['tests/test_roam_causal_window_v1_delivery.py',
                          'tests/test_roam_debounce_control_v1_delivery.py',
                          'tests/test_f7_affine_fresh_delivery.py',
                          'tests/test_current_scientific_conclusions_v1.py',
                          'tests/test_current_scientific_report_v1.py',
                          'tests/test_emg_f0_f7_bank_v1_delivery.py',
                          'tests/test_emg_bank_class_diagnostics_v1.py',
                          'tests/test_emg_window_bank_v1_delivery.py',
                          'tests/test_emg_calibrated_fusion_v1_delivery.py',
                          'tests/test_emg_calibration_burden_v1.py',
                          'tests/test_emg_calibration_cost_curve_v1.py',
                          'tests/test_frozen_emg_provider_bank_v1.py',
                          'tests/test_frozen_emg_bank_cli_v1.py',
                          'tests/test_song_f0_runtime_v1.py',
                          'tests/test_song_f0_stream_v1.py',
                          'tests/test_song_f0_stream_v1_delivery.py',
                          'tests/test_song_f0_stream_native_annotations_v1.py',
                          'tests/test_song_gui_v2_delivery.py',
                          'tests/test_personal_session_workflow_v1.py',
                          'tests/test_personal_session_cli_v1.py',
                          'tests/test_song_personal_session_protocol_v1.py',
                          'tests/test_song_personal_session_v1_delivery.py',
                          'tests/test_personal_session_stream_v1.py',
                          'tests/test_personal_session_gui_v1_delivery.py',
                          'tests/test_matched_normalized_bank_v1.py',
                          'tests/test_matched_normalization_cli_v1.py',
                          'tests/test_matched_normalization_delivery_v1.py',
                          'tests/test_source_quality_gate_v1.py',
                          'tests/test_personal_session_stream_v2.py',
                          'tests/test_song_raw_quality_v1_delivery.py',
                          'tests/test_personal_session_decision_v1.py',
                          'tests/test_integrated_decision_cli_v1.py',
                          'tests/test_integrated_decision_delivery_v1.py',
                          'tests/test_available_bank_fusion_v1.py',
                          'tests/test_available_bank_fusion_v1_delivery.py',
                          'tests/test_epn_holdout_user_robustness_v2.py',
                          'tests/test_mahalanobis_epn_holdout_v2.py',
                          'tests/test_f8_calibrated_manus_v2_delivery.py']
    verification_tests += ['tests/test_extended_window_decision_v1.py','tests/test_extended_window_cli_v1.py',
        'tests/test_personal_session_stream_v4.py','tests/test_extended_window_delivery_v1.py']
    verification_tests += ['tests/test_personal_temporal_unibo_v1_delivery.py','tests/test_temporal_live_gui_v1_delivery.py']
    verification_tests += ['tests/test_detected_personal_temporal_unibo_v3_delivery.py',
        'tests/test_calibration_rest_continuous_unibo_v1_delivery.py']
    for answer in answers:
        if answer['question_id'] in ('A','B','G','H'):
            answer['continuous_temporal_diagnostic'] = {
                'supported_references':596,'supported_matched':527,'calibration_recordings_excluded':140,
                'five_shot_correct_base':366,'five_shot_correct_full':367,
                'conditional_loss_base':.8000858528366488,'conditional_loss_full':.7004936843241519,
                'user_loss_wins':2,'users':7,'primary_eligible':False,'primary_pass':False,
                'missing_matched_class':'u07 fist; misses remain in the reference denominator',
                'scope':detected_temporal['scope']}
    for answer in answers:
        if answer['question_id'] in ('A','B','G','H'):
            answer['registered_rest_diagnostic']={
                'old_correct':registered_rest_acceptance['old_correct'],
                'new_correct':registered_rest_acceptance['registered_rest_correct'],
                'supported_references':registered_rest_acceptance['supported_references'],
                'primary_pass':registered_rest['primary_pass'],
                'user_success_wins':registered_rest['primary_user_success_wins'],
                'scope':registered_rest['scope']}
    verification_tests += ['tests/test_available_bank_quality_v2_delivery.py',
        'tests/test_formula_quality_register_v2_delivery.py']
    for name in verification_tests: sources[name] = sha(ROOT / name)
    result = {'schema': 'current_scientific_conclusions_v1', 'generator_sha256': sha(Path(__file__)),
              'requirement_document_sha256': '4da8b372c8f07936c1156935114849f85c0d83957c1ecacc6a0b6462bdf3b1f0',
              'requirement_lines': [1013, 1049], 'verification_tests': verification_tests,
              'source_sha256': sources, 'questions': answers,
              'available_quality_fusion': {k:quality_fusion[k] for k in ('verification_cases','checked_probability_values','rejected_invalid_quality_calls','maximum_probability_error','scope','default_promoted','physical_validation_proven')},
              'available_provider_fusion': {k:availability[k] for k in ('native_full_cases','native_missing_provider_cases','rejected_invalid_native_calls','software_scope','native_scope')},
              'portable_emg_bank':{k:portable[k] for k in ('package_path','package_sha256','channels','sample_rate_hz','window_samples','requires_IMU','native_provider_cases','native_fusion_cases','native_omission_cases','source_state_immutable','scope')},
              'song_f0_runtime':{k:song_runtime[k] for k in ('package_path','package_sha256','classes','sample_rate_hz','channels','window_samples','preprocessing_id','source_windows','source_trials','prediction_rows','records','source_state_immutable','scope')},
              'song_gui_v2':{k:song_gui[k] for k in ('bundles','records','source_parameters_exact','source_state_immutable','builtin_discovery_verified','default_promoted','physical_validation_proven','scope')},
              'song_personal_session':{k:song_personal[k] for k in ('cells','selected_source_policy','scope','evaluation_ids','personal_calibration_ids','reserved_current_ids','source_state_immutable','personal_state_immutable')},
              'song_personal_session_acceptance':{k:song_acceptance[k] for k in ('checked_native_cells','checked_trial_probabilities','maximum_probability_error','workflow_contract_id','scope')},
              'song_personal_gui':{k:song_lifecycle[k] for k in ('records','gui_flow','scope','physical_validation_proven','default_promoted')},
              'song_matched_normalization':{k:matched[k] for k in ('cells','primary_guards','primary_pass','source_model_fit_ids','source_normalization_ids','scope')},
              'song_matched_normalization_acceptance':{k:matched_acceptance[k] for k in ('checked_cells','checked_trial_probabilities','maximum_probability_error','maximum_normalization_error','scope')},
              'song_raw_quality':{k:quality[k] for k in ('cells','primary_guards','primary_pass','scope')},
              'song_raw_quality_acceptance':{k:quality_acceptance[k] for k in ('checked_cells','checked_trial_probabilities','maximum_probability_error','unmodified_soft_correct_to_wrong','unmodified_soft_wrong_to_correct','full_stream_records','scope')},
              'song_integrated_decision':{k:integrated[k] for k in ('cells','primary_guards','primary_pass','scope')},
              'song_integrated_decision_acceptance':{k:integrated_acceptance[k] for k in ('checked_cells','checked_trial_probabilities','maximum_probability_error','scope')},
              'song_extended_window':{k:extension[k] for k in ('cells','source_dimensions','primary_guards','primary_pass','full_anchor_guards','F4d_context_only','scope')},
              'song_extended_window_acceptance':extension_acceptance,
              'personal_temporal_unibo':{k:temporal[k] for k in ('pooled','primary_guards','primary_pass','primary_user_loss_wins','scope')},
              'personal_temporal_unibo_acceptance':temporal_acceptance,
              'detected_personal_temporal_unibo':{k:detected_temporal[k] for k in ('pooled','totals','primary_guards','primary_eligible','primary_pass','primary_user_loss_wins','scope')},
              'detected_personal_temporal_unibo_acceptance':detected_temporal_acceptance,
              'calibration_rest_continuous_unibo':{k:registered_rest[k] for k in ('pooled','primary_detection','primary_guards','primary_pass','primary_user_success_wins','scope')},
              'calibration_rest_continuous_unibo_acceptance':registered_rest_acceptance,
              'temporal_live_gui_acceptance':temporal_gui,
              'song_extended_gui_acceptance':{k:extension_gui[k] for k in ('records','verified_windows','scope','source_and_profiles_immutable','independent_direct_DFT_context','F4d_context_only')},
              'song_decision_gui_acceptance':{k:decision_gui[k] for k in ('records','verified_windows','scope',
                  'independent_filter_and_geometry_probabilities','chronological_confirmation_exact',
                  'source_and_profiles_immutable','unmodified_structural_off_probability_identity')},
              'song_f0_stream':{'classes':song_stream['classes'],'scope':song_stream['scope'],
                  'cells':[{k:row[k] for k in ('session','arm','raw_samples','emissions','unscored_emissions','one_pass_max_probability_error')} | {policy:{k:row[policy][k] for k in ('eligible_windows','eligible_trials','unknown_windows','trial_balanced_macro_f1','trial_balanced_accuracy','whole_stable_hold_correct','within_stable_trial_switches')} for policy in ('raw','confirmed')} for row in song_stream['records']]},
              'emg_native_calibration_cost':native_cost_curve['cells'],
              'emg_calibrated_fusion':{'scope':calibrated_fusion['scope'],'primary_five_shot':calibrated_fusion['primary_five_shot'],
                  'source_users':list(range(1,16)),'target_users':list(range(62,72)),
                  'cells':[{'shots_per_class':int(shots),'arm':arm,'actual_target_calibration_trials_per_user':6*int(shots) if arm=='reliability_bank' else 0,
                      **s[arm]['pooled'],'minimum_user_macro_f1':s[arm]['minimum_user_macro_f1']} for shots,s in calibrated_fusion['scores'].items() for arm in ('F0','uniform_bank','reliability_bank')]},
              'emg_window_bank': {'scope':window_bank['scope'],'primary':window_bank['primary'],
                  'source_dimension':window_bank['source_models']['window_bank']['dimension'],
                  'actual_target_calibration_trials':0,'evaluation_trials':sum(len(b['labels']) for b in window_bank['blocks']),
                  'cells':[{'arm':arm,'groups':window_bank['source_models'][arm]['groups'],
                            'dimension':window_bank['source_models'][arm]['dimension'],
                            **s['pooled'],'minimum_user_macro_f1':s['minimum_user_macro_f1']} for arm,s in window_bank['scores'].items()]},
              'emg_only_class_diagnostics': {
                  'native_class_axis': class_diagnostics['native_class_axis'],
                  'cells': [c for c in class_diagnostics['cells'] if c['shots_per_class']==5 and c['user']=='ALL' and c['arm'] in ('F0','F0_F7')],
                  'paired_class_changes': [c for c in class_diagnostics['paired_class_changes'] if c['shots_per_class']==5 and c['user']=='ALL'],
                  'scope': class_diagnostics['scope']},
              'interpretation_scope': 'Current independent versioned evidence, not resurrection of unavailable historical experiments. REPORT renders these current A-H answers; earlier experiment-specific sections remain scoped historical evidence, not a different current global conclusion.',
              'nominal_continuous_samples': sum(r['samples'] for r in stream['records']),
              'default_promoted': False, 'own_device_efficacy_proven': False,
              'full_seven_axis_robustness_proven': False, 'completion_proven': False}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Saved eight evidence-bound current scientific answers; no refitting')
    return result


if __name__ == '__main__': build()
