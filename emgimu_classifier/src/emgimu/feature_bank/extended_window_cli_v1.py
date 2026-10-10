"""Portable seven-provider lifecycle; F4d is exported only as context."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .core import FeatureBatch
from .frozen_emg_bank_cli_v1 import load_windows
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .personal_session_cli_v1 import _json
from .personal_session_stream_v1 import PREPROCESSING
from .personal_session_stream_v2 import load_gate
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1
from .extended_window_decision_v1 import CspQualityGateV1, ExtendedWindowDecisionV1


def load_extended_workflow(package, policy, acceptance, gate_package, gate_results):
    evidence = json.loads(Path(acceptance).read_text(encoding='utf8'))
    packed_policy = Path(policy).read_bytes()
    packed = Path(package).read_bytes()
    if (hashlib.sha256(packed_policy).hexdigest() != evidence['policy_sha256']
            or hashlib.sha256(packed).hexdigest() != evidence['source_bank_sha256']):
        raise ValueError('Extended source package/policy checksum differs')
    config = json.loads(packed_policy)
    if config['source_bank_sha256'] != evidence['source_bank_sha256']:
        raise ValueError('Extended policy source checksum differs')
    bank = pickle.loads(packed)
    if (not isinstance(bank, FrozenEmgProviderBankV1) or bank.bank_id_ != config['source_bank_id']
            or bank.sensor_contract_ != (250., 50, 8)):
        raise ValueError('Extended source bank/sensor identity differs')
    base = PersonalSessionWorkflowV1(bank, channel_ids=tuple(f'CH{i+1}' for i in range(8)),
        preprocessing_id=PREPROCESSING, rest_label='neutral',
        quality_options={'line_frequency_hz': 50, 'pre_highpass_available': False})
    gate = CspQualityGateV1(load_gate(gate_package, gate_results))
    workflow = ExtendedWindowDecisionV1(base, gate, anchor_mix=config['anchor_mix'])
    if (gate.policy_id != config['gate_policy_id'] or workflow.policy_id != config['decision_policy_id']
            or workflow.policy_id != evidence['decision_policy_id']):
        raise ValueError('Extended decision/gate configuration identity differs')
    return workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('enroll', 'session', 'predict'))
    for name in ('package', 'policy', 'acceptance', 'gate-package', 'gate-results', 'windows',
                 'user', 'session', 'output', 'preprocessing'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--channels', nargs=8, required=True)
    parser.add_argument('--labels'); parser.add_argument('--evaluation-trials')
    parser.add_argument('--personal-profile'); parser.add_argument('--session-profile')
    parser.add_argument('--providers', nargs='+'); parser.add_argument('--raw-windows')
    parser.add_argument('--anchor-mode', choices=('long_term', 'local', 'blended'), default='blended')
    parser.add_argument('--quality-mode', choices=('off', 'structural', 'soft'), default='off')
    parser.add_argument('--enable-anchor', action='store_true')
    parser.add_argument('--enable-session-routing', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'predict' and (args.labels or args.evaluation_trials):
            raise ValueError('Prediction never accepts labels or calibration reservations')
        if args.action != 'predict' and (not args.labels or args.providers or args.session_profile
                or args.raw_windows or args.enable_anchor or args.enable_session_routing
                or args.quality_mode != 'off' or args.anchor_mode != 'blended'):
            raise ValueError('Calibration needs separate labels; prediction controls cannot change enrollment')
        if ((args.action == 'enroll' and args.personal_profile)
                or (args.action == 'session' and not args.personal_profile)
                or (args.session_profile and not args.personal_profile)):
            raise ValueError('Invalid personal/session profile lifecycle')
        w = load_extended_workflow(args.package, args.policy, args.acceptance,
            args.gate_package, args.gate_results)
        batch, ids, offsets = load_windows(args.windows)
        common = dict(window_offsets=offsets, user_id=args.user, session_id=args.session,
            observed_channel_ids=args.channels, preprocessing_id=args.preprocessing)
        personal = None if not args.personal_profile else w.load_profile(args.personal_profile, user_id=args.user)
        if args.action != 'predict':
            labels = json.loads(Path(args.labels).read_text(encoding='utf8'))
            forbidden = [] if not args.evaluation_trials else json.loads(Path(args.evaluation_trials).read_text(encoding='utf8'))
            if args.action == 'enroll':
                profile = w.enroll_user(batch, ids, labels, forbidden_evaluation_trials=forbidden, **common)
            else:
                profile = w.calibrate_session(batch, ids, labels, personal=personal,
                    forbidden_evaluation_trials=forbidden, **common)
            w.save_profile(profile, args.output)
        else:
            session = None if not args.session_profile else w.load_profile(args.session_profile,
                user_id=args.user, session_id=args.session, personal=personal)
            raw = None
            if args.raw_windows:
                raw, raw_ids, raw_offsets = load_windows(args.raw_windows)
                if not np.array_equal(ids, raw_ids) or not np.array_equal(offsets, raw_offsets):
                    raise ValueError('Raw/filtered trial/offset axes differ')
            result = w.predict(batch, ids, personal=personal, session=session, available=args.providers,
                use_anchor=args.enable_anchor, use_session_routing=args.enable_session_routing,
                anchor_mode=args.anchor_mode, quality_mode=args.quality_mode, raw_batch=raw, **common)
            with Path(args.output).open('xb') as stream:
                stream.write((json.dumps(result, ensure_ascii=False, indent=2, default=_json)+'\n').encode('utf8'))
        print(args.action+': saved '+args.output, flush=True)
        return 0
    except (ValueError, TypeError, KeyError, OSError, pickle.UnpicklingError, EOFError, zipfile.BadZipFile) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
