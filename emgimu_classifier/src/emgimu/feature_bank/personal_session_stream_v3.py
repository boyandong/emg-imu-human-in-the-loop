"""Opt-in operational F7/F8 service with raw F9 and frozen V1/V2 lifecycle.

The population/reliability route remains the initial mode. Decision branches
must be explicitly selected; numerical stream parity is not live efficacy.
"""
from dataclasses import dataclass
import argparse
import json
import sys
from .personal_session_decision_cli_v1 import load_decision_workflow
from .personal_session_stream_v2 import PersonalSessionStreamV2
from .personal_session_cli_v1 import _json


@dataclass(frozen=True)
class _LifecycleProfile:
    """Expose lifecycle metadata without modifying serialized decision profiles."""
    decision: object

    @property
    def profile_id(self):
        return self.decision.profile_id

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'decision').base, name)


class _LifecycleWorkflow:
    def __init__(self, decision):
        self.decision = decision
        self.bank = decision.bank
        self.channels = decision.channels
        self.rest_label = decision.rest_label
        self.preprocessing_id = decision.preprocessing_id
        self.contract_id = decision.contract_id
        self.use_anchor = False
        self.use_session_routing = False
        self.anchor_mode = 'blended'

    def _input(self, *args, **kwargs):
        return self.decision._input(*args, **kwargs)

    @staticmethod
    def unwrap(profile):
        return None if profile is None else profile.decision

    def enroll_user(self, *args, **kwargs):
        return _LifecycleProfile(self.decision.enroll_user(*args, **kwargs))

    def calibrate_session(self, *args, personal, **kwargs):
        return _LifecycleProfile(self.decision.calibrate_session(*args,
            personal=self.unwrap(personal), **kwargs))

    def save_profile(self, profile, path):
        return self.decision.save_profile(self.unwrap(profile), path)

    def load_profile(self, path, **kwargs):
        if 'personal' in kwargs:
            kwargs['personal'] = self.unwrap(kwargs['personal'])
        return _LifecycleProfile(self.decision.load_profile(path, **kwargs))

    def predict(self, *args, personal=None, session=None, **kwargs):
        return self.decision.predict(*args, personal=self.unwrap(personal),
            session=self.unwrap(session), use_anchor=self.use_anchor,
            use_session_routing=self.use_session_routing,
            anchor_mode=self.anchor_mode, quality_mode='off', **kwargs)


class PersonalSessionStreamV3(PersonalSessionStreamV2):
    def __init__(self, decision, **kwargs):
        if decision.gate is None:
            raise ValueError('A bound raw quality gate is required for this service')
        super().__init__(_LifecycleWorkflow(decision), decision.gate, **kwargs)

    def info(self):
        info = super().info()
        info.update(schema='song_personal_session_stream_v3',
            decision_policy_id=self.workflow.decision.policy_id,
            use_anchor=self.workflow.use_anchor,
            use_session_routing=self.workflow.use_session_routing,
            anchor_mode=self.workflow.anchor_mode,
            anchor_enabled=self.personal is not None and self.workflow.use_anchor,
            session_routing_enabled=self.session is not None and self.workflow.use_session_routing,
            profile_schema='personal_session_decision_profile_v1',
            default_promoted=False)
        return info

    def _predict(self, batch, ids, offsets):
        result = self.workflow.predict(batch, ids, window_offsets=offsets,
            personal=self.personal, session=self.session, **self.kwargs)
        # V2 applies raw F9 to the F7-mixed provider probabilities and F8-routed
        # weights, rather than silently falling back to the original providers.
        self.provider_readout = dict(probabilities=result['decision_provider_probabilities'],
            trial_ids=result['trial_ids'])
        return result

    def command(self, message):
        if any(key in message for key in ('labels', 'trial_labels', 'evaluation_labels', 'evaluation_trials')):
            raise ValueError('Evaluation labels/reservations are not service inputs; calibration uses guided cues')
        if message['op'] == 'decision':
            if self.capture is not None:
                raise ValueError('Cancel/save calibration before changing decision branches')
            anchor = message.get('use_anchor')
            routing = message.get('use_session_routing')
            mode = message.get('anchor_mode', 'blended')
            if type(anchor) is not bool or type(routing) is not bool or mode not in ('long_term', 'local', 'blended'):
                raise ValueError('Explicit Boolean branches and valid anchor mode required')
            self.workflow.use_anchor = anchor
            self.workflow.use_session_routing = routing
            self.workflow.anchor_mode = mode
            self.mode = 'idle'
            self.reset()
            return self.info()
        result = super().command(message)
        if message['op'] == 'save':
            from pathlib import Path
            path = Path(result['calibration_audit_path'])
            audit = json.loads(path.read_text(encoding='utf8'))
            audit.update(schema='guided_calibration_capture_v3',
                decision_policy_id=self.workflow.decision.policy_id,
                profile_schema='personal_session_decision_profile_v1')
            path.write_text(json.dumps(audit, ensure_ascii=False)+'\n', encoding='utf8', newline='\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'base-acceptance', 'policy', 'acceptance', 'gate-package', 'gate-results', 'user', 'session'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--channels', nargs=8, required=True)
    args = parser.parse_args()
    try:
        workflow = load_decision_workflow(args.package, args.base_acceptance,
            args.policy, args.acceptance, args.gate_package, args.gate_results)
        service = PersonalSessionStreamV3(workflow, user_id=args.user,
            session_id=args.session, channel_ids=args.channels)
        print(json.dumps(dict(ok=True, result=service.info()), default=_json), flush=True)
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get('op') == 'quit':
                    break
                response = dict(ok=True, result=service.command(message))
            except Exception as exc:
                response = dict(ok=False, error=str(exc))
            print(json.dumps(response, ensure_ascii=True, default=_json), flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=True), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
