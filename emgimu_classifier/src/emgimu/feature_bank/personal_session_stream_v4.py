"""Seven-provider opt-in stream with persistent F4d context and raw CSP quality."""
import argparse
import json
from pathlib import Path
import sys
from .extended_window_cli_v1 import load_extended_workflow
from .personal_session_cli_v1 import _json
from .personal_session_stream_v3 import PersonalSessionStreamV3


class PersonalSessionStreamV4(PersonalSessionStreamV3):
    def reset(self):
        super().reset()
        self.spectral_context = None

    def info(self):
        result = super().info()
        result.update(schema='song_personal_session_stream_v4',
            profile_schema='extended_window_profile_v1',
            bank_provider_names=tuple(self.workflow.bank.providers_),
            spectral_context=self.spectral_context, F4d_context_only=True)
        return result

    def _predict(self, batch, ids, offsets):
        result = super()._predict(batch, ids, offsets)
        context = result['spectral_context']
        self.spectral_context = None if context is None else dict(
            window_minus_long=context['window_minus_long'][-1],
            session_minus_long=context['session_minus_long'], bands=context['bands'],
            channel_ids=context['channel_ids'], role=context['role'])
        return result

    def ingest(self, raw, indices):
        result = super().ingest(raw, indices)
        if 'probabilities' in result and self.spectral_context is not None:
            context = self.spectral_context
            # The feature/context window axis is the original chronological axis;
            # provider trial probabilities use their independently sorted axis.
            result['spectral_context'] = dict(
                window_minus_long=context['window_minus_long'],
                session_minus_long=context['session_minus_long'],
                output_sample_index=result['output_sample_indices'][-1],
                bands=context['bands'], channel_ids=context['channel_ids'], role=context['role'])
        return result

    def command(self, message):
        result = super().command(message)
        if message['op'] == 'save':
            path = Path(result['calibration_audit_path'])
            audit = json.loads(path.read_text(encoding='utf8'))
            audit.update(schema='guided_calibration_capture_v4',
                profile_schema='extended_window_profile_v1', F4d_context_only=True)
            path.write_text(json.dumps(audit, ensure_ascii=False)+'\n', encoding='utf8', newline='\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'policy', 'acceptance', 'gate-package', 'gate-results', 'user', 'session'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--channels', nargs=8, required=True)
    args = parser.parse_args()
    try:
        service = PersonalSessionStreamV4(load_extended_workflow(args.package, args.policy,
            args.acceptance, args.gate_package, args.gate_results),
            user_id=args.user, session_id=args.session, channel_ids=args.channels)
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
