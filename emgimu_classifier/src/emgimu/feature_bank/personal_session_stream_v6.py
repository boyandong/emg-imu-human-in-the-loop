"""Shared window/full-bout registration with immutable bound source profiles."""
import argparse
import json
import sys

from .extended_window_cli_v1 import load_extended_workflow
from .joint_bout_live_v1 import JointBoutLiveV1
from .personal_session_cli_v1 import _json
from .personal_session_stream_v3 import _LifecycleProfile
from .personal_session_stream_v5 import PersonalSessionStreamV5


class PersonalSessionStreamV6(PersonalSessionStreamV5):
    def __init__(self,decision,**kwargs):
        super().__init__(decision,**kwargs)
        self.temporal_service=JointBoutLiveV1(decision,self.filters,user_id=self.user,
            session_id=self.session_id,observed_channels=self.channels)

    def _bind_window_profiles(self):
        s=self.temporal_service
        self.personal=None if s.personal is None else _LifecycleProfile(s.personal.window)
        self.session=None if s.session is None else _LifecycleProfile(s.session.window)

    def info(self):
        result=super().info()
        result.update(schema='song_personal_session_stream_v6',profile_schema='joint_bout_profile_v1',
            shared_window_temporal_registration=True)
        return result

    def command(self,message):
        op=message['op']
        if op in ('profiles','begin','trial','save'):
            raise ValueError('Shared backend requires full-action registration and joint profile operations')
        try:result=super().command(message)
        except Exception:
            if op=='temporal_profiles':self.mode='idle';self.reset()
            raise
        if op in ('temporal_save','temporal_profiles'):
            self._bind_window_profiles()
            result.update(self.info())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','policy','acceptance','gate-package','gate-results','user','session'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True);args=parser.parse_args()
    try:
        service=PersonalSessionStreamV6(load_extended_workflow(args.package,args.policy,args.acceptance,
            args.gate_package,args.gate_results),user_id=args.user,session_id=args.session,channel_ids=args.channels)
        print(json.dumps(dict(ok=True,result=service.info()),default=_json),flush=True)
        for line in sys.stdin:
            try:
                message=json.loads(line)
                if message.get('op')=='quit':break
                response=dict(ok=True,result=service.command(message))
            except Exception as exc:response=dict(ok=False,error=str(exc))
            print(json.dumps(response,default=_json,ensure_ascii=True),flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False,error=str(exc))),flush=True);return 2
    return 0


if __name__=='__main__':sys.exit(main())
