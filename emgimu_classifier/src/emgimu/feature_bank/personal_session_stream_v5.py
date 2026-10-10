"""Opt-in full-action lifecycle beside the frozen seven-provider window stream."""
import argparse
import json
import sys
from .personal_session_stream_v4 import PersonalSessionStreamV4
from .temporal_bout_live_v1 import TemporalBoutLiveV1
from .extended_window_cli_v1 import load_extended_workflow
from .personal_session_cli_v1 import _json


class PersonalSessionStreamV5(PersonalSessionStreamV4):
    def __init__(self,decision,**kwargs):
        self.temporal_service=None
        super().__init__(decision,**kwargs)
        self.temporal_service=TemporalBoutLiveV1(decision,self.filters,user_id=self.user,
            session_id=self.session_id,observed_channels=self.channels)

    def reset(self):
        super().reset()
        if self.temporal_service is not None:self.temporal_service.reset()

    def info(self):
        result=super().info();result['schema']='song_personal_session_stream_v5'
        result['temporal']=None if self.temporal_service is None else self.temporal_service.info()
        return result

    def options(self):
        return dict(personal=self.workflow.unwrap(self.personal),session=self.workflow.unwrap(self.session),
            quality_mode=self.quality_mode,use_anchor=self.workflow.use_anchor,
            use_session_routing=self.workflow.use_session_routing,anchor_mode=self.workflow.anchor_mode)

    def command(self,message):
        if any(k in message for k in ('labels','trial_labels','evaluation_labels','evaluation_trials')):
            raise ValueError('Full-action prediction never accepts evaluation labels; calibration follows guided cues')
        op=message['op'];s=self.temporal_service
        if s.capture is not None and op not in ('info','emg','pause','gap','cancel','temporal_trial_start','temporal_trial_end','temporal_save'):
            raise ValueError('Save/cancel full-action calibration before changing workflow')
        if op.startswith('temporal_'):
            if self.capture is not None:raise ValueError('Save/cancel window calibration first')
            extra={}
            if op=='temporal_begin':
                self.mode='idle';self.reset();s.begin(message['kind'],message.get('shots',1));self.mode='temporal_calibrating'
            elif op=='temporal_trial_start':s.start_trial()
            elif op=='temporal_trial_end':extra=s.end_trial(self.quality_mode)
            elif op=='temporal_save':extra=s.save(message['path']);self.mode='idle';self.reset()
            elif op=='temporal_profiles':
                s.load(message.get('personal'),message.get('session'));self.mode='idle';self.reset()
            elif op=='temporal_mode':
                mode=message.get('mode')
                if mode not in ('off','manual','auto'):raise ValueError('Full-action mode must be off/manual/auto')
                if mode!='off' and s.personal is None:raise ValueError('Register/load a full-action personal profile first')
                s.requested_mode=mode;self.mode='idle';self.reset()
            elif op=='temporal_action_start':
                if self.mode!='recognizing' or s.requested_mode!='manual':raise ValueError('Start recognition in manual full-action mode first')
                s.action_start()
            elif op=='temporal_action_end':
                if self.mode!='recognizing' or s.requested_mode!='manual':raise ValueError('Manual full-action recognition required')
                extra['temporal_events']=[s.action_end(**self.options())]
            else:raise ValueError('Unknown full-action operation')
            return dict(**self.info(),**extra)
        if op=='recognize' and s.requested_mode!='off' and s.personal is None:
            raise ValueError('Full-action personal profile required')
        return super().command(message)

    def ingest(self,raw,indices):
        try:
            result=super().ingest(raw,indices)
            if result.get('reset_reason') or self.mode=='idle':return result
            events=self.temporal_service.feed(raw,indices,
                automatic=self.mode=='recognizing' and self.temporal_service.requested_mode=='auto',**self.options())
            result['temporal']=self.temporal_service.info()
            if events:result['temporal_events']=events
            return result
        except Exception:
            self.mode='idle';self.capture=None;self.reset();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','policy','acceptance','gate-package','gate-results','user','session'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True);args=parser.parse_args()
    try:
        service=PersonalSessionStreamV5(load_extended_workflow(args.package,args.policy,args.acceptance,args.gate_package,args.gate_results),
            user_id=args.user,session_id=args.session,channel_ids=args.channels)
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
