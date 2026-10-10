"""Versioned raw-quality integration; frozen V1 runtime and experiments remain intact."""
from collections import deque
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from .core import FeatureBatch
from .personal_session_stream_v1 import PersonalSessionStreamV1
from .personal_session_cli_v1 import load_workflow,_json
from .frozen_emg_bank_cli_v1 import load_windows
from .source_quality_gate_v1 import SourceQualityGateV1


def load_gate(package,results):
    r=json.loads(Path(results).read_text(encoding='utf8'));packed=Path(package).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=r['gate_sha256']:raise ValueError('Source quality package checksum differs')
    gate=pickle.loads(packed)
    if not isinstance(gate,SourceQualityGateV1) or gate.policy_id!=r['gate_policy_id']:
        raise ValueError('Source quality policy identity differs')
    return gate


class PersonalSessionStreamV2(PersonalSessionStreamV1):
    def __init__(self,workflow,gate,**kwargs):
        if not isinstance(gate,SourceQualityGateV1) or tuple(gate.channels)!=tuple(workflow.channels):
            raise ValueError('Source quality and classifier channel identities differ')
        self.gate=gate;self.quality_mode='off'
        super().__init__(workflow,**kwargs)

    def reset(self):
        super().reset()
        self.raw_history=deque(maxlen=49)
        self.quality_candidate=self.quality_active=None;self.quality_confirmations=0
        self.raw_trials={};self.accepted_raw={}

    def info(self):
        info=super().info()
        info.update(schema='song_personal_session_stream_v2',quality_mode=self.quality_mode,
            quality_rejection_enabled=self.quality_mode!='off',gate_policy_id=self.gate.policy_id,
            quality_input='raw_pre_software_highpass',raw_quality_range_provenance=self.gate.range_provenance)
        return info

    def _predict(self,batch,ids,offsets):
        result=super()._predict(batch,ids,offsets)
        mapped=self.workflow._input(batch,observed_channel_ids=self.channels,preprocessing_id=self.workflow.preprocessing_id)
        self.provider_readout=self.workflow.bank.predict_providers(mapped,ids,window_offsets=offsets,user_id=self.user)
        return result

    def command(self,message):
        op=message['op']
        if op=='quality':
            mode=message['mode']
            if mode not in ('off','structural','soft'):raise ValueError('Unknown quality mode')
            if self.capture is not None:raise ValueError('Cancel/save calibration before changing quality policy')
            self.quality_mode=mode;self.mode='idle';self.reset()
            return self.info()
        if op=='replay' and self.quality_mode!='off':
            if not message.get('raw_path'):raise ValueError('Quality-enabled replay needs a separate raw pre-highpass window NPZ')
            if self.capture is not None:raise ValueError('Cancel/save calibration before replay')
            filtered,ids,offsets=load_windows(message['path']);raw,raw_ids,raw_offsets=load_windows(message['raw_path'])
            if (not np.array_equal(ids,raw_ids) or not np.array_equal(offsets,raw_offsets)
                    or raw.emg.shape!=filtered.emg.shape or raw.sample_rate_hz!=filtered.sample_rate_hz):
                raise ValueError('Raw/filtered replay window trial/offset/sensor contracts differ')
            base=self._predict(filtered,ids,offsets)
            decision=self.gate.decide(self.provider_readout['probabilities'],base['weights'],raw,ids,
                observed_channel_ids=self.channels,class_names=base['class_names'],mode=self.quality_mode,
                provider_trial_ids={g:self.provider_readout['trial_ids'] for g in self.provider_readout['probabilities']})
            base.update(probabilities=decision['probabilities'],quality_decision=decision)
            return dict(**self.info(),prediction=base)
        if op=='save':
            c=self.capture
            if c is None or c['trial'] is not None or len(c['labels'])!=len(c['order']):
                return super().command(message)
            raw_path=Path(str(message['path'])+'.raw_windows.npz')
            raw_values=np.stack([self.accepted_raw[t][i] for t,i in zip(c['ids'],c['offsets'])])
            with raw_path.open('xb') as f:
                np.savez_compressed(f,emg=raw_values,sample_rate_hz=np.array(250.),trial_ids=np.asarray(c['ids']),
                                    window_offsets=np.asarray(c['offsets']))
            try:result=super().command(message)
            except Exception:
                raw_path.unlink();raise
            audit=Path(result['calibration_audit_path']);metadata=json.loads(audit.read_text(encoding='utf8'))
            metadata.update(schema='guided_calibration_capture_v2',quality_mode=self.quality_mode,
                gate_policy_id=self.gate.policy_id,raw_windows_path=str(raw_path),raw_pre_software_highpass=True)
            audit.write_text(json.dumps(metadata,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
            result['calibration_raw_windows_path']=str(raw_path)
            return result
        return super().command(message)

    def ingest(self,raw,indices):
        raw=np.asarray(raw);indices=np.asarray(indices)
        if (raw.ndim!=2 or raw.shape[1]!=8 or indices.shape!=(len(raw),)
                or indices.dtype.kind not in 'iu' or not np.isfinite(raw).all()):
            raise ValueError('Finite raw eight-channel samples and integer indices required')
        was_recognizing=self.mode=='recognizing';c=self.capture
        trial=None if c is None else c['trial']
        # Keep the pre-filter held portion alongside the parent filtered capture.
        if trial is not None and raw.ndim==2 and raw.shape[1]==8:
            values=self.raw_trials.setdefault(trial['id'],[])
            for offset,value in enumerate(raw):
                sample=trial['samples']+offset+1
                if c['settle']<sample<=c['settle']+c['hold']:values.append(value.copy())
        previous=list(self.raw_history)
        result=super().ingest(raw,indices)
        if result.get('reset_reason'):
            return result
        if trial is not None and self.capture is not None and trial['id'] in c['labels']:
            values=self.raw_trials.pop(trial['id'])
            windows=np.stack([np.stack(values[start:start+50]) for start in range(0,c['hold']-49,10)])
            observation=self.gate.observe(FeatureBatch(windows,250.),[trial['id']]*len(windows),observed_channel_ids=self.channels)
            bad=(observation['trial_structural_invalid_channels'].any() if self.quality_mode=='structural' else
                 observation['trial_soft_channel_quality'].max()==0 if self.quality_mode=='soft' else False)
            if bad:
                keep=[i for i,t in enumerate(c['ids']) if t!=trial['id']]
                for key in ('windows','ids','offsets'):c[key]=[c[key][i] for i in keep]
                del c['labels'][trial['id']]
                result=dict(**self.info(),calibration_rejected=True,
                    quality_message='This guided trial was not counted because of raw signal quality; retry after checking contact')
            else:self.accepted_raw[trial['id']]=windows
        if not was_recognizing or 'probabilities' not in result:
            if was_recognizing:self.raw_history.extend(raw.tolist())
            return result
        combined=np.asarray(previous+raw.tolist());history=len(previous)
        ends=result['output_sample_indices']
        windows=np.stack([combined[history+int(end)-int(indices[0])-49:history+int(end)-int(indices[0])+1] for end in ends])
        self.raw_history.extend(raw.tolist())
        readout=self.provider_readout
        ids=np.array([f'live:{self.session_id}:{self.epoch}:{end}' for end in ends])
        decision=self.gate.decide(readout['probabilities'],result['weights'],FeatureBatch(windows,250.),ids,
            observed_channel_ids=self.channels,class_names=result['classes'],mode=self.quality_mode,
            provider_trial_ids={g:readout['trial_ids'] for g in readout['probabilities']})
        lookup={t:i for i,t in enumerate(decision['trial_ids'])};order=[lookup[t] for t in ids]
        q=decision['probabilities'][order];rejected=decision['rejected'][order]
        confirmed=[]
        for probability,reject in zip(q,rejected):
            if reject:
                self.quality_candidate=self.quality_active=None;self.quality_confirmations=0
            else:
                candidate=result['classes'][int(probability.argmax())]
                self.quality_confirmations=self.quality_confirmations+1 if candidate==self.quality_candidate else 1
                self.quality_candidate=candidate
                if self.quality_confirmations>=2:self.quality_active=candidate
            confirmed.append(self.quality_active)
        result.update(probabilities=q,confirmed_labels=confirmed,quality_rejected=rejected,
            channel_quality=decision['channel_quality'][order],bad_channel_count=decision['bad_channel_count'][order],
            quality_provider_weights=decision['provider_weights'][order])
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','acceptance','gate-package','gate-results','user','session'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--channels',nargs=8,required=True);args=parser.parse_args()
    try:
        service=PersonalSessionStreamV2(load_workflow(args.package,args.acceptance),load_gate(args.gate_package,args.gate_results),
            user_id=args.user,session_id=args.session,channel_ids=args.channels)
        print(json.dumps(dict(ok=True,result=service.info()),default=_json),flush=True)
        for line in sys.stdin:
            try:
                message=json.loads(line)
                if message.get('op')=='quit':break
                response=dict(ok=True,result=service.command(message))
            except Exception as exc:response=dict(ok=False,error=str(exc))
            print(json.dumps(response,ensure_ascii=True,default=_json),flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False,error=str(exc)),ensure_ascii=True),flush=True);return 2
    return 0


if __name__=='__main__':sys.exit(main())
