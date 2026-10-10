"""Native eight-channel full-action capture and estimated-boundary composition."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import uuid
import zipfile
import numpy as np
from scipy.signal import sosfilt
from .core import FeatureBatch
from .autonomous_bouts_v1 import AutonomousBoutDetectorV1
from .personal_temporal_bouts_v1 import PersonalTemporalBoutsV1,TemporalBoutBatchV1
from .complete_bout_window_adapter_v1 import CompleteBoutWindowAdapterV1


class TemporalBoutLiveV1:
    def __init__(self,window,filters,*,user_id,session_id,observed_channels):
        self.window=window;self.filters=filters;self.user=user_id;self.session_id=session_id
        self.channels=tuple(window.channels);observed=tuple(observed_channels)
        if len(observed)!=8 or set(observed)!=set(self.channels):raise ValueError('Explicit eight-channel input order required')
        self.order=[observed.index(c) for c in self.channels]
        self.temporal=PersonalTemporalBoutsV1(source_bank_id=window.bank.bank_id_,sample_rate_hz=250.,
            channel_ids=self.channels,preprocessing_id=window.preprocessing_id,class_names=tuple(window.bank.classes_))
        self.adapter=CompleteBoutWindowAdapterV1(window,self.temporal)
        self.personal=self.session=self.long_detector=self.local_detector=None
        self.requested_mode='off';self.capture=None;self.reset()

    def reset(self):
        self.zi=[np.zeros((len(s),2,8)) for s in self.filters]
        self.last=None;self.history=np.empty((0,8));self.manual=None;self.capture=None
        self.recording='auto:'+self.session_id+':'+uuid.uuid4().hex
        self.detector=copy.deepcopy(self.local_detector if self.local_detector is not None else self.long_detector)
        if self.detector is not None:self.detector.reset()

    def info(self):
        c=self.capture
        return dict(mode=self.requested_mode,personal_profile_id=None if self.personal is None else self.personal.profile_id,
            session_profile_id=None if self.session is None else self.session.profile_id,
            capture=None if c is None else dict(kind=c['kind'],total=len(c['order']),completed=len(c['rows']),
                next_label=None if len(c['rows'])==len(c['order']) else c['order'][len(c['rows'])],
                collecting=c['trial'] is not None,samples=0 if c['trial'] is None else len(c['trial']['raw'])),
            action_open=self.manual is not None,automatic_detector_ready=self.detector is not None,
            boundary_policy='manual_cued_or_estimated_auto',decision_after_interval=True,
            physical_validation_proven=False,default_promoted=False)

    def begin(self,kind,shots):
        if self.capture is not None or kind not in ('personal','session') or type(shots) is not int or not 1<=shots<=5:
            raise ValueError('Save/cancel existing full-action capture; valid kind and1..5 shots required')
        if kind=='session' and (self.personal is None or self.personal.session_id==self.session_id):
            raise ValueError('A long-term full-action profile from another session is required')
        self.reset();classes=list(self.temporal.class_names);classes.remove(self.window.rest_label);classes.insert(0,self.window.rest_label)
        self.capture=dict(kind=kind,order=classes*shots,rows=[],trial=None)

    def start_trial(self):
        c=self.capture
        if c is None or c['trial'] is not None or len(c['rows'])==len(c['order']):raise ValueError('No full-action calibration trial ready')
        c['trial']=dict(raw=[],filtered=[],start=None,id='cue:'+self.session_id+':'+uuid.uuid4().hex)

    def _batch(self,rows,kind='complete_cued',raw=False):
        return TemporalBoutBatchV1(tuple(np.asarray(r['raw' if raw else 'filtered']) for r in rows),
            tuple(r['id'] for r in rows),tuple(r['recording'] for r in rows),tuple(r['start'] for r in rows),
            250.,self.channels,'raw_pre_software_highpass' if raw else self.window.preprocessing_id,kind)

    def end_trial(self,quality_mode):
        if quality_mode not in ('off','structural','soft'):raise ValueError('Explicit raw quality policy required')
        c=self.capture;t=None if c is None else c['trial']
        if t is None or not 250<=len(t['raw'])<=7500:raise ValueError('Capture a complete1..30 second action before ending it')
        row={**t,'recording':t['id'],'label':c['order'][len(c['rows'])]}
        x=np.asarray(row['raw']);windows=np.stack([x[a:a+50] for a in range(0,len(x)-49,10)])
        observation=self.window.gate.observe(FeatureBatch(windows,250.),[row['id']]*len(windows),observed_channel_ids=self.channels)
        bad=(quality_mode=='structural' and observation['trial_structural_invalid_channels'].any() or
             quality_mode=='soft' and observation['trial_soft_channel_quality'].max()==0)
        c['trial']=None
        if bad:return dict(calibration_rejected=True,reason='Raw signal quality rejected; retry this action')
        self._batch([row]).validate();c['rows'].append(row)
        return dict(calibration_rejected=False)

    def save(self,path):
        c=self.capture
        if c is None or c['trial'] is not None or len(c['rows'])!=len(c['order']):raise ValueError('Complete every cued full-action trial first')
        b=self._batch(c['rows']);labels={r['id']:r['label'] for r in c['rows']}
        p=self.temporal.enroll(b,labels,user_id=self.user,session_id=self.session_id,
            personal=self.personal if c['kind']=='session' else None)
        rest=[r for r in c['rows'] if r['label']==self.window.rest_label]
        detector=AutonomousBoutDetectorV1().fit_rest(np.concatenate([r['filtered'] for r in rest]),250.,
            source_trial_ids=[r['id'] for r in rest])
        target=Path(path);companions=[target,Path(str(target)+'.bouts.npz'),Path(str(target)+'.labels.json'),Path(str(target)+'.capture.json')]
        if any(v.exists() for v in companions):raise FileExistsError('Full-action profile/companions already exist; choose a new path')
        payload=pickle.dumps(p,protocol=pickle.HIGHEST_PROTOCOL);packed=pickle.dumps(detector,protocol=pickle.HIGHEST_PROTOCOL)
        manifest=dict(schema='temporal_live_profile_v1',contract_id=self.temporal.contract_id,profile_id=p.profile_id,
            profile_sha256=hashlib.sha256(payload).hexdigest(),detector_sha256=hashlib.sha256(packed).hexdigest())
        created=[]
        try:
            with companions[1].open('xb') as f:
                created.append(companions[1]);np.savez_compressed(f,raw=np.concatenate([r['raw'] for r in c['rows']]),
                    filtered=np.concatenate(b.sequences),offsets=np.r_[0,np.cumsum([len(x) for x in b.sequences])],
                    trial_ids=np.array(b.trial_ids),recording_ids=np.array(b.recording_ids),starts=np.array(b.starts),
                    channel_ids=np.array(b.channel_ids),sample_rate_hz=np.array(250.),preprocessing_id=np.array(b.preprocessing_id),
                    boundary_kind=np.array('complete_cued'))
            with companions[2].open('x',encoding='utf8') as f:created.append(companions[2]);json.dump(labels,f)
            with companions[3].open('x',encoding='utf8') as f:
                created.append(companions[3]);json.dump(dict(schema='temporal_live_capture_v1',trial_ids=b.trial_ids,
                    profile_id=p.profile_id,user_id=self.user,session_id=self.session_id,kind=c['kind'],
                    boundaries='user_marked_cued_intervals_not_physiological_ground_truth',settle_samples_discarded=0,
                    raw_and_filtered_samples_saved=True,detector_fit_only_neutral_calibration=True,
                    physical_validation_proven=False,default_promoted=False),f)
            with target.open('xb') as f:
                created.append(target)
                with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr('manifest.json',json.dumps(manifest));z.writestr('profile.pkl',payload);z.writestr('detector.pkl',packed)
        except Exception:
            for v in created:v.unlink()
            raise
        if c['kind']=='personal':self.personal=p;self.long_detector=detector;self.session=self.local_detector=None
        else:self.session=p;self.local_detector=detector
        self.reset();return dict(saved_temporal_profile_path=str(target),temporal_saved_kind=p.kind)

    def _load(self,path):
        with zipfile.ZipFile(path) as z:
            if sorted(z.namelist())!=['detector.pkl','manifest.json','profile.pkl']:raise ValueError('Unexpected full-action profile members')
            m=json.loads(z.read('manifest.json'));payload=z.read('profile.pkl');packed=z.read('detector.pkl')
        if (m.get('schema')!='temporal_live_profile_v1' or m.get('contract_id')!=self.temporal.contract_id or
            m.get('profile_sha256')!=hashlib.sha256(payload).hexdigest() or m.get('detector_sha256')!=hashlib.sha256(packed).hexdigest()):
            raise ValueError('Full-action bundle source/checksum differs')
        p=pickle.loads(payload);d=pickle.loads(packed);self.temporal._profile(p,self.user)
        if (p.profile_id!=m['profile_id'] or not isinstance(d,AutonomousBoutDetectorV1) or d.rate_!=250. or d.channels_!=8
                or not set(d.source_trial_ids_)<=set(p.trial_ids) or not np.isfinite([d.off_,d.on_]).all() or not 0<=d.off_<d.on_):
            raise ValueError('Full-action detector/profile identity differs')
        return p,d

    def load(self,personal,session):
        if self.capture is not None:raise ValueError('Save/cancel full-action calibration before loading profiles')
        p,d=(None,None) if not personal else self._load(personal)
        s,e=(None,None) if not session else self._load(session)
        if p is not None and p.kind!='personal':raise ValueError('Long-term full-action profile required')
        if s is not None and (p is None or s.kind!='session' or s.personal_profile_id!=p.profile_id or s.session_id!=self.session_id):
            raise ValueError('Full-action session parent/session identity differs')
        self.personal,self.long_detector,self.session,self.local_detector=p,d,s,e;self.reset()

    def action_start(self):
        if self.personal is None or self.manual is not None:raise ValueError('Load a full-action personal profile before starting a new action')
        self.manual=dict(raw=[],filtered=[],start=None,id='manual:'+self.session_id+':'+uuid.uuid4().hex)

    def _read(self,row,kind,available_at,**options):
        if self.personal is None:raise ValueError('Full-action personal profile required')
        r=self.adapter.predict(self._batch([row],kind),temporal_personal=self.personal,temporal_session=self.session,
            user_id=self.user,session_id=self.session_id,raw_batch=self._batch([row],kind,raw=True),**options)
        q=r['temporal']['arms']['base_full'][0]
        return dict(trial_id=row['id'],recording_id=row['recording'],start=row['start'],end=row['start']+len(row['raw']),
            available_at_sample_index=available_at,boundary_kind=kind,decision_after_interval=True,
            label=str(r['predicted_labels'][0]),rejected=bool(r['rejected'][0]),classes=list(self.temporal.class_names),
            probabilities=q,base_probabilities=r['temporal']['arms']['base'][0],
            DTW_probabilities=r['temporal']['arms']['DTW_blended'][0],signature_probabilities=r['temporal']['arms']['signature_blended'][0],
            window_unrepresented_tail_samples=r['window_unrepresented_tail_samples'][0],
            physical_validation_proven=False,default_promoted=False)

    def action_end(self,**options):
        t=self.manual
        if t is None or not 250<=len(t['raw'])<=7500:raise ValueError('A complete1..30 second manual action is required')
        row={**t,'recording':t['id']};self.manual=None
        return self._read(row,'complete_cued',self.last,**options)

    def feed(self,raw,indices,*,automatic=False,**options):
        raw=np.asarray(raw,dtype=float);indices=np.asarray(indices)
        if (raw.ndim!=2 or raw.shape[1]!=8 or indices.shape!=(len(raw),) or indices.dtype.kind not in 'iu'
                or not np.isfinite(raw).all() or (len(indices) and (indices[0]<0 or np.any(np.diff(indices)!=1)))):
            self.reset();raise ValueError('Finite contiguous native eight-channel samples/indices required')
        if not len(raw):return []
        if self.last is not None and indices[0]!=self.last+1:self.reset();raise ValueError('Full-action sampling discontinuity; restart required')
        x=raw
        for i,sos in enumerate(self.filters):x,self.zi[i]=sosfilt(sos,x,axis=0,zi=self.zi[i])
        x=x.astype(np.float32)[:,self.order];raw=raw[:,self.order];self.last=int(indices[-1])
        t=self.manual if self.capture is None else self.capture['trial']
        if t is not None:
            if len(t['raw'])+len(raw)>7500:
                if self.capture is None:self.manual=None
                else:self.capture['trial']=None
                raise ValueError('Full-action capture exceeded30 seconds; incomplete action discarded')
            if t['start'] is None:t['start']=int(indices[0])
            t['raw'].extend(raw.copy());t['filtered'].extend(x.copy())
        events=[]
        if automatic:
            if self.detector is None or self.personal is None:raise ValueError('Registered full-action profile and Rest-fitted detector required')
            combined=np.concatenate((self.history,raw));start=int(indices[0])-len(self.history)
            for b in self.detector.feed(x,int(indices[0]),trial_id=self.recording):
                row=dict(id=f'{self.recording}:{b.start}:{b.end}',recording=self.recording,start=b.start,
                    raw=combined[b.start-start:b.end-start],filtered=b.emg)
                events.append(self._read(row,'estimated',self.last,**options))
        self.history=np.concatenate((self.history,raw))[-7500:]
        return events
