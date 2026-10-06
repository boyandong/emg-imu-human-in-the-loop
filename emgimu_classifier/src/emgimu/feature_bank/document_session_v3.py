"""F8 domain descriptor; source/current sessions of one explicitly identified user.

This opt-in version assembles the goal's four descriptor blocks. It does not
change the historical summary export or claim that a routing policy improves.
"""
from itertools import combinations
import numpy as np

from .activation_profile import trial_weights
from .affine_spd_anchor import document_spd_matrices, affine_spd_distance
from .session_shift_summary import FamilySessionShiftSummary


class DocumentSessionDescriptorV3(FamilySessionShiftSummary):
    @staticmethod
    def _session_contract(batch, trials, session_ids):
        sessions=np.asarray(session_ids,dtype=object)
        if sessions.ndim!=1 or len(sessions)!=batch.windows:
            raise ValueError('Session IDs must align with windows')
        if any(not isinstance(s,str) or not s.strip() for s in sessions):
            raise ValueError('Explicit nonempty session identities required')
        trials=np.asarray(trials,dtype=object)
        if any(len(set(sessions[trials==t]))!=1 for t in set(trials)):
            raise ValueError('A trial cannot span multiple sessions')
        return frozenset(sessions)

    def fit_long_term(self,batch,labels,trials,*,user_id,session_ids,ring_topology=False,rest_label=None):
        if not isinstance(user_id,str) or not user_id.strip():
            raise ValueError('Explicit nonempty user identity required')
        if batch.channels!=8:
            raise ValueError('Document F8 V3 requires eight native channels')
        labels,trials=self._trial_contract(batch,labels,trials)
        sessions=self._session_contract(batch,trials,session_ids)
        super().fit_long_term(batch,labels,trials,ring_topology=ring_topology,rest_label=rest_label)
        self.user_id_=user_id
        self.source_session_ids_=sessions
        return self

    def _profiles(self,batch,labels,trials):
        profiles=super()._profiles(batch,labels,trials)
        y,ids=self._trial_contract(batch,labels,trials)
        covariance=document_spd_matrices(batch.emg)
        observations=self.quality_.transform(batch)
        # Same source-fixed relative quality rule as QualityFamily. No physical
        # fault attribution is inferred from these scores.
        blocks=observations[:,:6*batch.channels].reshape(batch.windows,6,batch.channels)
        quality=1.-np.clip(np.maximum.reduce((blocks[:,0],blocks[:,2],blocks[:,3],
                                             np.clip(np.abs(blocks[:,5])/6.,0.,1.))),0.,1.)
        for label in profiles:
            mask=y==label;weights=trial_weights(ids[mask]);weights/=weights.sum()
            profiles[label]['covariance']=np.tensordot(weights,covariance[mask],axes=(0,0))
            profiles[label]['channel_quality']=np.tensordot(weights,quality[mask],axes=(0,0))
            if 'ring' in profiles[label]:
                # F3a RLCS only, rather than the bundled CES/ringcov coordinates.
                profiles[label]['ring']=profiles[label]['ring'][:8]
        return profiles

    @staticmethod
    def _assemble(source,current):
        classes=sorted(source)
        if set(classes)!=set(current):raise ValueError('Source/current classes differ')
        families=['pattern','log_bands','covariance','channel_quality']
        if 'ring' in source[classes[0]]:families.append('ring')
        values=[];names=[]
        def put(name,value):
            names.append(name);values.append(float(value))
        for metric in ('residual_norm','cosine_agreement'):
            for family in families:
                for label in classes:
                    a=np.ravel(source[label][family]);b=np.ravel(current[label][family])
                    value=(np.linalg.norm(b-a) if metric=='residual_norm' else
                           np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-10))
                    put(f'F8.{family}.{metric}.{label}',value)
        for family in families:
            distance=affine_spd_distance if family=='covariance' else lambda a,b:np.linalg.norm(a-b)
            for a,b in combinations(classes,2):
                put(f'F8.{family}.geometry_delta.{a}.{b}',
                    distance(current[a][family],current[b][family])-
                    distance(source[a][family],source[b][family]))
        for label in classes:
            a,b=source[label],current[label]
            put(f'F8.log_scale_shift.{label}',b['log_scale']-a['log_scale'])
            put(f'F8.pattern_shift.{label}',np.linalg.norm(b['pattern']-a['pattern']))
            put(f'F8.spectral_shift.{label}',np.abs(b['log_bands']-a['log_bands']).mean())
            put(f'F8.spd_shift.{label}',affine_spd_distance(a['covariance'],b['covariance']))
            if 'ring' in a:put(f'F8.rlcs_shift.{label}',np.linalg.norm(b['ring']-a['ring']))
            for channel,value in enumerate(b['channel_quality']-a['channel_quality'],1):
                put(f'F8.channel_quality_shift.{label}.ch{channel}',value)
        vector=np.asarray(values,dtype=np.float64)
        if not np.isfinite(vector).all():raise ValueError('Nonfinite session descriptor')
        return vector,tuple(names)

    def from_calibration(self,batch,labels,trials,*,user_id,session_ids):
        if not hasattr(self,'user_id_'):raise RuntimeError('Fit the document F8 profile first')
        if user_id!=self.user_id_:raise ValueError('F8 requires the same user')
        labels,trials=self._trial_contract(batch,labels,trials)
        sessions=self._session_contract(batch,trials,session_ids)
        if len(sessions)!=1:raise ValueError('Current calibration must describe one session')
        if self.source_session_ids_.intersection(sessions):
            raise ValueError('Current calibration must come from new sessions')
        result=super().from_calibration(batch,labels,trials)
        current=self._profiles(batch,labels,trials)
        vector,names=self._assemble(self.reference_,current)
        for label in current:
            result['classes'][str(label)]['channel_quality_score_shift']=(
                current[label]['channel_quality']-self.reference_[label]['channel_quality']).tolist()
        result.update({'version':'document-f8-v3','user_id':self.user_id_,
                       'source_sessions':sorted(self.source_session_ids_),
                       'current_sessions':sorted(sessions),'phi_session':vector.tolist(),
                       'phi_feature_names':list(names),'phi_dimension':len(vector),
                       'descriptor_scope':'same-user domain summary; not a gesture prediction or validated router'})
        return result
