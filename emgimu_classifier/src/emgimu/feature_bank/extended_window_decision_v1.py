"""Seven-provider source CSP extension and explicitly contextual F4d coordinates.

F4d is never a classifier provider. The six historical source models remain
unchanged; this workflow/profile identity is distinct from the six-provider one.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import zipfile
import numpy as np
from .document_signal import DocumentCspFamily
from .personal_session_decision_v1 import PersonalSessionDecisionV1
from .source_quality_gate_v1 import SourceQualityGateV1
from .calibration import late_fusion_decision


class SourceCsp250V1(DocumentCspFamily):
    def fit(self, batch, labels=None, *, source_trial_ids):
        ids = np.asarray(source_trial_ids, dtype=object)
        if ((batch.sample_rate_hz, batch.emg.shape[1], batch.channels) != (250., 50, 8)
                or ids.shape != (batch.windows,)
                or any(not isinstance(t, str) or not t.strip() for t in ids)):
            raise ValueError('Source-fold Song250 windows and explicit trial provenance required')
        super().fit(batch, labels)
        self.source_trial_ids_ = tuple(np.unique(ids))
        self.sensor_contract_ = (250., 50, 8)
        return self

    def transform(self, batch):
        self._check()
        if (batch.sample_rate_hz, batch.emg.shape[1], batch.channels) != self.sensor_contract_:
            raise ValueError('CSP source sample-rate/window/channel contract differs')
        return super().transform(batch)


class CspQualityGateV1(SourceQualityGateV1):
    def __init__(self, frozen_gate):
        if not isinstance(frozen_gate, SourceQualityGateV1) or not hasattr(frozen_gate, 'policy_id'):
            raise ValueError('Already source-fitted raw gate required')
        self.__dict__.update(copy.deepcopy(frozen_gate.__dict__))
        self.parent_policy_id = frozen_gate.policy_id
        self.policy_id = hashlib.sha256(('csp_min_quality_v1|'+frozen_gate.policy_id).encode()).hexdigest()

    def decide(self, providers, weights, raw_batch, trial_ids, *, observed_channel_ids,
               class_names, mode, provider_trial_ids):
        if mode not in ('off', 'structural', 'soft'):
            raise ValueError('Explicit off/structural/soft mode required')
        observation = self.observe(raw_batch, trial_ids, observed_channel_ids=observed_channel_ids)
        names = tuple(providers)
        if (set(provider_trial_ids) != set(names) or
                any(tuple(provider_trial_ids[g]) != observation['trial_ids'] for g in names)):
            raise ValueError('Every provider needs the exact raw-quality trial axis')
        if mode == 'structural':
            channel = (~observation['trial_structural_invalid_channels']).astype(float)
            quality = {g: channel.min(1) for g in names}
        elif mode == 'soft':
            channel = observation['trial_soft_channel_quality']
            quality = {g: channel.min(1) if g in ('F0', 'F2ac', 'F2b') else channel.mean(1) for g in names}
        else:
            channel = np.ones((len(observation['trial_ids']), 8))
            quality = {g: np.ones(len(channel)) for g in names}
        decision = late_fusion_decision(providers, names, np.asarray(weights), tuple(class_names), quality,
            minimum_confidence=0.)
        effective = np.asarray(weights)[None, :]*np.column_stack([quality[g] for g in names])
        rejected = effective.sum(1) <= 1e-10
        effective[rejected] = weights
        effective /= effective.sum(1, keepdims=True)
        return dict(probabilities=decision.probabilities, labels=decision.labels,
            rejected=decision.rejected, rejection_reason=decision.rejection_reason,
            channel_quality=channel, bad_channel_count=(channel < .5).sum(1),
            structural_invalid_channels=observation['trial_structural_invalid_channels'],
            provider_weights=effective, trial_ids=observation['trial_ids'],
            policy_id=self.policy_id, mode=mode, physical_validation_proven=False, observations=observation)


@dataclass(frozen=True)
class ExtendedPersonalProfileV1:
    policy_id: str
    profile_id: str
    decision: object
    spectral_reference: np.ndarray

    @property
    def base(self):
        return self.decision.base


@dataclass(frozen=True)
class ExtendedSessionProfileV1:
    policy_id: str
    profile_id: str
    personal_profile_id: str
    decision: object
    spectral_reference: np.ndarray

    @property
    def base(self):
        return self.decision.base


class ExtendedWindowDecisionV1:
    def __init__(self, base, gate, *, anchor_mix=.5):
        if 'F2b' not in base.bank.providers_ or 'F4abc' not in base.bank.providers_:
            raise ValueError('Explicit source CSP and spectral providers required')
        family = base.bank.families_['F2b'][0]
        if not isinstance(family, SourceCsp250V1) or set(family.source_trial_ids_) != set(base.bank.policy_.source_trials):
            raise ValueError('CSP training-trial provenance differs from source bank')
        self.decision = PersonalSessionDecisionV1(base, gate=gate, anchor_mix=anchor_mix)
        self.bank, self.channels = base.bank, base.channels
        self.preprocessing_id, self.rest_label, self.gate = base.preprocessing_id, base.rest_label, gate
        self.bands = tuple(tuple(v) for v in self.bank.families_['F4abc'][0].bands_)
        self.policy_id = hashlib.sha256(json.dumps(['extended_window_decision_v1',
            self.decision.policy_id, self.bands, 'F4d_trial_balanced_context_only']).encode()).hexdigest()
        self.contract_id = self.policy_id

    def _input(self, *args, **kwargs):
        return self.decision._input(*args, **kwargs)

    def log_bands(self, batch):
        if (batch.sample_rate_hz, batch.emg.shape[1], batch.channels) != self.bank.sensor_contract_:
            raise ValueError('Spectral source sensor contract differs')
        x = np.asarray(batch.emg, float)
        x = (x-x.mean(1, keepdims=True))*np.hanning(x.shape[1])[None, :, None]
        power = abs(np.fft.rfft(x, axis=1))**2/x.shape[1]
        frequency = np.fft.rfftfreq(x.shape[1], 1/batch.sample_rate_hz)
        pieces = []
        for i, (low, high) in enumerate(self.bands):
            mask = (frequency >= low) & (frequency <= high if i == len(self.bands)-1 else frequency < high)
            if not mask.any():
                raise ValueError('Empty source frequency band')
            pieces.append(np.log(power[:, mask].sum(1)+1e-10))
        return np.concatenate(pieces, axis=1).astype(np.float32)

    def _reference(self, batch, ids, channels, preprocessing):
        mapped = self._input(batch, observed_channel_ids=channels, preprocessing_id=preprocessing)
        x, ids = self.log_bands(mapped).astype(float), np.asarray(ids)
        return np.stack([x[ids == t].mean(0) for t in np.unique(ids)]).mean(0)

    def _id(self, decision, reference):
        return hashlib.sha256(self.policy_id.encode()+decision.profile_id.encode()+
            np.ascontiguousarray(reference, dtype=np.float64).tobytes()).hexdigest()

    def _personal(self, profile, user):
        if not isinstance(profile, ExtendedPersonalProfileV1) or profile.policy_id != self.policy_id:
            raise ValueError('Extended personal policy differs')
        self.decision._personal(profile.decision, user)
        self._reference_identity(profile)

    def _reference_identity(self, profile):
        if (profile.spectral_reference.shape != (len(self.bands)*8,)
                or not np.isfinite(profile.spectral_reference).all()
                or self._id(profile.decision, profile.spectral_reference) != profile.profile_id):
            raise ValueError('Extended spectral profile identity differs')

    def _session(self, profile, personal, user, session):
        self._personal(personal, user)
        if profile is not None:
            if (not isinstance(profile, ExtendedSessionProfileV1) or profile.policy_id != self.policy_id
                    or profile.personal_profile_id != personal.profile_id):
                raise ValueError('Extended session policy/personal differs')
            self._reference_identity(profile)
        self.decision._session(None if profile is None else profile.decision, personal.decision, user, session)

    def enroll_user(self, batch, trial_ids, trial_labels, **kwargs):
        p = self.decision.enroll_user(batch, trial_ids, trial_labels, **kwargs)
        reference = self._reference(batch, trial_ids, kwargs['observed_channel_ids'], kwargs['preprocessing_id'])
        return ExtendedPersonalProfileV1(self.policy_id, self._id(p, reference), p, reference)

    def calibrate_session(self, batch, trial_ids, trial_labels, *, personal, **kwargs):
        self._personal(personal, kwargs['user_id'])
        p = self.decision.calibrate_session(batch, trial_ids, trial_labels, personal=personal.decision, **kwargs)
        reference = self._reference(batch, trial_ids, kwargs['observed_channel_ids'], kwargs['preprocessing_id'])
        return ExtendedSessionProfileV1(self.policy_id, self._id(p, reference), personal.profile_id, p, reference)

    def predict(self, batch, trial_ids, *, personal=None, session=None, **kwargs):
        if personal is not None:
            self._session(session, personal, kwargs['user_id'], kwargs['session_id'])
        elif session is not None:
            raise ValueError('Session requires personal profile')
        result = self.decision.predict(batch, trial_ids, personal=None if personal is None else personal.decision,
            session=None if session is None else session.decision, **kwargs)
        context = None
        if personal is not None:
            mapped = self._input(batch, observed_channel_ids=kwargs['observed_channel_ids'], preprocessing_id=kwargs['preprocessing_id'])
            values = self.log_bands(mapped).astype(float)-personal.spectral_reference
            ids = np.asarray(trial_ids)
            context = dict(window_minus_long=values, window_trial_ids=tuple(ids),
                trial_ids=tuple(np.unique(ids)),
                trial_minus_long=np.stack([values[ids == t].mean(0) for t in np.unique(ids)]),
                session_minus_long=None if session is None else session.spectral_reference-personal.spectral_reference,
                bands=self.bands, channel_ids=self.channels,
                role='Context/session coordinates only; not a gesture provider or fatigue measurement')
        result.update(decision_policy_id=self.policy_id,
            personal_profile_id=None if personal is None else personal.profile_id,
            session_profile_id=None if session is None else session.profile_id, spectral_context=context,
            scope='Seven declared EMG window providers including source-only CSP, operational F7/F8/raw F9 and context-only F4d; not full-bout F5, physical ring/F6 or device efficacy.')
        return result

    def save_profile(self, profile, path):
        if not isinstance(profile, (ExtendedPersonalProfileV1, ExtendedSessionProfileV1)) or profile.policy_id != self.policy_id:
            raise ValueError('Extended profile policy differs')
        self._reference_identity(profile)
        payload = pickle.dumps(profile, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = dict(schema='extended_window_profile_v1', policy_id=self.policy_id,
            profile_id=profile.profile_id, user_id=profile.base.user_id, session_id=profile.base.session_id,
            kind='personal' if isinstance(profile, ExtendedPersonalProfileV1) else 'session',
            payload_sha256=hashlib.sha256(payload).hexdigest())
        with Path(path).open('xb') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True))
                archive.writestr('profile.pkl', payload)

    def load_profile(self, path, *, user_id, session_id=None, personal=None):
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ['manifest.json', 'profile.pkl']:
                raise ValueError('Unexpected extended profile archive members')
            m, payload = json.loads(archive.read('manifest.json')), archive.read('profile.pkl')
        kind = 'personal' if session_id is None else 'session'
        if (m.get('schema') != 'extended_window_profile_v1' or m.get('policy_id') != self.policy_id
                or m.get('kind') != kind or m.get('user_id') != user_id
                or (session_id is not None and m.get('session_id') != session_id)
                or m.get('payload_sha256') != hashlib.sha256(payload).hexdigest()):
            raise ValueError('Extended profile checksum/user/session identity differs')
        profile = pickle.loads(payload)
        if kind == 'personal':
            self._personal(profile, user_id)
        else:
            self._session(profile, personal, user_id, session_id)
        if profile.profile_id != m['profile_id'] or profile.base.session_id != m['session_id']:
            raise ValueError('Extended manifest/payload identity differs')
        return profile
