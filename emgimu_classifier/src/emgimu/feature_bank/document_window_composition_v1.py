"""Source-trained composition of the document's independently tested EMG views.

Eight physical channel identities do not establish circular electrode geometry.
F3a/F3c and anatomical F6 therefore remain unavailable. F4d is context only;
F5b/c require the separate complete-bout workflow, not these 200 ms windows.
"""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .core import FeatureBatch
from .document_signal import RestNoiseLocalDetailFamily
from .document_scale_v3 import DocumentScalePatternV3
from .spec_spatial_v3 import SpecTraceCovarianceV3, SpecSpdTangentV3
from .document_ces_v3 import DocumentCesFamilyV3
from .document_spectral_v3 import DocumentSpectralStateV3
from .document_temporal_v3 import DocumentTemporalFormV3
from .extended_window_decision_v1 import SourceCsp250V1, CspQualityGateV1, ExtendedWindowDecisionV1
from .frozen_emg_provider_bank_v1 import FrozenEmgProviderBankV1
from .personal_session_workflow_v1 import PersonalSessionWorkflowV1
from .personal_session_stream_v2 import load_gate

SCHEMA = 'document_window_composition_v1'
GROUPS = ('F0', 'F1', 'F2ac', 'F3b', 'F4abc', 'F5window', 'F2b')
CLASSES = ('fist', 'index_pinch', 'neutral', 'open_hand')
CHANNELS = tuple(f'CH{i+1}' for i in range(8))
PREPROCESSING = 'song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1'
TYPES = ((RestNoiseLocalDetailFamily,), (DocumentScalePatternV3,),
         (SpecTraceCovarianceV3, SpecSpdTangentV3), (DocumentCesFamilyV3,),
         (DocumentSpectralStateV3,), (DocumentTemporalFormV3,), (SourceCsp250V1,))
FAMILY_GROUPS = {'F0': ('F0',), 'F1': ('F1',), 'F2': ('F2ac', 'F2b'),
                 'F3': ('F3b',), 'F4': ('F4abc',), 'F5': ('F5window',)}


def document_families():
    """New instances on every call; no mutation of older source packages."""
    return dict(zip(GROUPS, ([RestNoiseLocalDetailFamily(rest_label='neutral')],
        [DocumentScalePatternV3()], [SpecTraceCovarianceV3(), SpecSpdTangentV3()],
        [DocumentCesFamilyV3()], [DocumentSpectralStateV3()],
        [DocumentTemporalFormV3()], [SourceCsp250V1()])))


def source_trial_features(families, batch, trial_ids):
    ids = np.asarray(trial_ids)
    if ids.shape != (batch.windows,) or any(not isinstance(t, str) or not t.strip() for t in ids):
        raise ValueError('Explicit window-aligned source trial identities required')
    unique = np.unique(ids)
    columns = []
    for family in families:
        values = family.transform(batch)
        columns.append(np.stack([values[ids == t].mean(0) for t in unique]))
    return np.concatenate(columns, axis=1), tuple(unique)


def fit_document_source(batch, labels, trial_ids, *, forbidden_trials=(), seed=20261011):
    """Fit each representation/scaler/classifier using source trials alone.

    Call separately inside each source OOF fold; temperatures and population
    weights must be fitted on source OOF predictions by the experiment runner.
    """
    ids = np.asarray(trial_ids)
    y = np.asarray(labels)
    if ((batch.sample_rate_hz, batch.emg.shape[1], batch.channels) != (250., 50, 8)
            or not batch.windows or ids.shape != (batch.windows,) or y.shape != ids.shape
            or any(not isinstance(t, str) or not t.strip() for t in ids)
            or set(ids) & set(forbidden_trials) or set(y) != set(CLASSES)):
        raise ValueError('Disjoint source trials, all four classes and eight-channel250Hz/50sample input required')
    unique = tuple(np.unique(ids))
    if any(len(set(y[ids == t])) != 1 for t in unique):
        raise ValueError('Every independent source trial needs one label')
    targets = np.array([y[ids == t][0] for t in unique])
    families, models, metadata = document_families(), {}, {}
    for name, views in families.items():
        for family in views:
            if isinstance(family, SourceCsp250V1):
                family.fit(batch, y, source_trial_ids=ids)
            else:
                family.fit(batch, y)
            family.document_source_trials_ = unique
        x, axis = source_trial_features(views, batch, ids)
        assert axis == unique
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1., class_weight='balanced', max_iter=2000,
                                   random_state=seed).fit(scaler.transform(x), targets)
        if tuple(model.classes_) != CLASSES or model.n_iter_.max() >= 2000:
            raise ValueError('Source class axis or classifier convergence failed')
        names = tuple(n for family in views for n in family.feature_names)
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError('Feature names and source model dimensions differ')
        models[name] = (scaler, model)
        metadata[name] = dict(dimension=x.shape[1], feature_names=names,
            implementations=tuple(type(f).__name__ for f in views), iterations=model.n_iter_.tolist())
    return families, models, metadata


class DocumentWindowDecisionV1(ExtendedWindowDecisionV1):
    def __init__(self, base, gate, *, anchor_mix=.5):
        bank = base.bank
        if bank.providers_ != GROUPS or bank.classes_ != CLASSES or bank.sensor_contract_ != (250., 50, 8):
            raise ValueError('Document source provider/class/sensor axis differs')
        for group, expected in zip(GROUPS, TYPES):
            views = bank.families_[group]
            if tuple(type(f) for f in views) != expected or any(
                    tuple(getattr(f, 'document_source_trials_', ())) != tuple(bank.policy_.source_trials)
                    for f in views):
                raise ValueError('Document implementation or source-fold provenance differs')
        super().__init__(base, gate, anchor_mix=anchor_mix)
        self.policy_id = hashlib.sha256((SCHEMA+'|'+self.policy_id).encode()).hexdigest()
        self.contract_id = self.policy_id

    def predict(self, *args, **kwargs):
        result = super().predict(*args, **kwargs)
        result.update(document_composition_schema=SCHEMA,
            scope='Source-refitted document F0/F1/F2a,b,c/F3b/F4a,b,c/F5a window composition; operational F7/F8/raw F9 and context-only F4d. Ring F3a,c and anatomical F6 unavailable; full-bout F5b,c separate. No hardware, independent-cohort or default efficacy claim.')
        return result


def build_document_workflow(bank, source_gate):
    base = PersonalSessionWorkflowV1(bank, channel_ids=CHANNELS,
        preprocessing_id=PREPROCESSING, rest_label='neutral',
        quality_options={'line_frequency_hz': 50, 'pre_highpass_available': False})
    return DocumentWindowDecisionV1(base, CspQualityGateV1(source_gate))


def load_document_workflow(package, policy, gate_package, gate_results):
    config = json.loads(Path(policy).read_text(encoding='utf8'))
    payload = Path(package).read_bytes()
    if config.get('schema') != SCHEMA or hashlib.sha256(payload).hexdigest() != config.get('source_bank_sha256'):
        raise ValueError('Document source package/schema checksum differs')
    bank = pickle.loads(payload)
    if not isinstance(bank, FrozenEmgProviderBankV1) or bank.bank_id_ != config.get('source_bank_id'):
        raise ValueError('Document source bank identity differs')
    workflow = build_document_workflow(bank, load_gate(gate_package, gate_results))
    if workflow.policy_id != config.get('decision_policy_id'):
        raise ValueError('Document source decision identity differs')
    return workflow
