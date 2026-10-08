"""Chunked, trailing-window inference and explicit transition/hold diagnostics."""
import numpy as np
from .core import FeatureBatch


class CausalWindowRecognizerV1:
    def __init__(self, family, model, *, sample_rate_hz, hop_samples=10):
        family._check()
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0 or sample_rate_hz != family.sample_rate_hz_:
            raise ValueError('Stream rate differs from source fit')
        if type(hop_samples) is not int or not 1 <= hop_samples <= family.samples_:
            raise ValueError('Integer hop must fit within the source window')
        self.family, self.model = family, model
        self.rate, self.window, self.hop = sample_rate_hz, family.samples_, hop_samples
        self.classes = np.asarray(model.classes_).copy()
        if self.classes.ndim != 1 or len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError('Unique fitted classifier class axis required')
        self.classes.setflags(write=False)
        self.buffer = np.empty((0, 8)); self.buffer_start = 0
        self.received = 0; self.next_end = self.window-1

    def push(self, samples):
        incoming = np.asarray(samples, dtype=float)
        if incoming.ndim != 2 or incoming.shape[1] != 8 or not np.isfinite(incoming).all():
            raise ValueError('Finite ordered eight-channel samples required')
        if not np.array_equal(self.model.classes_, self.classes):
            raise ValueError('Fitted classifier class axis changed')
        combined = np.concatenate([self.buffer, incoming])
        received = self.received+len(incoming)
        ends = np.arange(self.next_end, received, self.hop, dtype=int)
        if len(ends):
            windows = np.stack([combined[e-self.window+1-self.buffer_start:e+1-self.buffer_start] for e in ends])
            probability = np.asarray(self.model.predict_proba(self.family.transform(FeatureBatch(windows, self.rate))))
            if (probability.shape != (len(ends), len(self.classes)) or not np.isfinite(probability).all()
                    or np.any(probability < 0) or not np.allclose(probability.sum(1), 1, atol=1e-8)):
                raise ValueError('Invalid classifier probability contract')
            next_end = int(ends[-1])+self.hop
        else:
            probability = np.empty((0, len(self.classes))); next_end = self.next_end
        start = max(0, next_end-self.window+1)
        self.buffer = combined[start-self.buffer_start:].copy()
        self.buffer_start, self.received, self.next_end = start, received, next_end
        return ends, probability.copy()


def transition_hold_diagnostics(truth, prediction, *, rate_hz, half_buffer_samples):
    """Versioned cue-label diagnostic, not a claim to reproduce ReactEMG metrics.

    Each reaction region may contain only old/new labels and must contain new.
    Every subsequent maintenance sample must be new. Overlapping or truncated
    reaction regions are ineligible, rather than counted as easy successes.
    """
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or truth.shape != prediction.shape or len(truth) < 2:
        raise ValueError('Aligned nonempty dense labels required')
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Integer labels required')
    if (np.any(truth < 0) or not np.isfinite(rate_hz) or rate_hz <= 0
            or not isinstance(half_buffer_samples, int) or half_buffer_samples < 1):
        raise ValueError('Known truth, positive rate and integer reaction tolerance required')
    changes = np.flatnonzero(np.diff(truth))+1
    rows = []
    for index, onset in enumerate(changes):
        left, right = int(onset-half_buffer_samples), int(onset+half_buffer_samples)
        next_onset = int(changes[index+1]) if index+1 < len(changes) else len(truth)+half_buffer_samples
        stop = min(len(truth), next_onset-half_buffer_samples)
        previous_end = int(changes[index-1]+half_buffer_samples) if index else 0
        row = {'onset_sample':int(onset), 'old_label':int(truth[onset-1]), 'new_label':int(truth[onset]),
               'reaction_start':left, 'reaction_stop':right, 'maintenance_stop':stop}
        if left < previous_end or right > len(truth) or stop <= right:
            rows.append({**row,'eligible':False,'reason':'overlapping_truncated_or_empty_hold','correct':None}); continue
        reaction = prediction[left:right]; new = row['new_label']
        hits = np.flatnonzero(reaction == new)
        bad = ~np.isin(reaction, [row['old_label'], new])
        hold = prediction[right:stop]
        reasons = []
        if not len(hits): reasons.append('no_new_label_in_reaction')
        if bad.any(): reasons.append('other_label_in_reaction')
        if np.any(hold != new): reasons.append('maintenance_error')
        rows.append({**row,'eligible':True,'correct':not reasons,'reasons':reasons,
                     'first_new_offset_seconds':float((left+hits[0]-onset)/rate_hz) if len(hits) else None,
                     'maintenance_samples':len(hold),'maintenance_error_samples':int(np.sum(hold != new)),
                     'maintenance_switches':int(np.sum(np.diff(hold) != 0))})
    eligible = [r for r in rows if r['eligible']]
    return {'events':rows,'annotated_transitions':len(rows),'eligible_transitions':len(eligible),
            'correct_transitions':sum(r['correct'] for r in eligible),
            'transition_hold_accuracy':sum(r['correct'] for r in eligible)/len(eligible) if eligible else None,
            'maintenance_switches':sum(r['maintenance_switches'] for r in eligible),
            'scope':'Nominal sample-grid cue-label diagnostic; offsets are not biological/device reaction times. Unknown predictions fail eligible events.'}
