"""Evaluate explicit normal/fault annotations; unknown is not treated as normal."""
import numpy as np


def evaluate_fault_gate(rejected, annotations, trial_ids, *, source_trial_ids, baseline_correct=None):
    decisions = np.asarray(rejected)
    labels = np.asarray(annotations,dtype=object); ids = np.asarray(trial_ids,dtype=object)
    source = tuple(source_trial_ids)
    if (not source or any(not isinstance(t,str) or not t.strip() for t in source)
            or decisions.ndim != 1 or decisions.dtype != bool or not len(decisions)
            or labels.shape != decisions.shape or ids.shape != decisions.shape
            or any(not isinstance(t,str) or not t.strip() for t in ids)):
        raise ValueError('Aligned Boolean gate decisions, annotations and explicit trial identities required')
    if set(source) & set(ids):
        raise ValueError('Fault-gate evaluation overlaps source rule fitting')
    if not set(labels) <= {'normal','fault','unknown'}:
        raise ValueError('Explicit normal, fault or unknown annotation required')
    known = labels != 'unknown'; normal = labels == 'normal'; fault = labels == 'fault'
    weight = np.zeros(len(ids),dtype=float)
    for trial in set(ids[known]):
        mask = (ids == trial) & known
        weight[mask] = 1/mask.sum()
    counts = {'true_fault_rejections':int(np.sum(decisions & fault)),
              'missed_faults':int(np.sum(~decisions & fault)),
              'normal_false_rejections':int(np.sum(decisions & normal)),
              'normal_accepted':int(np.sum(~decisions & normal)),
              'unknown_windows':int(np.sum(~known))}
    def ratio(numerator,denominator):
        return float(weight[numerator].sum()/weight[denominator].sum()) if weight[denominator].sum() else None
    metrics = {'fault_recall':ratio(decisions & fault,fault),
               'fault_precision':ratio(decisions & fault,decisions & known),
               'normal_false_rejection_rate':ratio(decisions & normal,normal)}
    harm = None
    if baseline_correct is not None:
        correct = np.asarray(baseline_correct)
        if correct.shape != decisions.shape or correct.dtype != bool:
            raise ValueError('Aligned Boolean baseline correctness required')
        harm = {'correct_normal_predictions_rejected':int(np.sum(correct & normal & decisions)),
                'correct_normal_rejection_rate':ratio(correct & normal & decisions,correct & normal)}
    return {'window_counts':counts,'trial_balanced_metrics':metrics,'baseline_harm':harm,
            'known_trials':len(set(ids[known])),'all_trials':len(set(ids)),
            'physical_validation_proven':False,
            'scope':'Descriptive caller-labelled gate evaluation with equal mass per known trial. Annotation origin must be verified separately. No threshold selection, physical-fault inference or automatic deployment promotion.'}
