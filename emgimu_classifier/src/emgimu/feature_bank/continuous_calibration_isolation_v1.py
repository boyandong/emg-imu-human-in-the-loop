"""Recording-level calibration exclusion on an unchanged continuous event axis.

Half-open sample bounds are used throughout. This module does not detect new
events, read evaluation labels for fitting, or rematch retained events.
"""
import numpy as np


def numbered_trial_intervals(counter, original_label, *, user, day, posture):
    """Recover converter trial identities before any resampling or cropping."""
    counter = np.asarray(counter)
    labels = np.asarray(original_label)
    if (counter.ndim != 1 or labels.shape != counter.shape or not len(counter)
            or not np.isfinite(counter).all() or not np.isfinite(labels).all()
            or np.any(counter < 0) or np.any(counter != np.floor(counter))
            or np.any(labels != np.floor(labels)) or not set(labels) <= set(range(1, 7))
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (user, day, posture))):
        raise ValueError('Finite integer counter/label axes and native identity required')
    edges = np.r_[0, np.flatnonzero(counter[1:] != counter[:-1]) + 1, len(counter)]
    result = {}
    for start, end in zip(edges[:-1], edges[1:]):
        number = int(counter[start])
        if number == 0:
            continue
        gestures = set(labels[start:end]) - {1}
        if len(gestures) != 1:
            raise ValueError('A numbered recording must contain exactly one original active gesture')
        gesture = int(next(iter(gestures)))
        trial = f'unibo-u{user:02d}-d{day:02d}-p{posture:02d}-g{gesture:02d}-r{number:02d}'
        if trial in result:
            raise ValueError('Ambiguous duplicate numbered recording identity')
        result[trial] = (int(start), int(end))
    return result


def projected_exclusion_mask(length, intervals, *, source_rate, target_rate, guard_source_samples):
    """Exclude output samples whose source-time centers touch guarded records."""
    if (isinstance(length, bool) or not isinstance(length, int) or length < 1
            or not np.isfinite(source_rate) or not np.isfinite(target_rate)
            or min(source_rate, target_rate) <= 0
            or isinstance(guard_source_samples, bool)
            or not isinstance(guard_source_samples, int) or guard_source_samples < 0):
        raise ValueError('Valid sample axis, rates and nonnegative integer guard required')
    centers = np.arange(length, dtype=float) * source_rate / target_rate
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (start, end))
                or start < 0 or end <= start):
            raise ValueError('Nonempty half-open source intervals required')
        mask |= (centers >= start - guard_source_samples) & (centers < end + guard_source_samples)
    return mask


def isolate_fixed_pairs(event_bounds, reference_bounds, pairs, exclusion_mask):
    """Remove intersecting intervals and both ends of each affected frozen pair.

    Paired exclusion is symmetric: an event straddling calibration cannot leave
    its matched reference in the denominator, and an excluded reference cannot
    leave its match as an artificial false positive. Unmatched eligible intervals
    remain, including genuine misses. No matching or boundary tuning takes place.
    """
    mask = np.asarray(exclusion_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_ or not len(mask):
        raise ValueError('Explicit nonempty boolean exclusion axis required')
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]

    def intersect(bounds):
        excluded = set()
        for i, (start, end) in enumerate(bounds):
            if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (start, end))
                    or not 0 <= start < end <= len(mask)):
                raise ValueError('Event/reference lies outside native recording')
            if prefix[end] != prefix[start]:
                excluded.add(i)
        return excluded

    direct_events = intersect(event_bounds)
    direct_references = intersect(reference_bounds)
    excluded_events = direct_events.copy()
    excluded_references = direct_references.copy()
    seen_events = set()
    seen_references = set()
    for event, reference in pairs:
        if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (event, reference))
                or not 0 <= event < len(event_bounds) or not 0 <= reference < len(reference_bounds)
                or event in seen_events or reference in seen_references):
            raise ValueError('Frozen pairs must be valid and one-to-one')
        seen_events.add(event)
        seen_references.add(reference)
        if event in direct_events or reference in direct_references:
            excluded_events.add(event)
            excluded_references.add(reference)
    retained_pairs = [(int(e), int(r)) for e, r in pairs
                      if e not in excluded_events and r not in excluded_references]
    return dict(retained_events=sorted(set(range(len(event_bounds))) - excluded_events),
                retained_references=sorted(set(range(len(reference_bounds))) - excluded_references),
                retained_pairs=retained_pairs, direct_excluded_events=sorted(direct_events),
                direct_excluded_references=sorted(direct_references),
                excluded_events=sorted(excluded_events), excluded_references=sorted(excluded_references))
