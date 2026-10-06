"""Source-Rest-fitted causal onset/release detector, without gesture labels.

Detected boundaries are estimates. They never silently certify the native
coverage required by CompleteSequenceBatch or replace an oracle benchmark.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DetectedBout:
    start: int
    end: int  # exclusive, includes the confirmed release tail
    emg: np.ndarray
    sample_rate_hz: float

    @property
    def duration_seconds(self):
        return (self.end - self.start) / self.sample_rate_hz


class AutonomousBoutDetectorV1:
    """25 ms trailing RMS, source Rest thresholds and fixed hysteresis.

    Calibration accepts explicit source Rest samples only. Streaming accepts
    no labels. The first quiet interval arms the detector; a stream starting
    during activity cannot emit a falsely complete beginning. Gaps, overflow
    and stream-end censoring discard candidates instead of emitting them.
    """
    def __init__(self, *, onset_s=.08, release_s=.12, preroll_s=.10,
                 min_duration_s=1., max_duration_s=30.):
        values = (onset_s, release_s, preroll_s, min_duration_s, max_duration_s)
        if not all(np.isfinite(v) for v in values) or min(values[:2]) <= 0 or preroll_s < 0:
            raise ValueError('finite positive confirmation durations required')
        if not 0 < min_duration_s < max_duration_s:
            raise ValueError('valid minimum and maximum duration required')
        self.policy = tuple(float(v) for v in values)

    def fit_rest(self, rest, sample_rate_hz, *, source_trial_ids):
        x = np.asarray(rest, dtype=float)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError('positive sample rate required')
        if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
            raise ValueError('finite source Rest [samples,channels] required')
        ids = tuple(source_trial_ids)
        if not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('explicit unique source trial IDs required')
        if len(x) < math.ceil(sample_rate_hz):
            raise ValueError('at least one second source Rest required')
        self.rate_ = float(sample_rate_hz)
        self.channels_ = x.shape[1]
        self.width_ = max(1, round(self.rate_ * .025))
        power = np.mean(x*x, axis=1)
        smooth = np.convolve(power, np.ones(self.width_)/self.width_, mode='valid')
        activity = np.sqrt(smooth)
        median = float(np.median(activity))
        mad = float(np.median(np.abs(activity - median)))
        scale = max(1.4826*mad, median*.05, 1e-10)
        self.off_ = max(float(np.quantile(activity, .95)), median + 3*scale)
        self.on_ = max(float(np.quantile(activity, .995)), median + 6*scale, self.off_*1.2)
        self.source_trial_ids_ = ids
        self.counts_ = tuple(max(0 if i == 2 else 1, math.ceil(v*self.rate_))
                             for i, v in enumerate(self.policy))
        self.reset()
        return self

    def reset(self):
        self.next_ = None
        self.power_ = deque(maxlen=getattr(self, 'width_', 1))
        preroll = self.counts_[2] if hasattr(self, 'counts_') else 1
        self.history_ = deque(maxlen=preroll)
        self.pending_ = []
        self.start_ = None
        self.high_ = self.low_ = 0
        self.armed_ = False

    def feed(self, samples, first_sample_index, *, trial_id):
        if not hasattr(self, 'on_'):
            raise RuntimeError('source Rest fit required')
        if not isinstance(trial_id, str) or not trial_id.strip() or trial_id in self.source_trial_ids_:
            raise ValueError('explicit disjoint evaluation trial ID required')
        x = np.asarray(samples, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.channels_ or not len(x) or not np.isfinite(x).all():
            self.reset()
            raise ValueError('finite nonempty samples with fitted channels required')
        if isinstance(first_sample_index, bool) or not isinstance(first_sample_index, int) or first_sample_index < 0:
            self.reset()
            raise ValueError('nonnegative integer native sample index required')
        if self.next_ is not None and (first_sample_index != self.next_ or trial_id != self.trial_):
            self.reset()
            raise ValueError('gap, overlap or changed trial invalidates detection')
        self.trial_ = trial_id
        onset, release, preroll, minimum, maximum = self.counts_
        events = []
        for k, row in enumerate(x):
            index = first_sample_index + k
            value = row.copy()
            self.power_.append(float(np.mean(value*value)))
            activity = math.sqrt(sum(self.power_)/len(self.power_))
            ready = len(self.power_) == self.width_
            self.low_ = self.low_+1 if ready and activity <= self.off_ else 0
            if self.start_ is None:
                if self.low_ >= release:
                    self.armed_ = True
                if self.armed_ and ready and activity >= self.on_:
                    if not self.high_:
                        self.pending_ = [v.copy() for _, v in self.history_]
                        self.candidate_start_ = self.history_[0][0] if self.history_ else index
                    self.pending_.append(value)
                    self.high_ += 1
                    if self.high_ >= onset:
                        self.start_ = self.candidate_start_
                        self.low_ = 0
                else:
                    self.high_ = 0
                    self.pending_ = []
            else:
                self.pending_.append(value)
                if len(self.pending_) > maximum:
                    self.start_ = None
                    self.pending_ = []
                    self.armed_ = False
                    self.high_ = self.low_ = 0
                elif self.low_ >= release:
                    if len(self.pending_) >= minimum:
                        native = np.array(self.pending_)
                        native.setflags(write=False)
                        events.append(DetectedBout(self.start_, index+1, native, self.rate_))
                    self.start_ = None
                    self.pending_ = []
                    self.high_ = 0
                    self.armed_ = True
            self.history_.append((index, value))
        self.next_ = first_sample_index + len(x)
        return events

    def finish(self):
        """Discard an unclosed candidate; never manufacture an offset at EOF."""
        censored = self.start_ is not None or self.high_ > 0
        self.reset()
        return censored
