"""Causal chunk-to-estimated-bout G5 composition; emits after release confirmation."""
import copy
from .autonomous_bouts_v1 import AutonomousBoutDetectorV1
from .detected_g5_reader_v1 import DetectedG5ReaderV1


class AutonomousG5StreamV1:
    def __init__(self, detector, reader):
        if not isinstance(detector, AutonomousBoutDetectorV1) or not hasattr(detector, 'on_'):
            raise ValueError('Source-Rest-fitted detector required')
        if not isinstance(reader, DetectedG5ReaderV1):
            raise ValueError('Source-fitted G5 reader required')
        if detector.channels_ != 4 or detector.rate_ != 200. or detector.policy[3] < 1.:
            raise ValueError('Native four-channel 200Hz detector with minimum one-second bouts required')
        self.detector = copy.deepcopy(detector)
        self.detector.reset()
        self.reader = reader
        self.source_ids = frozenset(detector.source_trial_ids_) | reader.source_recording_ids

    def feed(self, samples, first_sample_index, *, recording_id):
        if not isinstance(recording_id, str) or not recording_id.strip() or recording_id in self.source_ids:
            self.detector.reset()
            raise ValueError('Explicit recording disjoint from detector and classifier source required')
        try:
            bouts = self.detector.feed(samples, first_sample_index, trial_id=recording_id)
            outputs = []
            for bout in bouts:
                output = self.reader.read(bout, recording_id=recording_id)
                output['emitted_at_sample'] = self.detector.next_-1
                output['delivery_delay_samples'] = self.detector.next_-bout.end
                outputs.append(output)
            return outputs
        except Exception:
            self.detector.reset()
            raise

    def finish(self):
        """Discard unclosed action and reset stream; return end-censoring status."""
        return self.detector.finish()
