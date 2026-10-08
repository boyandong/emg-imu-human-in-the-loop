"""Explicit operator-supplied physical ring mapping; never infer it from Ch numbers."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .core import FeatureBatch


@dataclass(frozen=True)
class RingElectrodeLayoutV1:
    # These names identify measured channels (including differential pairs),
    # not assumed individual electrode identities.
    ring_channels: tuple[str,...]
    arm: str
    direction: str
    evidence_id: str
    topology: str

    def __post_init__(self):
        channels = tuple(self.ring_channels)
        if (len(channels) != 8 or len(set(channels)) != 8
                or any(not isinstance(c,str) or not c.strip() for c in channels)):
            raise ValueError('Eight unique explicit measured channel names required')
        if self.topology != 'circumferential_ring' or self.arm not in ('left','right'):
            raise ValueError('Explicit actual ring topology and arm required')
        if self.direction not in ('clockwise','counterclockwise'):
            raise ValueError('Explicit circumferential ordering direction required')
        if not isinstance(self.evidence_id,str) or not self.evidence_id.strip():
            raise ValueError('Physical mapping evidence reference required')
        object.__setattr__(self,'ring_channels',channels)

    @property
    def layout_id(self):
        # An annotation citation changing alone does not alter channel geometry.
        data = {'channels':self.ring_channels,'arm':self.arm,'direction':self.direction,'topology':self.topology}
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    def _apply(self,batch,observed_channel_ids):
        names = tuple(observed_channel_ids)
        if (batch.channels != 8 or len(names) != 8 or len(set(names)) != 8
                or set(names) != set(self.ring_channels)):
            raise ValueError('Observed channel identities must match the eight mapped channels')
        index = [names.index(c) for c in self.ring_channels]
        return FeatureBatch(np.asarray(batch.emg)[:,:,index].copy(),batch.sample_rate_hz,batch.imu,batch.posture)

    def apply_source(self,batch,*,observed_channel_ids):
        return self._apply(batch,observed_channel_ids)

    def apply_evaluation(self,batch,*,observed_channel_ids,model_layout_id):
        if model_layout_id != self.layout_id:
            raise ValueError('Physical channel layout differs from model source layout')
        return self._apply(batch,observed_channel_ids)
