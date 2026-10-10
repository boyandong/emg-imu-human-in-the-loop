"""Opt-in Song250 lifecycle service used by the collection GUI.

Each live window is a distinct inference item. Calibration pools features by
guided trial, as in the frozen offline workflow. This changes the evaluation
unit relative to the native trial experiment; it makes no live efficacy claim.
"""
from collections import deque
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from .core import FeatureBatch
from .frozen_emg_bank_cli_v1 import load_windows
from .personal_session_cli_v1 import load_workflow, _json


PREPROCESSING = 'song250_causal_hp40_order4_notch50_100_Q30_zero_session_initial_v1'


class PersonalSessionStreamV1:
    def __init__(self, workflow, *, user_id, session_id, channel_ids):
        if not user_id.strip() or not session_id.strip():
            raise ValueError('Explicit user and recording-session identities required')
        if (workflow.bank.sensor_contract_ != (250., 50, 8)
                or workflow.preprocessing_id != PREPROCESSING):
            raise ValueError('This causal service requires the frozen Song250 preprocessing contract')
        self.workflow = workflow
        self.user = user_id
        self.session_id = session_id
        self.channels = tuple(channel_ids)
        self.kwargs = dict(user_id=user_id, session_id=session_id,
                           observed_channel_ids=self.channels, preprocessing_id=PREPROCESSING)
        workflow._input(FeatureBatch(np.zeros((1, 50, 8)), 250.),
                        observed_channel_ids=self.channels, preprocessing_id=PREPROCESSING)
        self.personal = self.session = None
        self.mode = 'idle'
        self.capture = None
        filters = [butter(4, 40., btype='highpass', fs=250., output='sos')]
        for hz in (50., 100.):
            b, a = iirnotch(hz, Q=30., fs=250.)
            filters.append(tf2sos(b, a))
        self.filters = filters
        self.reset()

    def reset(self):
        self.zi = [np.zeros((len(sos), 2, 8)) for sos in self.filters]
        self.window = deque(maxlen=50)
        self.last_index = None
        self.samples = 0
        self.epoch = uuid.uuid4().hex
        self.candidate = self.active = None
        self.confirmations = 0

    def info(self):
        c = self.capture
        return dict(schema='song_personal_session_stream_v1', mode=self.mode,
            user_id=self.user, session_id=self.session_id,
            source_bank_id=self.workflow.bank.bank_id_, contract_id=self.workflow.contract_id,
            classes=self.workflow.bank.classes_,
            personal_profile_id=None if self.personal is None else self.personal.profile_id,
            session_profile_id=None if self.session is None else self.session.profile_id,
            personal_trials=0 if self.personal is None else len(self.personal.calibration_trials),
            session_trials=0 if self.session is None else len(self.session.calibration_trials),
            capture=None if c is None else dict(kind=c['kind'], completed=len(c['labels']),
                total=len(c['order']), next_label=None if len(c['labels'])==len(c['order']) else c['order'][len(c['labels'])],
                collecting=c['trial'] is not None, samples=0 if c['trial'] is None else c['trial']['samples'],
                trial_samples=c['settle']+c['hold']),
            physical_validation_proven=False, quality_rejection_enabled=False,
            normalization_used_by_classifier=False, hop_samples=10, consecutive_frames=2)

    def _predict(self, batch, ids, offsets):
        if self.personal is not None:
            return self.workflow.predict(batch, ids, window_offsets=offsets,
                personal=self.personal, session=self.session, **self.kwargs)
        mapped = self.workflow._input(batch, observed_channel_ids=self.channels, preprocessing_id=PREPROCESSING)
        result = self.workflow.bank.predict(mapped, ids, window_offsets=offsets, user_id=self.user)
        result.update(personal_profile_id=None, session_profile_id=None)
        return result

    def command(self, message):
        op = message['op']
        if op == 'info':
            return self.info()
        if op == 'profiles':
            if self.capture is not None:
                raise ValueError('Cancel the current calibration before replacing profiles')
            personal = None if not message.get('personal') else self.workflow.load_profile(message['personal'], user_id=self.user)
            if message.get('session') and personal is None:
                raise ValueError('A session profile requires its personal profile')
            session = None if not message.get('session') else self.workflow.load_profile(message['session'],
                user_id=self.user, session_id=self.session_id, personal=personal)
            self.personal, self.session = personal, session
            self.mode = 'idle'; self.reset()
        elif op in ('recognize', 'pause', 'gap', 'cancel'):
            self.mode = 'recognizing' if op == 'recognize' else 'idle'
            self.capture = None
            self.reset()
        elif op == 'begin':
            if self.capture is not None:
                raise ValueError('Cancel or save the existing calibration first')
            kind = message['kind']; shots = message.get('shots', 1)
            settle = message.get('settle_samples', 250); hold = message.get('hold_samples', 250)
            if (kind not in ('personal', 'session') or type(shots) is not int or not 1 <= shots <= 5
                    or type(settle) is not int or not 0 <= settle <= 2500
                    or type(hold) is not int or not 50 <= hold <= 2500):
                raise ValueError('Invalid guided calibration duration/kind/shots')
            if kind == 'session' and (self.personal is None or self.personal.session_id == self.session_id):
                raise ValueError('New-session calibration requires a personal profile from another recording session')
            classes = list(self.workflow.bank.classes_)
            classes.remove(self.workflow.rest_label); classes.insert(0, self.workflow.rest_label)
            self.capture = dict(kind=kind, order=classes*shots, labels={}, windows=[], ids=[], offsets=[],
                                settle=settle, hold=hold, trial=None, started=time.monotonic())
            self.mode = 'calibrating'; self.reset()
        elif op == 'trial':
            c = self.capture
            if c is None or c['trial'] is not None or len(c['labels']) == len(c['order']):
                raise ValueError('No guided trial is ready to start')
            c['trial'] = dict(samples=0, values=[], id=f'gui:{self.session_id}:{uuid.uuid4().hex}')
        elif op == 'save':
            c = self.capture
            if c is None or c['trial'] is not None or len(c['labels']) != len(c['order']):
                raise ValueError('Complete every guided trial before saving')
            batch = FeatureBatch(np.stack(c['windows']), 250.)
            args = dict(window_offsets=c['offsets'], **self.kwargs)
            if c['kind'] == 'personal':
                profile = self.workflow.enroll_user(batch, c['ids'], c['labels'], **args)
            else:
                profile = self.workflow.calibrate_session(batch, c['ids'], c['labels'], personal=self.personal, **args)
            # Keep labeled calibration input separate from unlabeled prediction.
            # All companions are exclusive, and a failed save retains calibration.
            path = Path(message['path'])
            window_path = Path(str(path)+'.windows.npz')
            labels_path = Path(str(path)+'.labels.json')
            audit_path = Path(str(path)+'.capture.json')
            if any(p.exists() for p in (path, window_path, labels_path, audit_path)):
                raise FileExistsError('A profile or calibration companion already exists; choose a new filename')
            created = []
            try:
                with window_path.open('xb') as f:
                    created.append(window_path)
                    np.savez_compressed(f, emg=batch.emg, sample_rate_hz=np.array(250.),
                        trial_ids=np.asarray(c['ids']), window_offsets=np.asarray(c['offsets']))
                with labels_path.open('x', encoding='utf8') as f:
                    created.append(labels_path); json.dump(c['labels'], f, ensure_ascii=False)
                with audit_path.open('x', encoding='utf8') as f:
                    created.append(audit_path)
                    json.dump(dict(schema='guided_calibration_capture_v1', user_id=self.user, session_id=self.session_id,
                        contract_id=self.workflow.contract_id, channel_ids=self.channels, preprocessing_id=PREPROCESSING,
                        trial_count=len(c['labels']), window_count=batch.windows, window_hop_samples=10,
                        settle_samples_per_trial=c['settle'], held_samples_per_trial=c['hold'],
                        nominal_capture_seconds=len(c['labels'])*(c['settle']+c['hold'])/250.,
                        begin_to_save_seconds=time.monotonic()-c['started'],
                        physical_validation_proven=False, labels_are_guided_cues=True), f)
                self.workflow.save_profile(profile, path)
            except Exception:
                for item in created:
                    item.unlink()
                raise
            if c['kind'] == 'personal':
                self.personal, self.session = profile, None
            else:
                self.session = profile
            self.capture = None; self.mode = 'idle'; self.reset()
            return dict(**self.info(), saved_profile_path=str(path), calibration_windows_path=str(window_path),
                        calibration_labels_path=str(labels_path), calibration_audit_path=str(audit_path))
        elif op == 'replay':
            if self.capture is not None:
                raise ValueError('Cancel or save calibration before replay')
            # Filtered windows only, explicit channel/preprocessing contract.
            batch, ids, offsets = load_windows(message['path'])
            result = self._predict(batch, ids, offsets)
            return dict(**self.info(), prediction=result)
        elif op == 'emg':
            return self.ingest(message['raw'], message['indices'])
        else:
            raise ValueError('Unknown lifecycle operation')
        return self.info()

    def ingest(self, raw, indices):
        raw = np.asarray(raw, dtype=np.float64); indices = np.asarray(indices)
        if (raw.ndim != 2 or raw.shape[1] != 8 or indices.shape != (len(raw),)
                or indices.dtype.kind not in 'iu' or not np.isfinite(raw).all()):
            raise ValueError('Finite eight-channel samples and integer sample indices required')
        if not len(raw) or self.mode == 'idle':
            return self.info()
        # Never mix frames or a calibration trial across stream discontinuities.
        if np.any(np.diff(indices) != 1) or (self.last_index is not None and indices[0] != self.last_index+1):
            self.mode = 'idle'; self.capture = None; self.reset()
            return dict(**self.info(), reset_reason='Sampling discontinuity; calibration cancelled, restart required')
        x = raw
        for i, sos in enumerate(self.filters):
            x, self.zi[i] = sosfilt(sos, x, axis=0, zi=self.zi[i])
        x = x.astype(np.float32)
        self.last_index = int(indices[-1])
        ends, windows = [], []
        for value, index in zip(x, indices):
            self.samples += 1
            if self.mode == 'calibrating':
                c = self.capture; t = c['trial']
                if t is None:
                    continue
                t['samples'] += 1
                if t['samples'] > c['settle']:
                    t['values'].append(value.copy())
                if t['samples'] == c['settle'] + c['hold']:
                    label = c['order'][len(c['labels'])]
                    for start in range(0, c['hold']-49, 10):
                        c['windows'].append(np.stack(t['values'][start:start+50]))
                        c['ids'].append(t['id']); c['offsets'].append(start//10)
                    c['labels'][t['id']] = label
                    c['trial'] = None
                continue
            self.window.append(value)
            if self.samples >= 50 and (self.samples-50) % 10 == 0:
                ends.append(int(index)); windows.append(np.stack(self.window))
        if not windows:
            return self.info()
        ids = np.array([f'live:{self.session_id}:{self.epoch}:{end}' for end in ends])
        result = self._predict(FeatureBatch(np.stack(windows), 250.), ids, np.zeros(len(ids), dtype=int))
        # Provider bank sorts trial IDs; restore chronological sample order.
        lookup = {t: i for i, t in enumerate(result['trial_ids'])}
        q = np.stack([result['probabilities'][lookup[t]] for t in ids])
        labels = result['class_names']; confirmed = []
        for p in q:
            candidate = labels[int(p.argmax())]
            self.confirmations = self.confirmations+1 if candidate == self.candidate else 1
            self.candidate = candidate
            if self.confirmations >= 2:
                self.active = candidate
            confirmed.append(self.active)
        return dict(**self.info(), output_sample_indices=ends, probabilities=q,
                    confirmed_labels=confirmed, weights=result['weights'], provider_names=result['provider_names'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'acceptance', 'user', 'session'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--channels', nargs=8, required=True)
    args = parser.parse_args()
    try:
        service = PersonalSessionStreamV1(load_workflow(args.package, args.acceptance),
            user_id=args.user, session_id=args.session, channel_ids=args.channels)
        print(json.dumps(dict(ok=True, result=service.info()), default=_json), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get('op') == 'quit':
                    break
                response = dict(ok=True, result=service.command(request))
            except Exception as exc:
                response = dict(ok=False, error=str(exc))
            print(json.dumps(response, ensure_ascii=True, default=_json), flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=True), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
