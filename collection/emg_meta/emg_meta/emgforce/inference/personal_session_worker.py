"""Persistent classifier process: keep sklearn out of the Qt environment."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import numpy as np
from PySide6.QtCore import QThread, Signal


def classifier_root():
    return Path(__file__).resolve().parents[5] / 'emgimu_classifier'


class PersonalSessionWorker(QThread):
    response = Signal(object)
    failed = Signal(str)

    def __init__(self, python, user, session, channels, parent=None):
        super().__init__(parent)
        self.python, self.user, self.session = str(python), user, session
        self.channels = tuple(channels)
        self.commands = queue.Queue(maxsize=64)
        self._stopping = threading.Event()
        self._accept_emg = False
        self.process = None
        self._lock = threading.Lock()

    def send(self, op, **values):
        try:
            self.commands.put_nowait(dict(op=op, **values))
        except queue.Full:
            self.failed.emit('识别处理积压，已停止；请重新加载，避免使用过期数据')
            self.stop()

    def submit_emg(self, raw, indices):
        if self._accept_emg and not self._stopping.is_set():
            self.send('emg', raw=np.asarray(raw).tolist(), indices=np.asarray(indices).tolist())

    def stop(self):
        self._stopping.set()
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()

    def _receive(self, process, operation='info'):
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('校准后端已退出；请检查所选 Python 是否安装项目依赖')
        payload = json.loads(line)
        if payload['ok']:
            result = payload['result']
            result['_operation'] = operation
            self._accept_emg = result['mode'] != 'idle'
            self.response.emit(result)
        else:
            # A recoverable contract error keeps the process and prior profile alive.
            self.failed.emit(payload['error'])

    def run(self):
        root = classifier_root()
        args = [self.python, '-u', '-m', 'emgimu.feature_bank.personal_session_stream_v1',
            '--package', str(root/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl'),
            '--acceptance', str(root/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'),
            '--user', self.user, '--session', self.session, '--channels', *self.channels]
        env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONIOENCODING='utf-8',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        try:
            with tempfile.TemporaryFile(mode='w+', encoding='utf8') as errors:
                with self._lock:
                    if self._stopping.is_set():
                        return
                    self.process = subprocess.Popen(args, cwd=root, env=env,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                        text=True, encoding='utf8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    process = self.process
                try:
                    self._receive(process)
                    while not self._stopping.is_set():
                        try:
                            message = self.commands.get(timeout=.05)
                        except queue.Empty:
                            if process.poll() is not None:
                                raise RuntimeError('校准后端意外退出')
                            continue
                        process.stdin.write(json.dumps(message, ensure_ascii=True)+'\n')
                        process.stdin.flush()
                        self._receive(process, message['op'])
                except Exception as exc:
                    if not self._stopping.is_set():
                        errors.seek(0)
                        self.failed.emit(str(exc)+'\n'+errors.read()[-2000:])
                finally:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    process.stdin.close(); process.stdout.close()
        except Exception as exc:
            if not self._stopping.is_set():
                self.failed.emit(str(exc))
