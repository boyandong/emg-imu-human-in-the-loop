"""Quality-aware service launcher; V1 launcher remains independently reproducible."""
import os
import json
import queue
import subprocess
import tempfile
from .personal_session_worker import PersonalSessionWorker,classifier_root


class PersonalSessionWorkerV2(PersonalSessionWorker):
    def run(self):
        root=classifier_root()
        args=[self.python,'-u','-m','emgimu.feature_bank.personal_session_stream_v2',
            '--package',str(root/'benchmarks/song_real8/song_personal_session_v1/source_bank.pkl'),
            '--acceptance',str(root/'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'),
            '--gate-package',str(root/'benchmarks/song_real8/song_raw_quality_v1/source_gate.pkl'),
            '--gate-results',str(root/'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'),
            '--user',self.user,'--session',self.session,'--channels',*self.channels]
        env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        try:
            with tempfile.TemporaryFile(mode='w+',encoding='utf8') as errors:
                with self._lock:
                    if self._stopping.is_set():return
                    self.process=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                        stderr=errors,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    process=self.process
                try:
                    self._receive(process)
                    while not self._stopping.is_set():
                        try:message=self.commands.get(timeout=.05)
                        except queue.Empty:
                            if process.poll() is not None:raise RuntimeError('质量识别后端意外退出')
                            continue
                        process.stdin.write(json.dumps(message,ensure_ascii=True)+'\n');process.stdin.flush()
                        self._receive(process,message['op'])
                except Exception as exc:
                    if not self._stopping.is_set():
                        errors.seek(0);self.failed.emit(str(exc)+'\n'+errors.read()[-2000:])
                finally:
                    if process.poll() is None:process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    process.stdin.close();process.stdout.close()
        except Exception as exc:
            if not self._stopping.is_set():self.failed.emit(str(exc))
