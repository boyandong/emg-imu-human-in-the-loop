"""Guided Song250 personal/session enrollment and live inference controls."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import uuid
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout)
from emgforce.inference.personal_session_worker import PersonalSessionWorker


DISPLAY = {'neutral':'自然静息', 'fist':'握拳', 'index_pinch':'食指与拇指捏合', 'open_hand':'张开手掌'}


class PersonalSessionPanel(QFrame):
    def __init__(self, profiles_root, activate, parent=None):
        super().__init__(parent)
        self.profiles_root = Path(profiles_root)
        self.activate = activate
        self.worker = None
        self.connected = False
        self.state = None
        self._pending_op = None
        self.setObjectName('card')
        layout = QVBoxLayout(self)
        title = QLabel('Song 8 通道个人／会话校准（实验功能）')
        title.setStyleSheet('font-size:16px; font-weight:700;')
        layout.addWidget(title)
        description = QLabel('250 Hz · 200 ms 窗口 · 40 ms 更新。先按动作建立个人档案，换一次佩戴或录制后使用新的会话编号。'
            '校准只调整特征组融合权重；实机准确率、跨天效果和质量拒识尚未验证。')
        description.setWordWrap(True); layout.addWidget(description)
        grid = QGridLayout()
        default_python = os.environ.get('EMGIMU_CLASSIFIER_PYTHON')
        if not default_python:
            candidate = Path('D:/miniconda/python.exe')
            default_python = str(candidate) if candidate.is_file() else sys.executable
        self.python = QLineEdit(default_python)
        self.user = QLineEdit(); self.user.setPlaceholderText('输入当前使用者编号，如 Song；请勿共用他人档案')
        self.session = QLineEdit('recording-'+uuid.uuid4().hex[:12])
        self.channels = QLineEdit('CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8')
        self.channels.setToolTip('按设备实际列顺序填写通道身份；不能据此推断电极解剖位置')
        self.personal = QLineEdit(); self.personal.setPlaceholderText('个人档案 ZIP；留空使用源模型')
        self.session_profile = QLineEdit(); self.session_profile.setPlaceholderText('当前会话 ZIP；必须匹配上方个人档案及会话编号')
        for row, (name, widget) in enumerate((('模型 Python',self.python),('使用者',self.user),('录制会话',self.session),
                ('通道顺序',self.channels),('个人档案',self.personal),('会话档案',self.session_profile))):
            grid.addWidget(QLabel(name),row,0); grid.addWidget(widget,row,1)
        for row, widget, name in ((0,self.python,'Python 程序 (*.exe)'),(4,self.personal,'个人档案 (*.zip)'),
                                  (5,self.session_profile,'会话档案 (*.zip)')):
            button = QPushButton('浏览…')
            button.clicked.connect(lambda checked=False, w=widget, n=name:self._browse(w,n))
            grid.addWidget(button,row,2)
        layout.addLayout(grid)
        actions = QHBoxLayout()
        self.load_button = QPushButton('加载校准后端')
        self.apply_button = QPushButton('应用档案')
        self.stop_button = QPushButton('关闭后端')
        for button in (self.load_button,self.apply_button,self.stop_button):actions.addWidget(button)
        layout.addLayout(actions)
        calibration = QHBoxLayout()
        self.shots = QSpinBox(); self.shots.setRange(1,5); self.shots.setValue(1)
        calibration.addWidget(QLabel('每类试次')); calibration.addWidget(self.shots)
        self.enroll_button = QPushButton('建立个人档案')
        self.session_button = QPushButton('校准新会话')
        self.trial_button = QPushButton('准备好，采集本动作')
        self.save_button = QPushButton('保存校准档案')
        self.cancel_button = QPushButton('取消校准')
        for button in (self.enroll_button,self.session_button,self.trial_button,self.save_button,self.cancel_button):
            calibration.addWidget(button)
        layout.addLayout(calibration)
        self.guidance = QLabel('输入使用者编号并加载后端。通道顺序须与实际采集一致。')
        self.guidance.setWordWrap(True); layout.addWidget(self.guidance)
        self.progress = QProgressBar(); self.progress.setRange(0,4); layout.addWidget(self.progress)
        recognition = QHBoxLayout()
        self.recognize_button = QPushButton('开始个人／会话识别')
        self.pause_button = QPushButton('暂停识别')
        self.replay_button = QPushButton('回放已滤波窗口…')
        for button in (self.recognize_button,self.pause_button,self.replay_button):recognition.addWidget(button)
        layout.addLayout(recognition)
        self.prediction = QLabel('等待识别'); self.prediction.setStyleSheet('font-size:20px; font-weight:700; color:#2563eb;')
        layout.addWidget(self.prediction)
        self.status = QLabel('未加载'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.load_button.clicked.connect(self.load_backend)
        self.apply_button.clicked.connect(self.apply_profiles)
        self.stop_button.clicked.connect(self.shutdown)
        self.enroll_button.clicked.connect(lambda:self._begin('personal'))
        self.session_button.clicked.connect(lambda:self._begin('session'))
        self.trial_button.clicked.connect(lambda:self._send('trial'))
        self.save_button.clicked.connect(self.save_profile)
        self.cancel_button.clicked.connect(lambda:self._send('cancel'))
        self.recognize_button.clicked.connect(lambda:self._send('recognize'))
        self.pause_button.clicked.connect(lambda:self._send('pause'))
        self.replay_button.clicked.connect(self.replay)
        self._update()

    def _browse(self, widget, filter):
        path,_ = QFileDialog.getOpenFileName(self,'选择文件',widget.text(),filter)
        if path:widget.setText(path)

    def load_backend(self):
        if not self.user.text().strip() or not self.session.text().strip():
            self.status.setText('请输入使用者和录制会话编号'); return
        channels = tuple(c.strip() for c in self.channels.text().split(','))
        if len(channels)!=8 or len(set(channels))!=8 or set(channels)!={f'CH{i+1}' for i in range(8)}:
            self.status.setText('通道顺序须包含 CH1 至 CH8 各一次'); return
        if not self.activate() or not self.shutdown():return
        self.worker = PersonalSessionWorker(self.python.text().strip(),self.user.text().strip(),
                                           self.session.text().strip(),channels,self)
        self.worker.response.connect(self._response)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.status.setText('正在校验源模型并加载后端…')
        self.worker.start(); self._update()

    def _send(self, op, **values):
        if self.worker is None or not self.worker.isRunning():return
        if op in ('recognize','begin','pause','gap','cancel','profiles'):
            self.prediction.setText('等待识别')
        self._pending_op = op
        self.worker.send(op,**values)

    def apply_profiles(self):
        self._send('profiles',personal=self.personal.text().strip() or None,
                   session=self.session_profile.text().strip() or None)

    def _begin(self,kind):
        self._send('begin',kind=kind,shots=self.shots.value())

    def save_profile(self):
        if not self.state or not self.state.get('capture'):return
        kind=self.state['capture']['kind']
        default = self.profiles_root / (kind+'-'+uuid.uuid4().hex[:12]+'.zip')
        path,_ = QFileDialog.getSaveFileName(self,'保存校准档案',str(default),'校准档案 (*.zip)')
        if path:
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            self._save_path = path; self._save_kind = kind
            self._send('save',path=path)

    def replay(self):
        path,_ = QFileDialog.getOpenFileName(self,'选择与模型预处理一致的已滤波窗口','','窗口文件 (*.npz)')
        if path:self._send('replay',path=path)

    def _response(self,state):
        was_loaded = self.state is None
        previous = self.state
        self.state = state
        c = state.get('capture')
        if c:
            self.progress.setRange(0,c['total']); self.progress.setValue(c['completed'])
            if c['next_label'] is None:
                self.guidance.setText('全部动作采集完成，请保存档案；也可取消放弃。')
            elif c['collecting']:
                remaining=max(0,(c['trial_samples']-c['samples'])/250.)
                self.guidance.setText(f"保持：{DISPLAY[c['next_label']]}，剩余 {remaining:.1f} 秒。前 1 秒用于动作稳定，随后采集 1 秒。")
            else:
                self.guidance.setText(f"下一动作：{DISPLAY[c['next_label']]}。准备好后点击采集；完成后先放松，再开始下一试次。")
        else:
            self.guidance.setText('可建立个人档案、应用已保存档案或开始识别。新会话校准需要来自另一录制会话的个人档案。')
        if state.get('reset_reason'):
            self.prediction.setText('等待识别'); self.guidance.setText('采样不连续，本次校准已取消；请重新开始。')
        if 'probabilities' in state:
            p=state['probabilities'][-1];label=state['confirmed_labels'][-1]
            values=' · '.join(f'{DISPLAY[k]} {v:.0%}' for k,v in zip(state['classes'],p))
            self.prediction.setText((DISPLAY.get(label,'等待连续两次确认'))+'\n'+values)
        if 'prediction' in state:
            p=state['prediction']; self.prediction.setText(f"回放完成：{len(p['trial_ids'])} 个试次；未读取评估标签")
        if (previous and previous.get('capture') and not c and state.get('_operation')=='save'
                and not state.get('reset_reason')):
            target=self.personal if self._save_kind=='personal' else self.session_profile
            target.setText(state['saved_profile_path'])
            if self._save_kind=='personal':self.session_profile.clear()
        self.status.setText(f"用户 {state['user_id']} · 会话 {state['session_id']} · 长期校准 {state['personal_trials']} 试次"
            f" · 本会话 {state['session_trials']} 试次 · {state['mode']} · 源模型 {state['source_bank_id'][:12]}")
        if state.get('_operation') == self._pending_op:self._pending_op=None
        self._update()
        if was_loaded and (self.personal.text().strip() or self.session_profile.text().strip()):self.apply_profiles()

    def _failed(self,message):
        self.status.setText('操作失败：'+message); self._pending_op=None
        self.prediction.setText('等待识别'); self._update()

    def _finished(self):
        self.state=None; self.prediction.setText('等待识别'); self._update()

    def _update(self):
        running = self.worker is not None and self.worker.isRunning()
        ready = running and self.state is not None
        c = None if not ready else self.state.get('capture')
        for field in (self.python,self.user,self.session,self.channels):field.setEnabled(not running)
        self.load_button.setEnabled(not running); self.stop_button.setEnabled(running)
        self.apply_button.setEnabled(ready and c is None)
        self.enroll_button.setEnabled(ready and self.connected and c is None)
        self.session_button.setEnabled(ready and self.connected and c is None and bool(self.state.get('personal_profile_id')))
        self.trial_button.setEnabled(bool(c and self.connected and not c['collecting'] and c['next_label']))
        self.save_button.setEnabled(bool(c and not c['collecting'] and c['next_label'] is None))
        self.cancel_button.setEnabled(bool(c))
        self.recognize_button.setEnabled(ready and self.connected and c is None)
        self.pause_button.setEnabled(ready and self.state['mode']=='recognizing')
        self.replay_button.setEnabled(ready and c is None)

    def set_connected(self,connected):
        self.connected=bool(connected)
        if not connected:self._send('gap')
        self._update()

    def submit_emg(self,raw,indices):
        if self.connected and self.worker is not None:self.worker.submit_emg(raw,indices)

    def notify_packet_loss(self,lost):
        if lost>0:self._send('gap')

    def shutdown(self,timeout_ms=5000):
        if self.worker is not None:
            worker=self.worker; worker.stop()
            if not worker.wait(timeout_ms):
                self.status.setText('后台尚未停止，请稍后重试'); return False
            # Queued signals from the old process cannot overwrite a new worker state.
            for signal,slot in ((worker.response,self._response),(worker.failed,self._failed),(worker.finished,self._finished)):
                try:signal.disconnect(slot)
                except RuntimeError:pass
            self.worker=None
        self.state=None; self.prediction.setText('等待识别'); self._update()
        return True
