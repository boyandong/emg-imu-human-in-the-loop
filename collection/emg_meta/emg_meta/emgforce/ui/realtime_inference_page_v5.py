"""Full-action calibration and manual/estimated decisions beside window output."""
from pathlib import Path
import uuid
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox,QFileDialog,QFormLayout,QGroupBox,QHBoxLayout,QLabel,QLineEdit,QPushButton,QVBoxLayout
from emgforce.inference.personal_session_worker_v5 import PersonalSessionWorkerV5
from .personal_session_panel import DISPLAY
from .realtime_inference_page_v4 import PersonalSessionPanelV4,RealtimeInferencePageV4


class PersonalSessionPanelV5(PersonalSessionPanelV4):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._selected_model=self.model_choice.currentIndex()
        self.model_choice.addItem('完整动作实验：七组窗口＋DTW／路径模板（8 通道 / 250 Hz）','seven_temporal')
        group=QGroupBox('完整动作识别（可选实验）');layout=QVBoxLayout(group)
        note=QLabel('需单独登记完整动作模板。点击开始后完成整个动作，再点击结束，保留全部信号。'
            '自动模式用登记时的静息信号估计边界，只在动作结束后判断。公开数据尚未通过整体收益标准，真实八通道效果未验证。')
        note.setWordWrap(True);layout.addWidget(note)
        self.temporal_personal=QLineEdit();self.temporal_session=QLineEdit()
        form=QFormLayout()
        for title,field in (('完整动作个人档案',self.temporal_personal),('完整动作会话档案',self.temporal_session)):
            row=QHBoxLayout();row.addWidget(field,1);button=QPushButton('浏览…')
            button.clicked.connect(lambda checked=False,field=field:self._browse(field,'完整动作档案 (*.zip)'))
            row.addWidget(button);form.addRow(title,row)
        layout.addLayout(form)
        row=QHBoxLayout()
        self.temporal_apply_button=QPushButton('应用完整动作档案')
        self.temporal_apply_button.clicked.connect(self._temporal_apply);row.addWidget(self.temporal_apply_button)
        self.temporal_enroll_button=QPushButton('登记完整动作个人模板')
        self.temporal_enroll_button.clicked.connect(lambda:self._send('temporal_begin',kind='personal',shots=self.shots.value()))
        row.addWidget(self.temporal_enroll_button)
        self.temporal_session_button=QPushButton('校准本次完整动作模板')
        self.temporal_session_button.clicked.connect(lambda:self._send('temporal_begin',kind='session',shots=self.shots.value()))
        row.addWidget(self.temporal_session_button);layout.addLayout(row)
        row=QHBoxLayout()
        self.temporal_trial_button=QPushButton('开始采集当前完整动作')
        self.temporal_trial_button.clicked.connect(lambda:self._send('temporal_trial_start'));row.addWidget(self.temporal_trial_button)
        self.temporal_end_button=QPushButton('结束当前完整动作')
        self.temporal_end_button.clicked.connect(lambda:self._send('temporal_trial_end'));row.addWidget(self.temporal_end_button)
        self.temporal_save_button=QPushButton('保存完整动作档案');self.temporal_save_button.clicked.connect(self._temporal_save)
        row.addWidget(self.temporal_save_button)
        self.temporal_cancel_button=QPushButton('取消完整动作采集');self.temporal_cancel_button.clicked.connect(lambda:self._send('cancel'))
        row.addWidget(self.temporal_cancel_button);layout.addLayout(row)
        self.temporal_guidance=QLabel('选择完整动作实验模型并加载后，可登记模板。');self.temporal_guidance.setWordWrap(True)
        layout.addWidget(self.temporal_guidance)
        row=QHBoxLayout();row.addWidget(QLabel('完整动作判断方式'))
        self.temporal_mode=QComboBox()
        for title,key in (('关闭完整动作分支','off'),('手动标记动作起止','manual'),('自动估计动作边界（实验）','auto')):
            self.temporal_mode.addItem(title,key)
        self.temporal_mode.currentIndexChanged.connect(self._temporal_mode_changed);row.addWidget(self.temporal_mode,1)
        self.temporal_action_start=QPushButton('标记动作开始');self.temporal_action_start.clicked.connect(lambda:self._send('temporal_action_start'))
        row.addWidget(self.temporal_action_start)
        self.temporal_action_end=QPushButton('标记结束并判断');self.temporal_action_end.clicked.connect(lambda:self._send('temporal_action_end'))
        row.addWidget(self.temporal_action_end);layout.addLayout(row)
        self.temporal_prediction=QLabel('完整动作结果：等待动作结束');self.temporal_prediction.setWordWrap(True)
        layout.addWidget(self.temporal_prediction)
        self.layout().insertWidget(self.layout().count()-1,group)
        self.temporal_group=group;self._update()

    def _model_changed(self):
        previous=getattr(self,'_selected_model',0)
        if not self.shutdown():
            with QSignalBlocker(self.model_choice):self.model_choice.setCurrentIndex(previous)
            return
        self._selected_model=self.model_choice.currentIndex()
        self.personal.clear();self.session_profile.clear()
        if hasattr(self,'temporal_personal'):
            self.temporal_personal.clear();self.temporal_session.clear()
            with QSignalBlocker(self.temporal_mode):self.temporal_mode.setCurrentIndex(0)
        self.context_note.setText('频谱变化：等待个人档案和有效窗口；仅表示相对参考变化。')
        self.status.setText('模型已切换，请重新加载并使用对应档案。');self._update()

    def load_backend(self):
        if self.model_choice.currentData()!='seven_temporal':return super().load_backend()
        channels=tuple(c.strip() for c in self.channels.text().split(','))
        if not self.user.text().strip() or not self.session.text().strip() or len(channels)!=8 or set(channels)!={f'CH{i+1}' for i in range(8)}:
            self.status.setText('请输入用户、会话编号以及包含 CH1 至 CH8 的通道顺序');return
        if not self.activate() or not self.shutdown():return
        self.worker=PersonalSessionWorkerV5(self.python.text().strip(),self.user.text().strip(),self.session.text().strip(),channels,self)
        self.worker.response.connect(self._response);self.worker.failed.connect(self._failed);self.worker.finished.connect(self._finished)
        self.status.setText('正在加载窗口模型与完整动作接口…');self.worker.start();self._update()

    def _temporal_apply(self):
        self._send('temporal_profiles',personal=self.temporal_personal.text().strip() or None,
            session=self.temporal_session.text().strip() or None)

    def _temporal_save(self):
        path,_=QFileDialog.getSaveFileName(self,'保存完整动作档案',str(self.profiles_root/('complete-'+uuid.uuid4().hex[:12]+'.zip')),'完整动作档案 (*.zip)')
        if path:
            Path(path).parent.mkdir(parents=True,exist_ok=True);self._send('temporal_save',path=path)

    def _temporal_mode_changed(self):
        if self.state and self.state.get('temporal') is not None:self._send('temporal_mode',mode=self.temporal_mode.currentData())

    def _send(self,op,**values):
        if hasattr(self,'temporal_prediction') and op in ('pause','gap','cancel','begin','profiles','quality','decision','temporal_begin','temporal_profiles','temporal_mode','recognize'):
            self.temporal_prediction.setText('完整动作结果：等待动作结束')
        super()._send(op,**values)

    def _response(self,state):
        first=self.state is None;super()._response(state)
        t=state.get('temporal')
        if t is None:return
        with QSignalBlocker(self.temporal_mode):self.temporal_mode.setCurrentIndex(self.temporal_mode.findData(t['mode']))
        c=t['capture']
        if c:
            label=DISPLAY.get(c['next_label'],c['next_label'])
            text=(f"正在采集：{label}，已收到 {c['samples']/250:.1f} 秒；完成整个动作后点击结束。" if c['collecting'] else
                  f"下一完整动作：{label}；已完成 {c['completed']}/{c['total']}。" if c['next_label'] else '全部完整动作已采集，请保存档案。')
            self.temporal_guidance.setText(text)
        else:self.temporal_guidance.setText('完整动作模板已加载。选择手动或自动方式，再点击开始实时识别。' if t['personal_profile_id'] else '请先登记或加载独立的完整动作个人档案。')
        if state.get('calibration_rejected'):self.temporal_guidance.setText('当前完整动作信号质量不足，未计入校准；检查接触后重新采集。')
        if state.get('reset_reason'):self.temporal_prediction.setText('完整动作结果：采样中断，已丢弃未结束动作')
        if state.get('saved_temporal_profile_path'):
            field=self.temporal_personal if state['temporal_saved_kind']=='personal' else self.temporal_session
            field.setText(state['saved_temporal_profile_path'])
            if state['temporal_saved_kind']=='personal':self.temporal_session.clear()
        if state.get('temporal_events'):
            e=state['temporal_events'][-1];boundary='自动估计边界' if e['boundary_kind']=='estimated' else '手动标记边界'
            label='信号质量不足 · 未知' if e['rejected'] else DISPLAY.get(e['label'],e['label'])
            probabilities=' · '.join(f'{DISPLAY.get(k,k)} {v:.0%}' for k,v in zip(e['classes'],e['probabilities']))
            self.temporal_prediction.setText(f"完整动作结果：{label} · {boundary} · {(e['end']-e['start'])/250:.2f} 秒\n{probabilities}\n动作结束后输出；不代表实时短窗口准确率。")
        self._update()
        if first and (self.temporal_personal.text().strip() or self.temporal_session.text().strip()):self._temporal_apply()

    def _failed(self,message):
        super()._failed(message)
        if hasattr(self,'temporal_prediction'):self.temporal_prediction.setText('完整动作结果：操作失败，请查看状态提示')

    def _finished(self):
        super()._finished()
        if hasattr(self,'temporal_prediction'):self.temporal_prediction.setText('完整动作结果：等待动作结束')

    def shutdown(self,timeout_ms=5000):
        result=super().shutdown(timeout_ms)
        if result and hasattr(self,'temporal_prediction'):self.temporal_prediction.setText('完整动作结果：等待动作结束')
        return result

    def _update(self):
        super()._update()
        if not hasattr(self,'temporal_group'):return
        t=None if not self.state else self.state.get('temporal');ready=bool(t and self.worker and self.worker.isRunning())
        c=None if not ready else t['capture'];window_capture=bool(self.state and self.state.get('capture'))
        available=ready and not window_capture
        self.temporal_group.setEnabled(available)
        for field in (self.temporal_personal,self.temporal_session):field.setEnabled(not c)
        self.temporal_apply_button.setEnabled(available and not c)
        self.temporal_enroll_button.setEnabled(available and self.connected and not c)
        self.temporal_session_button.setEnabled(bool(available and self.connected and not c and t['personal_profile_id']))
        self.temporal_trial_button.setEnabled(bool(c and self.connected and not c['collecting'] and c['next_label']))
        self.temporal_end_button.setEnabled(bool(c and self.connected and c['collecting'] and c['samples']>=250))
        self.temporal_save_button.setEnabled(bool(c and not c['collecting'] and c['next_label'] is None))
        self.temporal_cancel_button.setEnabled(bool(c))
        self.temporal_mode.setEnabled(bool(available and not c and t['personal_profile_id']))
        manual=bool(available and self.connected and not c and self.state['mode']=='recognizing' and t['mode']=='manual')
        self.temporal_action_start.setEnabled(manual and not t['action_open'])
        self.temporal_action_end.setEnabled(manual and t['action_open'])
        if c:
            for widget in (self.model_choice,self.decision,self.quality,self.apply_button,self.enroll_button,self.session_button,self.recognize_button,self.replay_button):widget.setEnabled(False)


class RealtimeInferencePageV5(RealtimeInferencePageV4):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);old=self.personal_session_panel;layout=old.parentWidget().layout()
        replacement=PersonalSessionPanelV5(self.models_root/'decision_profiles_v3',self._activate_personal_workflow,self)
        layout.replaceWidget(old,replacement);old.setParent(None);old.deleteLater();self.personal_session_panel=replacement
