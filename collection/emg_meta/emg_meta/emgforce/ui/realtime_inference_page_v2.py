"""Versioned collection page with explicit raw-quality controls."""
from pathlib import Path
from PySide6.QtWidgets import QComboBox,QFileDialog,QHBoxLayout,QLabel
from emgforce.inference.personal_session_worker_v2 import PersonalSessionWorkerV2
from .personal_session_panel import PersonalSessionPanel
from .realtime_inference_page import RealtimeInferencePage


class PersonalSessionPanelV2(PersonalSessionPanel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        row=QHBoxLayout();row.addWidget(QLabel('信号质量处理'))
        self.quality=QComboBox()
        for title,key in (('关闭（保留原识别行为）','off'),('结构异常：掉零／平线／传输边界 → 未知','structural'),
                          ('实验性软权重：异常特征组降低权重','soft')):self.quality.addItem(title,key)
        row.addWidget(self.quality,1);self.layout().insertLayout(2,row)
        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.quality_note=QLabel('结构异常使用滤波前原始数据。软权重可能误拒正常动作；均未通过真实设备故障验证。')
        self.quality_note.setWordWrap(True);self.layout().insertWidget(3,self.quality_note)

    def load_backend(self):
        if not self.user.text().strip() or not self.session.text().strip():
            self.status.setText('请输入使用者和录制会话编号');return
        channels=tuple(c.strip() for c in self.channels.text().split(','))
        if len(channels)!=8 or len(set(channels))!=8 or set(channels)!={f'CH{i+1}' for i in range(8)}:
            self.status.setText('通道顺序须包含 CH1 至 CH8 各一次');return
        if not self.activate() or not self.shutdown():return
        self.worker=PersonalSessionWorkerV2(self.python.text().strip(),self.user.text().strip(),self.session.text().strip(),channels,self)
        self.worker.response.connect(self._response);self.worker.failed.connect(self._failed);self.worker.finished.connect(self._finished)
        self.status.setText('正在校验源模型和原始信号质量规则…');self.worker.start();self._update()

    def _quality_changed(self):
        if self.worker is not None and self.state is not None:self._send('quality',mode=self.quality.currentData())

    def _response(self,state):
        first=self.state is None
        super()._response(state)
        capture=state.get('capture');self.quality.setEnabled(capture is None)
        if state.get('calibration_rejected'):
            self.guidance.setText('本试次原始信号质量不足，未计入校准；检查电极接触后，重新采集当前动作。')
        if 'quality_rejected' in state and state['quality_rejected'][-1]:
            self.prediction.setText('信号质量不足 · 未知')
            bad=state['bad_channel_count'][-1]
            self.guidance.setText(f'异常通道数：{bad}。恢复后需连续两次有效预测才确认动作。')
        if state.get('prediction',{}).get('quality_decision'):
            decision=state['prediction']['quality_decision'];count=sum(decision['rejected'])
            self.prediction.setText(f"回放完成：{len(decision['trial_ids'])} 试次，{count} 个质量拒识；未读取评估标签")
        if first and self.quality.currentData()!='off':self._quality_changed()

    def replay(self):
        path,_=QFileDialog.getOpenFileName(self,'选择已滤波窗口','','窗口文件 (*.npz)')
        if not path:return
        if self.quality.currentData()=='off':self._send('replay',path=path);return
        raw,_=QFileDialog.getOpenFileName(self,'选择相同试次和窗口顺序的滤波前原始窗口','','原始窗口 (*.npz)')
        if raw:self._send('replay',path=path,raw_path=raw)


class RealtimeInferencePageV2(RealtimeInferencePage):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.personal_session_panel;layout=old.parentWidget().layout()
        replacement=PersonalSessionPanelV2(self.models_root/'personal_session_profiles',self._activate_personal_workflow,self)
        layout.replaceWidget(old,replacement);old.setParent(None);old.deleteLater()
        self.personal_session_panel=replacement
