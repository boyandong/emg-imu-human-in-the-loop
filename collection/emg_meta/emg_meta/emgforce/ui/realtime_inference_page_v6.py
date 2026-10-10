"""One guided registration for window and complete-action recognition."""
from pathlib import Path
import uuid

from PySide6.QtWidgets import QFileDialog,QGridLayout,QLabel
from emgforce.inference.personal_session_worker_v6 import PersonalSessionWorkerV6
from .realtime_inference_page_v5 import PersonalSessionPanelV5,RealtimeInferencePageV5


class PersonalSessionPanelV6(PersonalSessionPanelV5):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_choice.addItem('统一登记：七组窗口＋完整动作（8 通道 / 250 Hz）','seven_joint')
        self._legacy_window_controls=[self.apply_button,self.enroll_button,self.session_button,
            self.trial_button,self.save_button,self.cancel_button]
        for grid in self.findChildren(QGridLayout):
            item=grid.itemAtPosition(4,1)
            if item is not None and item.widget() is self.personal:
                self._legacy_window_controls += [grid.itemAtPosition(row,col).widget()
                    for row in (4,5) for col in range(3) if grid.itemAtPosition(row,col) is not None]
        self._joint_texts=[]
        for label in self.findChildren(QLabel):
            original=label.text()
            if original.startswith('需单独登记完整动作模板'):
                new='一次登记同时建立窗口校准和完整动作模板，保存为一个档案。按提示开始动作，完成后点击结束；每次采集 1 至 30 秒。自动模式在估计动作结束后判断。真实八通道识别效果仍待验证。'
            elif original.startswith('250 Hz · 200 ms'):
                new='250 Hz · 短窗口连续识别与完整动作判断共用个人／会话档案。换一次佩戴或录制后使用新的会话编号。校准成本按实际采集次数计算一次。'
            elif original=='完整动作个人档案':new='统一个人档案'
            elif original=='完整动作会话档案':new='统一会话档案'
            else:continue
            self._joint_texts.append((label,original,new))
        self._joint_buttons=[(self.temporal_apply_button,'应用完整动作档案','应用统一档案'),
            (self.temporal_enroll_button,'登记完整动作个人模板','登记统一个人档案'),
            (self.temporal_session_button,'校准本次完整动作模板','校准本次统一档案'),
            (self.temporal_save_button,'保存完整动作档案','保存统一档案')]
        self.joint_cost=QLabel('统一校准成本：尚未登记');self.joint_cost.setWordWrap(True)
        self.temporal_group.layout().addWidget(self.joint_cost)
        self._set_joint_presentation();self._update()

    def _is_joint(self):
        return hasattr(self,'model_choice') and self.model_choice.currentData()=='seven_joint'

    def _set_joint_presentation(self):
        if not hasattr(self,'_legacy_window_controls'):return
        joint=self._is_joint()
        for widget in self._legacy_window_controls:widget.setVisible(not joint)
        for widget,old,new in self._joint_texts:widget.setText(new if joint else old)
        for widget,old,new in self._joint_buttons:widget.setText(new if joint else old)
        self.temporal_group.setTitle('统一登记与完整动作识别（可选实验）' if joint else '完整动作识别（可选实验）')
        self.joint_cost.setVisible(joint)

    def _model_changed(self):
        super()._model_changed();self._set_joint_presentation()
        if hasattr(self,'joint_cost') and self.state is None:self.joint_cost.setText('统一校准成本：尚未登记')

    def load_backend(self):
        if not self._is_joint():return super().load_backend()
        channels=tuple(c.strip() for c in self.channels.text().split(','))
        if not self.user.text().strip() or not self.session.text().strip() or len(channels)!=8 or set(channels)!={f'CH{i+1}' for i in range(8)}:
            self.status.setText('请输入用户、会话编号以及包含 CH1 至 CH8 的通道顺序');return
        if not self.activate() or not self.shutdown():return
        self.worker=PersonalSessionWorkerV6(self.python.text().strip(),self.user.text().strip(),self.session.text().strip(),channels,self)
        self.worker.response.connect(self._response);self.worker.failed.connect(self._failed);self.worker.finished.connect(self._finished)
        self.status.setText('正在加载统一登记与识别后端…');self.worker.start();self._update()

    def apply_profiles(self):
        if self._is_joint():return self._temporal_apply()
        return super().apply_profiles()

    def _temporal_save(self):
        if not self._is_joint():return super()._temporal_save()
        path,_=QFileDialog.getSaveFileName(self,'保存统一档案',str(self.profiles_root/('joint-'+uuid.uuid4().hex[:12]+'.zip')),'统一档案 (*.zip)')
        if path:
            Path(path).parent.mkdir(parents=True,exist_ok=True);self._send('temporal_save',path=path)

    def _response(self,state):
        super()._response(state)
        if not self._is_joint():return
        t=state.get('temporal') or {};capture=t.get('capture')
        if capture:
            self.progress.setRange(0,capture['total']);self.progress.setValue(capture['completed'])
        entries=[]
        for title,key in (('长期','calibration_cost'),('本会话','session_calibration_cost')):
            cost=t.get(key)
            if cost:
                entries.append(f"{title} {cost['unique_native_calibration_trials']} 次 / {cost['native_signal_seconds']:.1f} 秒有效信号")
        self.joint_cost.setText('统一校准成本：'+('；'.join(entries)+'。同一批数据供两套识别使用，只计一次。' if entries else '尚未登记'))
        if not capture:
            self.temporal_guidance.setText('统一档案已应用，窗口识别与完整动作判断共用本次校准。选择判断方式后开始识别。'
                if t.get('personal_profile_id') else '请登记或加载统一个人档案；无需另行采集窗口校准。')

    def _update(self):
        super()._update()
        if self._is_joint() and hasattr(self,'_legacy_window_controls'):
            for widget in self._legacy_window_controls:widget.setEnabled(False)


class RealtimeInferencePageV6(RealtimeInferencePageV5):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);old=self.personal_session_panel;layout=old.parentWidget().layout()
        replacement=PersonalSessionPanelV6(self.models_root/'decision_profiles_v4',self._activate_personal_workflow,self)
        layout.replaceWidget(old,replacement);old.setParent(None);old.deleteLater();self.personal_session_panel=replacement
