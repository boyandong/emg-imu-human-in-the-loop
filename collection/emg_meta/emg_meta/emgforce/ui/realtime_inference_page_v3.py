"""Explicit operational F7/F8 choices; no experimental default promotion."""
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from emgforce.inference.personal_session_worker_v3 import PersonalSessionWorkerV3
from .realtime_inference_page_v2 import PersonalSessionPanelV2
from .realtime_inference_page import RealtimeInferencePage


BRANCHES = {
    'baseline': (False, False),
    'F8_only': (False, True),
    'F7_F8': (True, True),
}


class PersonalSessionPanelV3(PersonalSessionPanelV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for label in self.findChildren(QLabel):
            if label.text().startswith('250 Hz · 200 ms'):
                label.setText('250 Hz · 200 ms 窗口 · 40 ms 更新。先建立个人档案；更换佩戴或录制后，用新的会话编号校准。'
                    '可选择现有方案、会话权重调整或实验性个人动作原型。实机准确率和跨天效果尚未验证。')
        self.personal.setPlaceholderText('新版个人档案 ZIP；与旧版档案格式分开')
        self.session_profile.setPlaceholderText('新版会话 ZIP；需匹配个人档案与当前会话编号')
        row = QHBoxLayout()
        row.addWidget(QLabel('个人／会话决策'))
        self.decision = QComboBox()
        for label, key in (('现有校准方案', 'baseline'),
                ('仅 F8：按会话变化调整特征权重', 'F8_only'),
                ('实验性 F7＋F8：个人动作原型＋会话权重', 'F7_F8')):
            self.decision.addItem(label, key)
        row.addWidget(self.decision, 1)
        self.layout().insertLayout(2, row)
        self.decision.currentIndexChanged.connect(self._decision_changed)
        self.decision_note = QLabel('先建立或加载新版个人档案；F8 还需要当前会话校准。F7＋F8 尚未通过整体收益标准，按需试用。')
        self.decision_note.setWordWrap(True)
        self.layout().insertWidget(3, self.decision_note)

    def load_backend(self):
        if not self.user.text().strip() or not self.session.text().strip():
            self.status.setText('请输入使用者和录制会话编号')
            return
        channels = tuple(c.strip() for c in self.channels.text().split(','))
        if len(channels) != 8 or len(set(channels)) != 8 or set(channels) != {f'CH{i+1}' for i in range(8)}:
            self.status.setText('通道顺序须包含 CH1 至 CH8 各一次')
            return
        if not self.activate() or not self.shutdown():
            return
        self.worker = PersonalSessionWorkerV3(self.python.text().strip(), self.user.text().strip(),
            self.session.text().strip(), channels, self)
        self.worker.response.connect(self._response)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.status.setText('正在校验源模型并加载新版个人／会话决策…')
        self.worker.start()
        self._update()

    def _decision_changed(self):
        if self.worker is not None and self.state is not None:
            anchor, routing = BRANCHES[self.decision.currentData()]
            self._send('decision', use_anchor=anchor, use_session_routing=routing, anchor_mode='blended')

    def _send(self, op, **values):
        if op in ('decision', 'quality'):
            self.prediction.setText('等待识别')
        super()._send(op, **values)

    def _response(self, state):
        first = self.state is None
        super()._response(state)
        self.decision.setEnabled(state.get('capture') is None)
        anchor = '已启用' if state['anchor_enabled'] else '未启用'
        routing = '已启用' if state['session_routing_enabled'] else '未启用'
        self.status.setText(self.status.text()+f' · F7 {anchor} · F8 {routing}')
        if first and self.decision.currentData() != 'baseline':
            self._decision_changed()


class RealtimeInferencePageV3(RealtimeInferencePage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.personal_session_panel
        layout = old.parentWidget().layout()
        replacement = PersonalSessionPanelV3(self.models_root/'decision_profiles_v1',
            self._activate_personal_workflow, self)
        layout.replaceWidget(old, replacement)
        old.setParent(None)
        old.deleteLater()
        self.personal_session_panel = replacement
