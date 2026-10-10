"""Explicit six/seven-provider selection and contextual spectral readout."""
import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from emgforce.inference.personal_session_worker_v3 import PersonalSessionWorkerV3
from emgforce.inference.personal_session_worker_v4 import PersonalSessionWorkerV4
from .realtime_inference_page_v3 import PersonalSessionPanelV3, RealtimeInferencePageV3


class PersonalSessionPanelV4(PersonalSessionPanelV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        row = QHBoxLayout()
        row.addWidget(QLabel('识别模型'))
        self.model_choice = QComboBox()
        self.model_choice.addItem('现有六组特征模型', 'six')
        self.model_choice.addItem('实验新版：七组特征＋CSP（8 通道 / 250 Hz）', 'seven')
        row.addWidget(self.model_choice, 1)
        self.layout().insertLayout(2, row)
        self.model_choice.currentIndexChanged.connect(self._model_changed)
        self.context_note = QLabel('频谱变化：选择 CSP 新版并建立个人档案后显示。')
        self.context_note.setWordWrap(True)
        self.layout().insertWidget(3, self.context_note)

    def _model_changed(self):
        if not self.shutdown():
            return
        self.personal.clear()
        self.session_profile.clear()
        self.prediction.setText('等待识别')
        self.context_note.setText('频谱变化：等待新版个人档案和有效窗口；仅为相对参考变化，不表示疲劳程度。')
        self.status.setText('模型已切换，请加载后建立或选择与该版本匹配的档案。')
        seven = self.model_choice.currentData() == 'seven'
        self.decision_note.setText(
            'CSP 新版在已有单人同日记录上改善识别。F7＋F8 提高 F1，但置信度损失仍较大；按需试用。'
            if seven else 'F7＋F8 尚未通过现有六组模型的整体收益标准，按需试用。')

    def load_backend(self):
        if not self.user.text().strip() or not self.session.text().strip():
            self.status.setText('请输入使用者和录制会话编号')
            return
        channels = tuple(c.strip() for c in self.channels.text().split(','))
        if len(channels) != 8 or set(channels) != {f'CH{i+1}' for i in range(8)}:
            self.status.setText('通道顺序须包含 CH1 至 CH8 各一次')
            return
        if not self.activate() or not self.shutdown():
            return
        worker = PersonalSessionWorkerV4 if self.model_choice.currentData() == 'seven' else PersonalSessionWorkerV3
        self.worker = worker(self.python.text().strip(), self.user.text().strip(),
            self.session.text().strip(), channels, self)
        self.worker.response.connect(self._response)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.status.setText('正在校验并加载所选模型…')
        self.worker.start()
        self._update()

    def _response(self, state):
        super()._response(state)
        self.model_choice.setEnabled(state.get('capture') is None)
        if state.get('quality_rejected') and state['quality_rejected'][-1]:
            self.context_note.setText('频谱变化：当前信号质量不足，暂不解释。')
            return
        context = state.get('spectral_context')
        if state.get('prediction'):
            context = state['prediction'].get('spectral_context')
        if context is None:
            self.context_note.setText('频谱变化：等待新版个人档案和有效窗口；仅为相对参考变化，不表示疲劳程度。')
            return
        window = np.asarray(context['window_minus_long'], dtype=float)
        latest = window if window.ndim == 1 else window[-1]
        session = context.get('session_minus_long')
        text = f'相对个人参考的频谱变化：当前窗口平均绝对对数差 {np.abs(latest).mean():.3f}'
        if session is not None:
            text += f'；当次校准参考 {np.abs(np.asarray(session)).mean():.3f}'
        self.context_note.setText(text+'。仅为相对参考变化，不表示疲劳程度。')


class RealtimeInferencePageV4(RealtimeInferencePageV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.personal_session_panel
        layout = old.parentWidget().layout()
        replacement = PersonalSessionPanelV4(self.models_root/'decision_profiles_v2',
            self._activate_personal_workflow, self)
        layout.replaceWidget(old, replacement)
        old.setParent(None)
        old.deleteLater()
        self.personal_session_panel = replacement
