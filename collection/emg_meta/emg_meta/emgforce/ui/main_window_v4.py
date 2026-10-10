"""Current application composition with opt-in full-action controls."""
from .main_window import MainWindow
from .realtime_inference_page_v5 import RealtimeInferencePageV5


class MainWindowV4(MainWindow):
    def __init__(self,project_root,*,song_realtime=False):
        self._decision_song_realtime=song_realtime
        super().__init__(project_root,song_realtime=song_realtime)

    def _wire(self):
        old=self.realtime_inference_page;index=self.main_tabs.indexOf(old);selected=self.main_tabs.currentWidget() is old
        replacement=RealtimeInferencePageV5(self.project_root/'models',prefer_song_spd=self._decision_song_realtime)
        self.main_tabs.removeTab(index);self.main_tabs.insertTab(index,replacement,'实时识别');self.realtime_inference_page=replacement
        if selected:self.main_tabs.setCurrentWidget(replacement)
        old.shutdown();old.deleteLater();super()._wire()
