"""Current desktop entry exposes full-action controls and keeps six as initial choice."""
from pathlib import Path
from PySide6.QtWidgets import QApplication
import main_decision_v3
from emgforce.ui.main_window_v4 import MainWindowV4


def test_current_launcher_targets_actual_full_action_page(tmp_path):
    app=QApplication.instance() or QApplication([]);window=main_decision_v3.create_window(tmp_path)
    try:
        assert isinstance(window,MainWindowV4)
        page=window.realtime_inference_page;p=page.personal_session_panel
        assert window.main_tabs.currentWidget() is page
        assert p.model_choice.currentData()=='six' and p.model_choice.findData('seven_temporal')==2
        assert p.temporal_mode.currentData()=='off'
        installer=Path(main_decision_v3.__file__).with_name('install_decision_shortcut_v3.ps1').read_text(encoding='utf-8-sig')
        assert 'main_decision_v3.py' in installer
    finally:window.close();app.processEvents()
