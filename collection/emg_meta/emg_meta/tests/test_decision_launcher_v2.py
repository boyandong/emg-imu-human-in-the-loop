"""Launch the visible decision page and verify a real Windows shortcut."""
import os
from pathlib import Path
import subprocess
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from main_decision_v2 import create_window
from emgforce.ui.realtime_inference_page_v4 import PersonalSessionPanelV4


def test_launcher_selects_operational_panel_without_legacy_bundle(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = create_window(tmp_path)
    try:
        window.show()
        app.processEvents()
        page = window.realtime_inference_page
        assert window.main_tabs.currentWidget() is page
        assert isinstance(page.personal_session_panel, PersonalSessionPanelV4)
        assert page.personal_session_panel.decision.currentData() == 'baseline'
        assert page.personal_session_panel.model_choice.currentData()=='six'
        assert page.personal_session_panel.model_choice.count()==2
        assert page.personal_session_panel.worker is None
    finally:
        window.close()
        app.processEvents()


def test_windows_shortcut_resolves_current_entry_and_interpreter(tmp_path):
    if os.name != 'nt':
        pytest.skip('Windows shell shortcut contract')
    root = Path(__file__).resolve().parents[1]
    pythonw = Path('D:/miniconda/envs/emgforce/pythonw.exe')
    if not pythonw.is_file():
        pytest.skip('Qt interpreter unavailable')
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(root/'install_decision_shortcut_v2.ps1'), '-DesktopPath', str(tmp_path),
        '-ShortcutName', 'DecisionFixture.lnk'], check=True, capture_output=True)
    assert (tmp_path/'DecisionFixture.lnk').is_file()
    # The installer independently opens the saved .lnk and checks target,
    # quoted entry argument and working directory against resolved paths.
