"""Launch the current collection application at its personal/session panel."""
import logging
from pathlib import Path
import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QScrollArea
from emgforce.ui.main_window_v4 import MainWindowV4 as MainWindow


def create_window(project_root=None):
    window = MainWindow(Path(project_root) if project_root is not None else Path(__file__).resolve().parent)
    page = window.realtime_inference_page
    window.main_tabs.setCurrentWidget(page)
    for area in page.findChildren(QScrollArea):
        QTimer.singleShot(0, lambda area=area: area.ensureWidgetVisible(page.personal_session_panel, 0, 16))
    return window


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    app = QApplication(sys.argv)
    app.setApplicationName('EMG Data Collection')
    app.setOrganizationName('EMGForce')
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app.setFont(QFont('Microsoft YaHei UI', 10))
    window = create_window()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
