from __future__ import annotations

import sys
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from emgforce.ui import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("EMG Data Collection")
    app.setOrganizationName("EMGForce")
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    window = MainWindow(project_root=Path(__file__).resolve().parent)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
