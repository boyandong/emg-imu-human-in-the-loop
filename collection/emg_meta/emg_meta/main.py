from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from emgforce.ui import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song-realtime", action="store_true",
                        help="Open realtime recognition and load the local Song F0+SPD model")
    args, qt_args = parser.parse_known_args(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("EMG Data Collection")
    app.setOrganizationName("EMGForce")
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    window = MainWindow(project_root=Path(__file__).resolve().parent,
                        song_realtime=args.song_realtime)
    window.show()
    if args.song_realtime:
        QTimer.singleShot(0, window.load_preferred_song_realtime)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
