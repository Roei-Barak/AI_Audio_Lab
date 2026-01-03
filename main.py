"""Entry point for AI_AUDIO_LAB application."""
from __future__ import annotations

import sys
from PyQt6 import QtWidgets

from ui.main_window import MainWindow
from core.engine import AudioEngine


def main() -> int:
    """Create the QApplication, show the main window and run the loop."""
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Touch the engine so initialization happens early (optional)
    _ = AudioEngine.instance()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
