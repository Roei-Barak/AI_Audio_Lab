"""
Desktop entry point (PyQt6).

Usage:
  python main_desktop.py
"""
import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from interfaces.desktop.main_window import KaraokeMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI_Audio_Lab")
    app.setApplicationDisplayName("🎤 Karaoke Studio")
    win = KaraokeMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
