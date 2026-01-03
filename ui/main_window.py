"""Main application window and lightweight waveform placeholder."""
from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui

from core.engine import AudioEngine


class WaveformViewer(QtWidgets.QWidget):
    """Placeholder widget for a waveform/visualizer.

    Replace this with a performant canvas for real-time rendering later.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setStyleSheet("background-color: #0f0f10; border: 1px solid #2b2b2b;")
        label = QtWidgets.QLabel("Waveform View (placeholder)", self)
        label.setStyleSheet("color: #eaeaea; font-weight: 500;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)


class MainWindow(QtWidgets.QMainWindow):
    """Main window for AI_AUDIO_LAB with basic transport controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI_AUDIO_LAB")
        self.resize(900, 580)
        self._apply_dark_theme()
        self._build_ui()
        self.engine = AudioEngine.instance()

    def _apply_dark_theme(self) -> None:
        """Apply a simple dark palette for a modern look."""
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#121212"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#0f0f10"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#1a1a1a"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#2b2b2b"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#eaeaea"))
        self.setPalette(palette)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.waveform = WaveformViewer()
        layout.addWidget(self.waveform)

        transport = QtWidgets.QHBoxLayout()
        transport.addStretch()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addStretch()

        layout.addLayout(transport)

        self.setCentralWidget(central)

        # Connections
        self.play_btn.clicked.connect(self.on_play)
        self.stop_btn.clicked.connect(self.on_stop)

    def on_play(self) -> None:
        """Start the audio engine; handle errors via message box."""
        try:
            if not self.engine.is_running():
                self.engine.start()
                self.play_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Audio Error", str(exc))

    def on_stop(self) -> None:
        """Stop the audio engine safely."""
        try:
            if self.engine.is_running():
                self.engine.stop()
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Audio Error", str(exc))
