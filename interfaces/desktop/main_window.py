from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QHBoxLayout

from interfaces.desktop.widgets.pipeline_widget import PipelineWidget
from interfaces.desktop.widgets.batch_widget import BatchWidget
from interfaces.desktop.widgets.subtitle_editor import SubtitleEditorWidget
from interfaces.desktop.widgets.progress_widget import ProgressWidget
from logic import mm


class KaraokeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 AI_Audio_Lab — Karaoke Studio")
        self.resize(1100, 750)
        self._build_ui()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(2000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: tabs
        tabs = QTabWidget()
        tabs.addTab(PipelineWidget(),       "⚡ שיר בודד")
        tabs.addTab(BatchWidget(),          "📚 רשימה")
        tabs.addTab(SubtitleEditorWidget(), "📝 עריכת כתוביות")
        layout.addWidget(tabs, stretch=3)

        # Right: mission control
        layout.addWidget(ProgressWidget(), stretch=1)

    def _refresh_status(self):
        self.statusBar().showMessage(mm.get_status())
