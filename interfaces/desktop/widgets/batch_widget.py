from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QComboBox, QCheckBox, QPushButton, QProgressBar,
)

from logic import run_karaoke_pipeline


class BatchWorker(QThread):
    item_done_signal = pyqtSignal(int, int, str, str)  # (current, total, video, logs)
    all_done_signal  = pyqtSignal()

    def __init__(self, songs, lang, save_4, bidi, force):
        super().__init__()
        self.songs  = songs
        self.lang   = lang
        self.save_4 = save_4
        self.bidi   = bidi
        self.force  = force

    def run(self):
        total = len(self.songs)
        for i, song in enumerate(self.songs, 1):
            video, logs = run_karaoke_pipeline(
                source=song, lang=self.lang,
                save_4_stems=self.save_4, use_bidi=self.bidi, force=self.force,
            )
            self.item_done_signal.emit(i, total, video or "", logs)
        self.all_done_signal.emit()


class BatchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("רשימת שירים (כל שיר בשורה):"))

        self._text_edit = QTextEdit(placeholderText="https://youtube.com/...\nשם שיר 2\n...")
        self._text_edit.setMinimumHeight(150)
        layout.addWidget(self._text_edit)

        row = QHBoxLayout()
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["he - עברית", "en - English"])
        row.addWidget(QLabel("שפה:"))
        row.addWidget(self._lang_combo)

        self._cb_4stems = QCheckBox("4 ערוצים")
        self._cb_bidi   = QCheckBox("BIDI")
        self._cb_force  = QCheckBox("Force")
        row.addWidget(self._cb_4stems)
        row.addWidget(self._cb_bidi)
        row.addWidget(self._cb_force)
        layout.addLayout(row)

        self._start_btn = QPushButton("▶ התחל רשימה")
        self._start_btn.clicked.connect(self._start)
        layout.addWidget(self._start_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._log_box = QTextEdit(readOnly=True)
        layout.addWidget(self._log_box)

    def _start(self):
        text = self._text_edit.toPlainText()
        songs = [line.strip() for line in text.splitlines() if line.strip()]
        if not songs:
            self._log_box.append("⚠️ אין שירים ברשימה")
            return

        lang = "he" if self._lang_combo.currentIndex() == 0 else "en"
        self._start_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._log_box.clear()

        self._worker = BatchWorker(
            songs=songs, lang=lang,
            save_4=self._cb_4stems.isChecked(),
            bidi=self._cb_bidi.isChecked(),
            force=self._cb_force.isChecked(),
        )
        self._worker.item_done_signal.connect(self._on_item_done)
        self._worker.all_done_signal.connect(self._on_all_done)
        self._worker.start()

    def _on_item_done(self, current: int, total: int, video: str, logs: str):
        pct = int(current / total * 100)
        self._progress_bar.setValue(pct)
        status = "✅" if video else "❌"
        self._log_box.append(f"\n{status} שיר {current}/{total}: {video or 'נכשל'}")
        self._log_box.append(logs[-500:] if len(logs) > 500 else logs)

    def _on_all_done(self):
        self._start_btn.setEnabled(True)
        self._log_box.append("\n✅ הרשימה הושלמה!")
