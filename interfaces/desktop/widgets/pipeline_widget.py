from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QProgressBar, QTextEdit, QGroupBox,
)

from logic import run_karaoke_pipeline


class PipelineWorker(QThread):
    progress_signal = pyqtSignal(str, int)   # (step_name, pct)
    done_signal     = pyqtSignal(str, str)   # (video_path, logs)
    error_signal    = pyqtSignal(str)

    def __init__(self, source, lang, save_4, bidi, force, font_size, color):
        super().__init__()
        self.source    = source
        self.lang      = lang
        self.save_4    = save_4
        self.bidi      = bidi
        self.force     = force
        self.font_size = font_size
        self.color     = color

    def run(self):
        def on_progress(step, pct):
            self.progress_signal.emit(step, pct)

        video, logs = run_karaoke_pipeline(
            source=self.source,
            lang=self.lang,
            save_4_stems=self.save_4,
            use_bidi=self.bidi,
            force=self.force,
            font_size=self.font_size,
            color_hex=self.color,
            on_progress=on_progress,
        )
        self.done_signal.emit(video or "", logs)


class PipelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Input group
        input_group = QGroupBox("קלט")
        ig_layout = QVBoxLayout(input_group)

        self._url_edit = QLineEdit(placeholderText="YouTube URL או שם שיר")
        ig_layout.addWidget(QLabel("חיפוש / קישור:"))
        ig_layout.addWidget(self._url_edit)

        row = QHBoxLayout()
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(["he - עברית", "en - English"])
        row.addWidget(QLabel("שפה:"))
        row.addWidget(self._lang_combo)
        ig_layout.addLayout(row)

        self._cb_4stems = QCheckBox("שמור גם 4 ערוצים")
        self._cb_bidi   = QCheckBox("תיקון עברית BIDI")
        self._cb_force  = QCheckBox("עיבוד מחדש (Force)")
        ig_layout.addWidget(self._cb_4stems)
        ig_layout.addWidget(self._cb_bidi)
        ig_layout.addWidget(self._cb_force)

        layout.addWidget(input_group)

        # Controls
        self._start_btn = QPushButton("▶ התחל")
        self._start_btn.setStyleSheet("font-size: 14px; padding: 8px;")
        self._start_btn.clicked.connect(self._start)
        layout.addWidget(self._start_btn)

        # Progress
        self._step_label = QLabel("מוכן")
        layout.addWidget(self._step_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        # Log
        self._log_box = QTextEdit(readOnly=True)
        self._log_box.setMinimumHeight(200)
        layout.addWidget(QLabel("לוגים:"))
        layout.addWidget(self._log_box)

    def _start(self):
        source = self._url_edit.text().strip()
        if not source:
            self._log_box.append("⚠️ יש להזין URL או שם שיר")
            return

        lang = "he" if self._lang_combo.currentIndex() == 0 else "en"

        self._start_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._log_box.clear()

        self._worker = PipelineWorker(
            source=source,
            lang=lang,
            save_4=self._cb_4stems.isChecked(),
            bidi=self._cb_bidi.isChecked(),
            force=self._cb_force.isChecked(),
            font_size=80,
            color="#00FFFF",
        )
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.done_signal.connect(self._on_done)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: str, pct: int):
        self._step_label.setText(step)
        self._progress_bar.setValue(pct)

    def _on_done(self, video_path: str, logs: str):
        self._log_box.setPlainText(logs)
        if video_path:
            self._step_label.setText(f"✅ הושלם: {video_path}")
            self._progress_bar.setValue(100)
        else:
            self._step_label.setText("❌ נכשל - בדוק לוגים")
        self._start_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._log_box.append(f"❌ {msg}")
        self._step_label.setText("שגיאה")
        self._start_btn.setEnabled(True)
