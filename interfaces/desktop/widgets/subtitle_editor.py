import os
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QTextEdit,
)

from logic import backend
from config import WORK_DIR


class SubtitleEditorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ass_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # File row
        file_row = QHBoxLayout()
        self._path_label = QLabel("לא נטען קובץ")
        file_row.addWidget(self._path_label, stretch=1)
        load_btn = QPushButton("📂 טען ASS")
        load_btn.clicked.connect(self._load_ass)
        file_row.addWidget(load_btn)
        save_btn = QPushButton("💾 שמור")
        save_btn.clicked.connect(self._save_ass)
        file_row.addWidget(save_btn)
        layout.addLayout(file_row)

        # Table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Start", "End", "Text"])
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        # Log
        self._log = QTextEdit(readOnly=True, maximumHeight=80)
        layout.addWidget(self._log)

    def _load_ass(self):
        path, _ = QFileDialog.getOpenFileName(self, "בחר קובץ ASS", WORK_DIR, "ASS Files (*.ass)")
        if not path:
            return
        self._ass_path = path
        self._path_label.setText(os.path.basename(path))
        df = backend.ass_to_dataframe(path)
        self._table.setRowCount(0)
        for _, row in df.iterrows():
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setItem(r, 0, QTableWidgetItem(str(row["Start"])))
            self._table.setItem(r, 1, QTableWidgetItem(str(row["End"])))
            self._table.setItem(r, 2, QTableWidgetItem(str(row["Text"])))
        self._log.append(f"✅ נטענו {self._table.rowCount()} שורות")

    def _save_ass(self):
        if not self._ass_path:
            self._log.append("⚠️ אין קובץ פתוח")
            return
        import pandas as pd
        rows = []
        for r in range(self._table.rowCount()):
            rows.append({
                "Start": self._table.item(r, 0).text() if self._table.item(r, 0) else "",
                "End":   self._table.item(r, 1).text() if self._table.item(r, 1) else "",
                "Text":  self._table.item(r, 2).text() if self._table.item(r, 2) else "",
            })
        df = pd.DataFrame(rows)
        out = os.path.join(WORK_DIR, f"edited_{int(time.time())}.ass")
        backend.dataframe_to_ass(df, self._ass_path, out)
        self._ass_path = out
        self._log.append(f"✅ נשמר: {out}")
