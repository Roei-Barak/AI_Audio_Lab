from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem

from logic import mm, task_manager


class ProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2000)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("📋 Mission Control"))

        self._status_label = QLabel("🟢 פנוי")
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["ID", "שיר", "סטטוס", "שלב", "VRAM", "זמן (s)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

    def _refresh(self):
        self._status_label.setText(mm.get_status())
        df = mm.get_df()
        if df is None or df.empty:
            self._table.setRowCount(0)
            return
        self._table.setRowCount(len(df))
        for r, (_, row) in enumerate(df.iterrows()):
            for c, val in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(str(val)))
