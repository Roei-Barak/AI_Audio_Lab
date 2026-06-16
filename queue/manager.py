import threading
import time
from typing import Dict, Optional

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from config import DEVICE
from queue.task import KaraokeTask, TaskStatus


class TaskManager:
    """Thread-safe registry of all submitted tasks."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tasks: Dict[str, KaraokeTask] = {}

    def submit(self, task: KaraokeTask) -> str:
        with self.lock:
            self.tasks[task.task_id] = task
        return task.task_id

    def get(self, task_id: str) -> Optional[KaraokeTask]:
        return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
        return False

    def active_count(self) -> int:
        with self.lock:
            return sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)

    def get_status_summary(self) -> str:
        active = self.active_count()
        vram_info = ""
        if DEVICE == "cuda":
            try:
                import torch
                free, total = torch.cuda.mem_get_info()
                vram_info = f" (VRAM פנוי: {free/1024**3:.1f}GB)"
            except Exception:
                pass

        if active == 0:
            return f"🟢 פנוי{vram_info}"
        return f"🔄 עובד ({active} משימות){vram_info}"

    def to_dataframe(self):
        if not _PANDAS:
            return []
        with self.lock:
            rows = [
                {
                    "ID": t.task_id,
                    "שיר": t.song_name[:35],
                    "סטטוס": t.status.value if hasattr(t.status, "value") else str(t.status),
                    "שלב": t.step,
                    "VRAM": f"{t.vram_used_gb:.1f}GB" if t.vram_used_gb else "-",
                    "זמן (s)": t.duration_s(),
                }
                for t in list(self.tasks.values())
            ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["ID", "שיר", "סטטוס", "שלב", "VRAM", "זמן (s)"]
        )
