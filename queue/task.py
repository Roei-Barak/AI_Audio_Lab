import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TaskStatus(str, Enum):
    PENDING   = "⏳ ממתין"
    RUNNING   = "🔄 עובד"
    DONE      = "✅ הושלם"
    ERROR     = "❌ שגיאה"
    CANCELLED = "🚫 בוטל"


@dataclass
class KaraokeTask:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    song_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    step: str = ""
    progress_pct: float = 0.0
    result_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    vram_used_gb: float = 0.0

    def log(self, msg: str) -> str:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        print(entry, flush=True)
        return entry

    def duration_s(self) -> float:
        return round(time.time() - self.created_at, 1)
