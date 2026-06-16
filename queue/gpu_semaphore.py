import gc
import threading

from config import DEVICE, MAX_CONCURRENT_GPU_TASKS


class GpuSemaphore:
    """Singleton semaphore ensuring sequential GPU task execution."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sem = threading.Semaphore(MAX_CONCURRENT_GPU_TASKS)
        return cls._instance

    def __enter__(self):
        self._sem.acquire()
        self._flush()
        return self

    def __exit__(self, *_):
        self._flush()
        self._sem.release()

    def _flush(self):
        if DEVICE == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()

    def vram_free_gb(self) -> float:
        if DEVICE != "cuda":
            return 99.0
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()
            return round(free / 1024 ** 3, 1)
        except Exception:
            return 0.0
