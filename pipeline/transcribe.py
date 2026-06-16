from backend import BackendProcessor
from config import HF_TOKEN
from queue.gpu_semaphore import GpuSemaphore


def step_transcribe(
    vocals_path: str,
    output_dir: str,
    task_log,
    lang: str = "he",
    hf_token: str = None,
) -> dict:
    """
    Transcribe vocals using Whisper (ivrit-ai for Hebrew, openai for English).
    GPU-gated via GpuSemaphore.

    Returns dict: {ass_path}
    Raises ValueError on failure.
    """
    proc = BackendProcessor(log_func=task_log)
    token = hf_token or HF_TOKEN

    task_log("⏳ ממתין למשאבי GPU לתמלול...")
    with GpuSemaphore():
        task_log(f"📝 מתמלל ({lang})...")
        ass_path = proc.transcribe(vocals_path, output_dir, lang, hf_token=token)

    if not ass_path:
        raise ValueError("תמלול נכשל")

    task_log(f"✅ נוצר קובץ ASS: {ass_path}")
    return {"ass_path": ass_path}
