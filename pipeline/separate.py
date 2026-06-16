import os
from pathlib import Path

from backend import BackendProcessor
from queue.gpu_semaphore import GpuSemaphore


def step_separate(
    audio_path: str,
    output_dir: str,
    task_log,
    mode: str = "2_stems",
    force: bool = False,
) -> dict:
    """
    Separate stems from audio using Kim_Vocal_2 (2-stem) or htdemucs_ft (4-stem).
    GPU-gated via GpuSemaphore.

    Returns dict: {vocals, playback, all_stems}
    Raises ValueError on failure.
    """
    proc = BackendProcessor(log_func=task_log)

    task_log("⏳ ממתין למשאבי GPU...")
    with GpuSemaphore():
        task_log(f"🚀 מתחיל הפרדה ({mode})...")
        stems = proc.separate(audio_path, output_dir, mode=mode)

    if not stems:
        raise ValueError("הפרדת stems נכשלה")

    if mode == "2_stems" and len(stems) >= 2:
        return {"vocals": stems[0], "playback": stems[1], "all_stems": stems}

    # 4-stem or other: try to identify vocals/playback
    vocals = None
    playback = None
    for p in stems:
        name = os.path.basename(p).lower()
        if "vocal" in name:
            vocals = p
        elif "instrumental" in name or "no_vocal" in name or "other" in name:
            playback = p

    if not vocals:
        vocals = stems[0]
    if not playback:
        playback = stems[1] if len(stems) > 1 else stems[0]

    return {"vocals": vocals, "playback": playback, "all_stems": stems}
