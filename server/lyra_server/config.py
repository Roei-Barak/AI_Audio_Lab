"""Server configuration, read from environment variables."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _ffmpeg() -> str:
    exe = os.environ.get("LYRA_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # same fallback the existing Gradio app uses
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        return "ffmpeg"


@dataclass(frozen=True)
class Config:
    data_dir: Path
    engine: str  # "ai" = real models (audio-separator + Whisper), "mock" = fast local stand-ins for dev/tests
    ffmpeg: str
    ffprobe: str
    max_file_bytes: int = 20 * 1024**3
    max_duration_s: float = 3 * 3600
    host_label: str = "המחשב הזה"
    auth: bool = True  # accounts required (LYRA_AUTH=off only for local development and tests)


def load() -> Config:
    data = Path(os.environ.get("LYRA_DATA_DIR", "Karaoke_Output/lyra")).resolve()
    data.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    probe = os.environ.get("LYRA_FFPROBE") or shutil.which("ffprobe") or str(Path(ff).with_name("ffprobe"))
    return Config(
        data_dir=data,
        engine=os.environ.get("LYRA_ENGINE", "ai"),
        ffmpeg=ff,
        ffprobe=probe,
        host_label=os.environ.get("LYRA_HOST_LABEL", "המחשב הזה"),
        auth=os.environ.get("LYRA_AUTH", "on").lower() not in ("off", "0", "false"),
    )
