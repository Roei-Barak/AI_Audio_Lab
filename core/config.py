"""
core/config.py — Shared configuration constants for AI_Audio_Lab.

All modules import from here to guarantee consistency.
"""

import gc
import os

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
WORK_DIR = os.path.abspath("Karaoke_Output")
os.makedirs(WORK_DIR, exist_ok=True)

UVR_MODEL_DIR = os.path.join(WORK_DIR, "uvr_models")
os.makedirs(UVR_MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
try:
    import torch

    def _cudnn_path() -> None:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(torch_lib)

    _cudnn_path()
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
except ImportError:
    DEVICE = "cpu"
    TORCH_DTYPE = None

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "separation_2stem": "Kim_Vocal_2.onnx",
    "separation_4stem": "htdemucs_ft.yaml",
    "whisper_he": "ivrit-ai/whisper-large-v3-turbo",
    "whisper_en": "openai/whisper-large-v3",
}

# ---------------------------------------------------------------------------
# ASS subtitle default header
# ---------------------------------------------------------------------------
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# ASS subtitle style colour presets  {name: (primary_hex, outline_hex)}
SUBTITLE_PRESETS: dict[str, tuple[str, str]] = {
    "לבן קלאסי":   ("#FFFFFF", "#000000"),
    "זהב קריוקי":  ("#FFD700", "#000000"),
    "ניאון ירוק":  ("#39FF14", "#000000"),
    "כחול קרח":   ("#00BFFF", "#000000"),
    "אדום אש":    ("#FF4500", "#000000"),
}

# Subtitle vertical alignment options
SUBTITLE_POSITIONS: dict[str, int] = {
    "מרכז":  5,   # ASS alignment: center-middle
    "למטה":  2,   # bottom-center
    "למעלה": 8,   # top-center
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def cleanup_gpu() -> None:
    """Release GPU memory and run garbage collection."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    gc.collect()
