import os

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

WORK_DIR = os.path.abspath("Karaoke_Output")
os.makedirs(WORK_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

MODELS = {
    "2_stems": "Kim_Vocal_2.onnx",
    "4_stems": "htdemucs_ft.yaml",
}
WHISPER_MODELS = {
    "he": "ivrit-ai/whisper-large-v3-turbo",
    "en": "openai/whisper-large-v3-turbo",
}

MAX_CONCURRENT_GPU_TASKS = 1
ASS_DEFAULT_FONT_SIZE = 80
ASS_DEFAULT_COLOR = "#00FFFF"
