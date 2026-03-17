"""
check_install.py — בדיקת מהירה שהכל מותקן כראוי.

הרץ: python check_install.py
"""

import sys
import importlib

REQUIRED = [
    ("yt_dlp",          "yt-dlp",           True),
    ("imageio_ffmpeg",  "imageio-ffmpeg",    True),
    ("gradio",          "gradio",            True),
    ("pandas",          "pandas",            True),
    ("numpy",           "numpy",             True),
    ("soundfile",       "soundfile",         True),
    ("arabic_reshaper", "arabic-reshaper",   True),
    ("bidi",            "python-bidi",       True),
    ("fastapi",         "fastapi",           False),
    ("uvicorn",         "uvicorn",           False),
    ("torch",           "torch",             False),
    ("transformers",    "transformers",      False),
    ("librosa",         "librosa",           False),
    ("audio_separator", "audio-separator",   False),
]

print("\n🔍 בודק התקנות...\n")
ok_count = 0
warn_count = 0
err_count = 0

for module, pkg, required in REQUIRED:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        print(f"  ✅ {pkg:<22} {ver}")
        ok_count += 1
    except ImportError:
        if required:
            print(f"  ❌ {pkg:<22} חסר! pip install {pkg}")
            err_count += 1
        else:
            print(f"  ⚠️  {pkg:<22} לא מותקן (אופציונלי)")
            warn_count += 1

# FFmpeg check
print()
try:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"  ✅ FFmpeg                {ffmpeg}")
    ok_count += 1
except Exception as e:
    print(f"  ❌ FFmpeg                {e}")
    err_count += 1

# Core modules check
print()
print("🔍 בודק מודולים מקומיים...\n")
local_modules = [
    "core.config",
    "core.backend",
    "modules.downloader",
    "modules.separator",
    "modules.transcriber",
    "modules.renderer",
    "cli.main",
]
for mod in local_modules:
    try:
        importlib.import_module(mod)
        print(f"  ✅ {mod}")
        ok_count += 1
    except Exception as e:
        print(f"  ❌ {mod}: {e}")
        err_count += 1

# PyTorch/CUDA info
print()
try:
    import torch
    device = "CUDA ✅" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"  💻 Device: {device}")
except ImportError:
    print("  ℹ️  PyTorch לא מותקן — עובד ב-CPU mode")

# Summary
print()
print("─" * 50)
print(f"  ✅ תקין:    {ok_count}")
print(f"  ⚠️  אופציונלי: {warn_count}")
print(f"  ❌ חסר:     {err_count}")
print()

if err_count == 0:
    print("🎉 הכל מוכן! הרץ: run_gui.bat")
else:
    print("⚠️  תקן את השגיאות למעלה לפני הרצה.")
    sys.exit(1)
