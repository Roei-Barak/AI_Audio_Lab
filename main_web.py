"""
Web UI entry point (Gradio).

Usage:
  python main_web.py
  python main_web.py --port 7860 --share
"""
import argparse
import sys

import gradio as gr
import app as karaoke_app
from config import WORK_DIR


def main():
    p = argparse.ArgumentParser(description="AI_Audio_Lab — Web UI")
    p.add_argument("--port",  type=int, default=7860, help="פורט (ברירת מחדל: 7860)")
    p.add_argument("--share", action="store_true",    help="Gradio share link ציבורי")
    p.add_argument("--host",  default="127.0.0.1",    help="כתובת שרת (ברירת מחדל: 127.0.0.1)")
    args = p.parse_args()

    karaoke_app.app.queue(default_concurrency_limit=10).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[WORK_DIR],
    )


if __name__ == "__main__":
    main()
