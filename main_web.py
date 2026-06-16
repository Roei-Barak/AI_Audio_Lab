"""
Web UI entry point (Gradio).

Usage (local):
  python main_web.py

Usage (server with auth):
  python main_web.py --host 0.0.0.0 --port 7860 --no-browser --user admin --password secret
"""
import argparse
import os

import gradio as gr
import app as karaoke_app
from config import WORK_DIR


def main():
    p = argparse.ArgumentParser(description="AI_Audio_Lab — Web UI")
    p.add_argument("--port",       type=int, default=7860,      help="פורט (ברירת מחדל: 7860)")
    p.add_argument("--host",       default="127.0.0.1",          help="כתובת bind (0.0.0.0 לגישה מרחוק)")
    p.add_argument("--no-browser", action="store_true",           help="אל תפתח דפדפן אוטומטית (לשרת)")
    p.add_argument("--user",       default=None,                  help="שם משתמש לאימות בסיסי")
    p.add_argument("--password",   default=os.environ.get("KARAOKE_PASSWORD"),
                                                                  help="סיסמה (או KARAOKE_PASSWORD env var)")
    args = p.parse_args()

    auth = None
    if args.user and args.password:
        auth = (args.user, args.password)

    karaoke_app.app.queue(default_concurrency_limit=10).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(),
        allowed_paths=[WORK_DIR],
        auth=auth,
    )


if __name__ == "__main__":
    main()
