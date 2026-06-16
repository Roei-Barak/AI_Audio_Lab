"""
Web UI entry point (Gradio).

Usage (local, no auth):
  python main_web.py

Usage (server with user-file auth):
  python main_web.py --host 0.0.0.0 --port 7860 --no-browser

Manage users:
  python auth/manage_users.py
"""
import argparse
import os

import gradio as gr
import app as karaoke_app
from auth.auth import check_credentials, list_users
from config import WORK_DIR


def main():
    p = argparse.ArgumentParser(description="AI_Audio_Lab — Web UI")
    p.add_argument("--port",       type=int, default=7860, help="פורט (ברירת מחדל: 7860)")
    p.add_argument("--host",       default="127.0.0.1",    help="כתובת bind (0.0.0.0 לגישה מרחוק)")
    p.add_argument("--no-browser", action="store_true",    help="אל תפתח דפדפן (לשרת)")
    p.add_argument("--no-auth",    action="store_true",    help="הפעל בלי אימות (לפיתוח מקומי)")
    args = p.parse_args()

    # Auth: disabled only when --no-auth is explicit OR users.json is empty
    approved = list_users()
    use_auth = not args.no_auth and len(approved) > 0

    if use_auth:
        print(f"[auth] הגנה פעילה — {len(approved)} משתמשים מאושרים")
        auth_kwargs = dict(
            auth=check_credentials,
            auth_message="🎤 AI_Audio_Lab — נא להתחבר עם פרטי הגישה שלך",
        )
    else:
        if not args.no_auth:
            print("[auth] ⚠️  אין משתמשים בקובץ — גישה חופשית (הרץ: python auth/manage_users.py)")
        auth_kwargs = {}

    karaoke_app.app.queue(default_concurrency_limit=10).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(),
        allowed_paths=[WORK_DIR],
        **auth_kwargs,
    )


if __name__ == "__main__":
    main()
