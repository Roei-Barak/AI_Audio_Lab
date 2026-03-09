"""
gui/desktop_app.py — Desktop launcher for the Karaoke Studio Pro UI.

Starts the Gradio web server in a background thread and opens the browser
automatically.  Optionally embeds the UI in a PyQt6 WebView window
(requires PyQt6 and PyQt6-WebEngine to be installed).

Usage:
    python gui/desktop_app.py              # browser-based desktop mode
    python gui/desktop_app.py --webview    # embedded PyQt6 WebView (if installed)
    python gui/desktop_app.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Browser-based launcher (no extra dependencies)
# ---------------------------------------------------------------------------

def launch_browser_mode(port: int = 7860, share: bool = False) -> None:
    """Start Gradio in a thread then open the default browser."""
    from gui.gradio_app import build_app

    url = f"http://127.0.0.1:{port}"
    app = build_app()

    # Start Gradio server in background thread
    def _serve() -> None:
        app.queue(default_concurrency_limit=10).launch(
            server_name="127.0.0.1",
            server_port=port,
            share=share,
            inbrowser=False,   # we open the browser ourselves
            quiet=True,
        )

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    # Wait for server to be ready
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    print(f"🌐 פותח דפדפן: {url}")
    webbrowser.open(url)

    # Keep the process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 סגור")


# ---------------------------------------------------------------------------
# PyQt6 WebView launcher (optional — richer desktop experience)
# ---------------------------------------------------------------------------

def launch_webview_mode(port: int = 7860) -> None:
    """Embed the Gradio UI inside a PyQt6 QWebEngineView window."""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtCore import QUrl
    except ImportError:
        print(
            "❌ PyQt6 ו-PyQt6-WebEngine נדרשים למצב חלון.\n"
            "   pip install PyQt6 PyQt6-WebEngine\n"
            "   חלופית, השתמש במצב ברירת מחדל (דפדפן).",
            file=sys.stderr,
        )
        print("ממשיך במצב דפדפן…")
        launch_browser_mode(port)
        return

    from gui.gradio_app import build_app

    app_qt = QApplication(sys.argv)
    app_qt.setApplicationName("Karaoke Studio Pro")

    # Start Gradio server in background
    gr_app = build_app()
    url = f"http://127.0.0.1:{port}"

    def _serve() -> None:
        gr_app.queue(default_concurrency_limit=10).launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=False,
            quiet=True,
        )

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    # Wait for server
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    # Create window
    window = QMainWindow()
    window.setWindowTitle("🎤 Karaoke Studio Pro")
    window.resize(1280, 800)

    view = QWebEngineView()
    view.setUrl(QUrl(url))
    window.setCentralWidget(view)
    window.show()

    sys.exit(app_qt.exec())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="🎤 Karaoke Studio Pro — Desktop Launcher",
    )
    parser.add_argument(
        "--webview", action="store_true",
        help="הפעל בחלון PyQt6 WebView (דורש PyQt6-WebEngine)",
    )
    parser.add_argument("--port", type=int, default=7860, help="פורט (ברירת מחדל: 7860)")
    parser.add_argument("--share", action="store_true", help="צור קישור ציבורי Gradio")
    args = parser.parse_args()

    print("🎤 Karaoke Studio Pro — מפעיל…")

    if args.webview:
        launch_webview_mode(port=args.port)
    else:
        launch_browser_mode(port=args.port, share=args.share)


if __name__ == "__main__":
    main()
