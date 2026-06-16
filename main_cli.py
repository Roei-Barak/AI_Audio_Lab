"""
CLI entry point.

Usage:
  python main_cli.py --url "https://youtube.com/watch?v=..."   --lang he
  python main_cli.py --url "שם השיר"                           --lang he --bidi
  python main_cli.py --file song.mp4                           --lang en --stems 4
"""
import sys
from interfaces.cli import main

if __name__ == "__main__":
    sys.exit(main())
