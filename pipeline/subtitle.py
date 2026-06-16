import os
import re

from utils.bidi_support import apply_bidi_to_ass


def step_apply_style(ass_path: str, font_size: int, color_hex: str) -> str:
    """Update ASS subtitle style in-place. Returns ass_path."""
    if color_hex.startswith("#") and len(color_hex) == 7:
        r, g, b = color_hex[1:3], color_hex[3:5], color_hex[5:7]
        ass_color = f"&H00{b}{g}{r}".upper()
    else:
        ass_color = "&H0000FFFF"

    new_style = (
        f"Style: Karaoke,Arial,{int(font_size)},{ass_color},"
        "&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1"
    )

    with open(ass_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    content = re.sub(r"^Style:.*$", new_style, content, flags=re.MULTILINE)

    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(content)

    return ass_path


def step_apply_bidi(ass_path: str) -> str:
    """Apply Hebrew BIDI fix. Returns path to new BIDI-fixed ASS file."""
    out = ass_path.replace(".ass", "_bidi.ass")
    return apply_bidi_to_ass(ass_path, out)
