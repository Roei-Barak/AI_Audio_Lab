try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except ImportError:
    BIDI_AVAILABLE = False


def fix_hebrew_text(text: str) -> str:
    if not BIDI_AVAILABLE or not text:
        return text
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return text


def apply_bidi_to_ass(ass_path: str, output_path: str) -> str:
    """Read ASS, apply BIDI fix to every Dialogue text field, write to output_path."""
    with open(ass_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    fixed = []
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.rstrip("\n").split(",", 9)
            if len(parts) == 10:
                parts[9] = fix_hebrew_text(parts[9])
                line = ",".join(parts) + "\n"
        fixed.append(line)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.writelines(fixed)

    return output_path
