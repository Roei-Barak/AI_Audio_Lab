try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

_DEFAULT_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,100,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_to_dataframe(ass_path):
    """Parse ASS Dialogue lines into DataFrame(Start, End, Text)."""
    # Accept both string paths and objects with .name (Gradio file uploads)
    if ass_path is None:
        return pd.DataFrame(columns=["Start", "End", "Text"])
    path = ass_path.name if hasattr(ass_path, "name") else str(ass_path)

    data = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("Dialogue:"):
                    parts = line.split(",", 9)
                    if len(parts) == 10:
                        data.append({
                            "Start": parts[1].strip(),
                            "End":   parts[2].strip(),
                            "Text":  parts[9].strip(),
                        })
    except Exception:
        pass
    if _PANDAS:
        return pd.DataFrame(data, columns=["Start", "End", "Text"])
    return data


def dataframe_to_ass(df, original_ass_path, output_path: str) -> str:
    """Serialize edited DataFrame back to ASS, preserving original header."""
    header_lines = []
    orig = None
    if original_ass_path is not None:
        orig = original_ass_path.name if hasattr(original_ass_path, "name") else str(original_ass_path)

    if orig:
        try:
            with open(orig, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.startswith("Dialogue:"):
                        break
                    header_lines.append(line)
        except Exception:
            header_lines = []

    if not header_lines:
        header_lines = [_DEFAULT_HEADER]

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.writelines(header_lines)
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                f.write(f"Dialogue: 0,{row['Start']},{row['End']},Karaoke,,0,0,0,,{row['Text']}\n")

    return output_path
