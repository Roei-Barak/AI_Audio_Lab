from backend import BackendProcessor


def step_render(
    video_path,
    playback_path: str,
    ass_path: str,
    output_dir: str,
    task_log,
) -> str:
    """
    Render karaoke video with burned-in subtitles via FFmpeg.
    video_path may be None → black background is used.

    Returns output video path.
    Raises ValueError on failure.
    """
    proc = BackendProcessor(log_func=task_log)
    result = proc.render_custom_karaoke(video_path, playback_path, ass_path, output_dir)
    if not result:
        raise ValueError("רינדור וידאו נכשל")
    return result
