"""ffmpeg 찾기 · 영상 정보 읽기."""

import re
import shutil
import subprocess


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg 를 찾지 못했어요. `pip install imageio-ffmpeg` 를 하거나 ffmpeg 를 PATH 에 넣어 주세요.")


def probe(path: str, exe: str | None = None) -> dict:
    """영상 크기(width, height)와 길이(duration 초, 못 읽으면 0)."""
    exe = exe or ffmpeg_exe()
    out = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=60).stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out)
    if not m:
        raise RuntimeError(f"영상 크기를 못 읽었어요: {path}")
    d = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else 0.0
    return {"width": int(m.group(1)), "height": int(m.group(2)), "duration": dur}
