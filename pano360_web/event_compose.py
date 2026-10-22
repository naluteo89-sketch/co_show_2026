"""
크롭 생성 영상(칠한 범위만 생성한 작은 영상) → 4K 360 영상 합성.
ComfyUI 에서 4K 프레임 수백 장을 메모리에 올리면 서버가 죽어서(8초·넓은 범위), 붙이기는 여기서 ffmpeg 로 한 프레임씩 한다.
바탕 = 8K 사진을 4K 로 줄인 정지 화면, 칠한 마스크(경계 부드럽게) 안만 영상. 소리는 크롭 영상의 것을 그대로 붙임.
"""

import re
import subprocess

import numpy as np
from PIL import Image

from cinemagraph import ffmpeg_exe


def video_size(path: str, exe: str | None = None) -> tuple[int, int]:
    exe = exe or ffmpeg_exe()
    out = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=60).stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out)
    if not m:
        raise RuntimeError(f"영상 크기를 못 읽었어요: {path}")
    return int(m.group(1)), int(m.group(2))


def _frame_count(path: str, exe: str, fps: int = 24) -> int:
    """프레임 수 ≈ 길이 × fps (못 읽으면 0 → 끝 페이드 안 함)."""
    out = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=60).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    if not m:
        return 0
    return int(round((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) * fps))


def front_shift(mask_path: str, width: int = 2048) -> int:
    """모든 사건은 정면 기준: 칠한 범위의 가로 중심이 정면(360 가운데)에 오도록 오른쪽으로 돌릴 픽셀 수
    (width 기준, 짝수). 원형 평균이라 좌우 끝에 걸친 범위도 맞음. 360 은 좌우가 이어져 있어서 돌려도 화질 손실이 없다."""
    m = np.asarray(Image.open(mask_path).convert("L").resize((width, width // 2), Image.BILINEAR), np.float32)
    col = m.sum(0)
    if col.sum() <= 0:
        return 0
    ang = np.arange(width) / width * 2 * np.pi
    c = np.arctan2((col * np.sin(ang)).sum(), (col * np.cos(ang)).sum()) % (2 * np.pi)
    cx = c / (2 * np.pi) * width
    sh = int(round((width / 2 - cx) / 2)) * 2
    return ((sh + width // 2) % width) - width // 2


def roll_image(path_in: str, path_out: str, shift: int, base_width: int = 2048, **save):
    """360 그림을 front_shift 만큼 돌려 저장 (그림 크기에 맞게 비례)."""
    im = Image.open(path_in)
    a = np.asarray(im)
    k = int(round(shift * a.shape[1] / base_width))
    Image.fromarray(np.roll(a, k, axis=1)).save(path_out, **save)
    return k


def compose_crop_video(crop_mp4: str, still_path: str, mask_path: str, crop: dict, out_mp4: str, *,
                       out_w: int = 4096, fps: int = 24, crf: int = 18, max_mbps: int = 40, audio: bool = True,
                       shift: int = 0, end_fade: float = 0.75) -> dict:
    """crop = workflow.crop_plan() 결과 ('mask' 2048 좌표, 'out' 4096 좌표).
    shift = front_shift() 값(2048 기준) — 생성 때 사진·마스크를 이만큼 돌렸으면 바탕 사진도 같은 값으로 돌림.
    (crop·mask 는 이미 돌린 좌표계라 영상 프레임은 그대로 붙임)
    end_fade 초: 마지막 구간에서 영상 → 원본 사진으로 서서히 바뀌어 마지막 프레임 = 첫 화면(뷰어가 사진으로 돌아갈 때 끊김 없음).
    노드도 마지막 프레임을 사진으로 잡아 두므로(1226) 이 구간은 거의 같은 그림끼리 섞임."""
    exe = ffmpeg_exe()
    W, H = int(out_w), int(out_w) // 2
    k = W / 4096
    ox, oy, ow, oh = (int(round(v * k)) for v in crop["out"])
    ow, oh = min(ow, W - ox), min(oh, H - oy)
    still = np.asarray(Image.open(still_path).convert("RGB").resize((W, H), Image.LANCZOS), np.uint8)
    if shift:
        still = np.roll(still, int(round(shift * W / 2048)), axis=1)
    mx, my, mw, mh = crop["mask"]
    alpha = np.asarray(Image.open(mask_path).convert("L").crop((mx, my, mx + mw, my + mh)).resize((ow, oh), Image.BILINEAR),
                       np.float32)[..., None] / 255.0
    roi = still[oy:oy + oh, ox:ox + ow].astype(np.float32)
    gw, gh = video_size(crop_mp4, exe)
    dec = subprocess.Popen([exe, "-v", "error", "-i", crop_mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-i", crop_mp4, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),            # 물보라는 비트레이트를 많이 먹어서 상한을 둠
            "-maxrate", f"{max_mbps}M", "-bufsize", f"{max_mbps * 2}M", "-pix_fmt", "yuv420p", "-g", str(fps * 2),
            "-movflags", "+faststart", out_mp4]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n, size = 0, gw * gh * 3
    total = _frame_count(crop_mp4, exe, fps)
    nf = max(0, int(round(end_fade * fps)))
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            f = np.asarray(Image.frombuffer("RGB", (gw, gh), buf).resize((ow, oh), Image.LANCZOS), np.float32)
            frame = still.copy()
            a = alpha
            if nf and total and n >= total - nf:                 # 끝: 첫 화면(사진)으로 서서히
                a = alpha * max(0.0, (total - 1 - n) / nf)
            frame[oy:oy + oh, ox:ox + ow] = np.clip(roi * (1 - a) + f * a + 0.5, 0, 255).astype(np.uint8)
            enc.stdin.write(frame.tobytes())
            n += 1
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg 실패: {err[-600:]}")
    finally:
        for p in (dec, enc):
            if p.poll() is None:
                p.kill()
    if n == 0:
        raise RuntimeError("크롭 영상에서 프레임을 못 읽었어요")
    return {"frames": n, "size": [W, H], "crop_size": [gw, gh], "shift": shift}
