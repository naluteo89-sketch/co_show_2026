"""
크롭 생성 영상(칠한 범위만 생성한 작은 영상) → 4K 360 영상 합성.
ComfyUI 에서 4K 프레임 수백 장을 메모리에 올리면 서버가 죽어서(8초·넓은 범위), 붙이기는 여기서 ffmpeg 로 한 프레임씩 한다.
바탕 = 8K 사진을 4K 로 줄인 정지 화면, 칠한 마스크(경계 부드럽게) 안만 영상. 소리는 크롭 영상의 것을 그대로 붙임.
"""

import subprocess

import numpy as np
from PIL import Image

from ..erp import roll_array
from .ffmpeg import ffmpeg_exe, probe


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
    a = np.asarray(Image.open(path_in))
    Image.fromarray(roll_array(a, shift, base_width)).save(path_out, **save)
    return int(round(shift * a.shape[1] / base_width))


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
        still = roll_array(still, shift)
    mx, my, mw, mh = crop["mask"]
    alpha = np.asarray(Image.open(mask_path).convert("L").crop((mx, my, mx + mw, my + mh)).resize((ow, oh), Image.BILINEAR),
                       np.float32)[..., None] / 255.0
    roi = still[oy:oy + oh, ox:ox + ow].astype(np.float32)
    info = probe(crop_mp4, exe)
    gw, gh = info["width"], info["height"]
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
    total = int(round(info["duration"] * fps))           # 프레임 수 ≈ 길이 × fps (못 읽으면 0 → 끝 페이드 안 함)
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


# ----------------------------------------------------------------------------- 평소 움직임 + 사건 = 끊김 없는 반복 영상
def _smooth(w: float) -> float:
    return w * w * (3 - 2 * w)


def _decode(path: str, size, exe: str):
    """영상 → RGB 프레임(uint8) 을 하나씩 (메모리에 다 올리지 않음)."""
    w, h = size
    p = subprocess.Popen([exe, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    try:
        n = w * h * 3
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if p.poll() is None:
            p.kill()


def _mix(a, b, w):
    return np.clip(a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w + 0.5, 0, 255).astype(np.uint8)


def live_plan(n_amb: int, n_evt: int, fps: int = 24, calm_seconds: float = 8.0, fade: int = 12) -> dict:
    """반복 영상 순서 계산 (프레임 수).
    평소 움직임 A(n_amb) 를 끝↔처음 교차 페이드로 끊김 없는 반복 Lp(L = n_amb - fade) 로 만들고,
    Lp 를 reps 번 이어 붙인 뒤 사건 E 로 교차 페이드, E 끝은 다시 Lp 처음으로 교차 페이드 → 파일 끝 = 파일 처음.
    파일은 Lp[fade] 에서 시작 (E→Lp 교차 페이드의 다음 프레임)."""
    fade = int(max(2, min(fade, n_amb // 4, n_evt // 4)))
    L = n_amb - fade
    if L <= 2 * fade:
        raise ValueError("평소 움직임 영상이 너무 짧아요")
    reps = max(1, -(-int(round(calm_seconds * fps + 2 * fade)) // L))     # 평소 구간이 calm_seconds 이상
    calm = reps * L - 2 * fade            # 사건이 섞여 들어오기 시작하는 프레임 (그 앞은 평소 움직임만)
    total = calm + n_evt
    return {"fade": fade, "loop_len": L, "reps": reps, "event_start": calm, "total": total,
            "calm_frames": calm, "fps": fps}


def build_live(amb_mp4: str, evt_mp4: str, out_mp4: str, *, fps: int = 24, calm_seconds: float = 8.0,
               fade: int = 12, crf: int = 12, audio: bool = True, duck: float = 0.35) -> dict:
    """크롭 공간에서: [평소 움직임 반복 × reps] → 사건 → 다시 평소 (끝 = 처음, 무한 반복해도 이음매 없음).
    소리: 평소 소리를 반복으로 깔고, 사건 소리를 사건 위치에 겹침 (사건 동안 평소 소리는 duck 배로 줄임)."""
    exe = ffmpeg_exe()
    size = video_size(amb_mp4, exe)
    if video_size(evt_mp4, exe) != size:
        raise ValueError("평소 움직임과 사건 영상의 크기가 달라요 (범위를 바꿨다면 둘 다 다시 만들어 주세요)")
    from collections import deque
    k0 = int(max(2, fade))
    head, tail, n_amb = [], deque(maxlen=k0), 0
    for f in _decode(amb_mp4, size, exe):                  # 실제 프레임 수 + 처음·끝 k 프레임
        if n_amb < k0:
            head.append(f.copy())
        tail.append(f.copy())
        n_amb += 1
    n_evt = sum(1 for _ in _decode(evt_mp4, size, exe))
    plan = live_plan(n_amb, n_evt, fps, calm_seconds, fade)
    k, L, reps = plan["fade"], plan["loop_len"], plan["reps"]
    tail = list(tail)[-k:]
    head = head[:k]

    def loop_frames(start: int, stop: int):
        """Lp[start:stop] — 처음 k 프레임은 A 끝과 교차 페이드 (반복 이음매)."""
        if start >= stop:
            return
        for i, f in enumerate(_decode(amb_mp4, size, exe)):
            if i >= stop:
                break
            if i < start:
                continue
            yield _mix(tail[i], f, _smooth((i + 1) / (k + 1))) if i < k else f

    cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", out_mp4 + ".v.mp4"]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        def put(f):
            nonlocal n
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
            n += 1
        segs = [(k, L)] + [(0, L)] * (reps - 1)            # 첫 반복은 Lp[k] 부터 (파일 처음 = 사건 끝 교차 페이드 다음)
        last_s, _ = segs[-1]
        for s, e in segs[:-1]:
            for f in loop_frames(s, e):
                put(f)
        lp_end = list(range(L - k, L))                      # 마지막 반복의 끝 k 프레임은 사건 처음과 섞음
        amb_last = {}
        for i, f in enumerate(loop_frames(last_s, L)):
            idx = last_s + i
            if idx < L - k:
                put(f)
            else:
                amb_last[idx] = f.copy()
        evt_tail = deque()
        for j, f in enumerate(_decode(evt_mp4, size, exe)):
            if j < k:
                put(_mix(amb_last[lp_end[j]], f, _smooth((j + 1) / (k + 1))))
                continue
            evt_tail.append(f.copy())
            if len(evt_tail) > k:
                put(evt_tail.popleft())
        evt_tail = list(evt_tail)
        for j, f in enumerate(evt_tail):                     # 사건 끝 → Lp 처음 (= 파일 처음 바로 앞)
            put(_mix(f, _mix(tail[j], head[j], _smooth((j + 1) / (k + 1))), _smooth((j + 1) / (k + 1))))
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg 실패: {err[-600:]}")
    finally:
        if enc.poll() is None:
            enc.kill()
    dur = n / fps
    t_evt = plan["event_start"] / fps
    d_evt = n_evt / fps
    import os
    if audio:
        flt = (f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,"
               f"volume='if(between(t,{t_evt:.3f},{t_evt + d_evt:.3f}),{duck},1)':eval=frame[a0];"
               f"[2:a]adelay={int(t_evt * 1000)}|{int(t_evt * 1000)},apad[a1];"
               f"[a0][a1]amix=inputs=2:duration=first:normalize=0[a]")
        r = subprocess.run([exe, "-y", "-v", "error", "-i", out_mp4 + ".v.mp4", "-stream_loop", "-1", "-i", amb_mp4,
                            "-i", evt_mp4, "-filter_complex", flt, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", out_mp4],
                            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:                                 # 소리가 없는 영상이면 영상만
            os.replace(out_mp4 + ".v.mp4", out_mp4)
        else:
            os.remove(out_mp4 + ".v.mp4")
    else:
        os.replace(out_mp4 + ".v.mp4", out_mp4)
    plan.update(frames=n, seconds=round(dur, 2), event_at=round(t_evt, 2))
    return plan
